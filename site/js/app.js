// =============================================
// 焚诀 FlameSutra - 核心脚本 v2.0
// 双模式(修仙/学术) + 代码折叠 + Pyodide运行
// =============================================

const state = { currentPage: null, searchQuery: '', sidebarOpen: false, searchOpen: false, viewMode: 'xiuxian', pyodideReady: false, pyodide: null, kgExpanded: false };

const SITE_DATA = window.__SITE_DATA__ || {};
const ALL_PAGES = SITE_DATA.pages || [];
const HOME_DATA = SITE_DATA.home || {};

document.addEventListener('DOMContentLoaded', () => {
  initRouter(); initSidebar(); initSearch(); initScrollTop();
  initProgress(); initViewModeToggle(); initPyodide();
});

// ============ ROUTER ============
function initRouter() {
  const hash = window.location.hash.slice(1) || 'home';
  navigateTo(hash);
  window.addEventListener('hashchange', () => {
    const h = window.location.hash.slice(1) || 'home';
    if (h !== state.currentPage) navigateTo(h);
  });
}

async function navigateTo(pageId) {
  // Decode URI-encoded hash (e.g. %E5%BC%82%E7%81%AB → 异火)
  try { pageId = decodeURIComponent(pageId); } catch (e) {}

  // Check if this is an in-page anchor (not a known page ID)
  const isKnownPage = pageId === 'home' || ALL_PAGES.some(p => p.id === pageId);

  if (!isKnownPage) {
    // Try to find this ID as a heading within the current page
    // Strategy: exact match → collapsed-hyphen match → text-content search
    let el = document.getElementById(pageId);

    if (!el) {
      // Try with collapsed hyphens (--- → --, etc.)
      const collapsed = pageId.replace(/-{2,}/g, '--');
      if (collapsed !== pageId) el = document.getElementById(collapsed);
    }

    if (!el) {
      // Try normalizing: collapse all multi-hyphens to single
      const normalized = pageId.replace(/-+/g, '-');
      el = document.getElementById(normalized);
    }

    if (!el) {
      // Fuzzy: search all headings for one whose slugified text matches
      const target = pageId.replace(/-+/g, '-').toLowerCase();
      const headings = document.querySelectorAll('.md-content h1, .md-content h2, .md-content h3, .md-content h4');
      for (const h of headings) {
        const hSlug = (h.id || '').replace(/-+/g, '-').toLowerCase();
        if (hSlug === target) { el = h; break; }
      }
    }

    if (el) {
      // It's an in-page anchor — scroll to it, expand its section card if needed
      const card = el.closest('.section-card');
      if (card) expandSectionCard(card);
      const offset = 90;
      const pos = el.getBoundingClientRect().top + window.pageYOffset - offset;
      window.scrollTo({ top: pos, behavior: 'smooth' });
      // Restore the correct page hash (keep current page in URL)
      if (state.currentPage && state.currentPage !== pageId) {
        history.replaceState(null, '', `#${state.currentPage}`);
      }
      return;
    }
    // Not found in current page either — show 404
    const contentArea = document.getElementById('content');
    contentArea.innerHTML = `<div class="md-content"><h1>404</h1><p>未知页面: ${pageId}</p><a href="#home">返回总览</a></div>`;
    return;
  }

  state.currentPage = pageId;
  state.searchOpen = false;
  document.querySelector('.search-results')?.classList.remove('show');
  closeSidebar();

  document.querySelectorAll('.sidebar-item').forEach(el => el.classList.toggle('active', el.dataset.page === pageId));
  document.querySelectorAll('.header-nav a').forEach(el => el.classList.toggle('active', el.dataset.page === pageId));

  const contentArea = document.getElementById('content');

  if (pageId === 'home') { renderHome(contentArea); }
  else {
    const page = ALL_PAGES.find(p => p.id === pageId);
    contentArea.innerHTML = '<div class="loading-spinner">加载中</div>';
    try {
      const resp = await fetch(`pages/${pageId}.html`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const md = await resp.text();
      renderMarkdownPage(contentArea, md, page);
    } catch (e) {
      console.error('Fetch/render error:', e);
      contentArea.innerHTML = `<div class="md-content"><h1>404 - 页面未找到</h1><p>该卷的修炼内容尚未加载。</p><a href="#home">返回总览</a></div>`;
    }
  }

  // Instant scroll to top on page change (no smooth — avoids visible roll-back lag)
  window.scrollTo({ top: 0, behavior: 'instant' });
}

function renderMarkdownPage(container, md, pageMeta) {
  marked.setOptions({ gfm: true, breaks: false });
  let html;
  try {
    html = DOMPurify.sanitize(marked.parse(md));
  } catch (e) {
    console.error('Markdown parse error:', e);
    html = `<pre style="color:red;white-space:pre-wrap;">解析错误: ${e.message}</pre>`;
  }

  const idx = ALL_PAGES.findIndex(p => p.id === pageMeta.id);
  const prevPage = idx > 0 ? ALL_PAGES[idx - 1] : null;
  const nextPage = idx < ALL_PAGES.length - 1 ? ALL_PAGES[idx + 1] : null;

  const breadcrumb = `
    <div class="breadcrumb">
      <a href="#home">焚诀</a>
      ${pageMeta.category ? `<span>/</span><a href="#${pageMeta.category}">${pageMeta.categoryLabel || pageMeta.category}</a><span>/</span>` : ''}
      <span class="current">${pageMeta.title}</span>
    </div>`;

  const pageNav = `
    <div class="page-nav">
      ${prevPage ? `<a class="page-nav-btn prev" href="#${prevPage.id}"><div class="nav-label">← 上一卷</div><div class="nav-title">${prevPage.title}</div></a>` : '<div></div>'}
      ${nextPage ? `<a class="page-nav-btn next" href="#${nextPage.id}"><div class="nav-label">下一卷 →</div><div class="nav-title">${nextPage.title}</div></a>` : '<div></div>'}
    </div>`;

  container.innerHTML = `${breadcrumb}<div class="md-content">${html}</div>${pageNav}`;

  // Auto-assign IDs to all headings (marked.js v9 does NOT generate heading IDs)
  // This enables in-page anchor links from markdown TOCs like [title](#anchor)
  container.querySelectorAll('.md-content h1, .md-content h2, .md-content h3, .md-content h4, .md-content h5, .md-content h6').forEach(h => {
    if (!h.id) {
      h.id = slugifyHeading(h.textContent);
    }
  });

  // === Knowledge Graph + Section Splitting (BEFORE code block wrapping) ===
  // Order matters: split sections first, then wrap code blocks inside the new structure
  initKnowledgeGraphAndSections(container, pageMeta);

  // Process code blocks: wrap with collapsible header (after section splitting)
  wrapCodeBlocks(container);

  // Build TOC (after all DOM restructuring)
  buildTOC(container.querySelector('.md-content'));
}

// ============ CODE BLOCK WRAPPER (d2l-style collapsible) ============
function wrapCodeBlocks(container) {
  const preElements = container.querySelectorAll('.md-content pre');
  preElements.forEach((pre) => {
    // Skip if already wrapped
    if (pre.closest('.code-block-wrapper')) return;

    const codeEl = pre.querySelector('code');
    if (!codeEl) return;

    // Safety check: ensure pre has a valid parentNode and we won't create a cycle
    const parent = pre.parentNode;
    if (!parent || parent === pre || parent.contains(pre) === false) return;
    // Additional guard: if pre is already inside a code-body, skip (splitIntoSectionCards may have moved it)
    if (pre.closest('.section-card-body') && pre.closest('.code-block-wrapper')) return;

    const codeText = codeEl.textContent || '';
    const lines = codeText.split('\n').length;

    // Detect language from first line comment or class
    let lang = 'python';
    const clsList = Array.from(pre.classList);
    for (const c of clsList) {
      if (c.startsWith('language-') || c.startsWith('lang-')) {
        lang = c.replace(/^(language-|lang-)/, '');
        break;
      }
    }
    // Also check code element
    if (lang === 'python' && codeEl.className) {
      const cm = codeEl.className.match(/language-(\w+)/);
      if (cm) lang = cm[1];
    }

    // Create wrapper div
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper collapsed';

    // Extract first meaningful line for preview
    const firstLine = codeText.split('\n').find(l => l.trim().length > 0) || '';
    const preview = firstLine.length > 60 ? firstLine.substring(0, 60) + '…' : firstLine;

    // Header bar
    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `
      <div class="code-header-left">
        <span class="code-collapse-icon">▼</span>
        <span class="code-lang">${lang}</span>
        <span class="code-lines-info">${lines} 行</span>
        <span class="code-preview">${preview.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
      </div>
      <div class="code-header-right">
        <button class="code-btn run-btn" data-action="run">▶ 运行</button>
        <button class="code-btn" data-action="copy">复制</button>
      </div>`;

    // Body (contains original pre)
    const body = document.createElement('div');
    body.className = 'code-body';
    body.style.maxHeight = '600px'; // max expanded height
    body.appendChild(pre);

    // Output area
    const output = document.createElement('div');
    output.className = 'code-output';

    wrapper.appendChild(header);
    wrapper.appendChild(body);
    wrapper.appendChild(output);

    // Final safety: verify parent still contains pre and wrapper is not a descendant of pre
    if (pre.parentNode !== parent || parent.contains(wrapper)) {
      // DOM state changed mid-operation; abort this block silently
      return;
    }
    parent.replaceChild(wrapper, pre);

    // Click header to toggle collapse
    header.addEventListener('click', (e) => {
      // Don't toggle if clicking a button
      if (e.target.closest('.code-btn')) return;
      wrapper.classList.toggle('collapsed');
    });

    // Button handlers
    header.querySelector('[data-action="copy"]').addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        await navigator.clipboard.writeText(codeText);
        const btn = e.target.closest('.code-btn');
        const orig = btn.textContent;
        btn.textContent = '✓ 已复制';
        setTimeout(() => btn.textContent = orig, 1500);
      } catch (err) { console.error('Copy failed:', err); }
    });

    header.querySelector('[data-action="run"]').addEventListener('click', (e) => {
      e.stopPropagation();
      runCode(wrapper, codeText);
    });
  });
}

// ============ PYODIDE ENGINE ============
async function initPyodide() {
  // Lazy-load Pyodide on first run button click
  window.loadPyodideAndRun = async function(code, outputEl, runBtn) {
    if (state.pyodideReady && state.pyodide) {
      executePyodideCode(code, outputEl);
      return;
    }

    showPyodideLoading(true);
    try {
      state.pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
      });
      // Pre-install common packages
      await state.pyodide.loadPackage(["numpy", "matplotlib"]);
      state.pyodideReady = true;
      showPyodideLoading(false);
      executePyodideCode(code, outputEl);
    } catch (err) {
      showPyodideLoading(false);
      outputEl.textContent = `❌ 加载 Python 引擎失败: ${err.message}\n\n提示：首次使用需要下载约 20MB 的 WASM 运行时，请确保网络畅通。`;
      outputEl.className = 'code-output show error';
      runBtn.classList.remove('running');
    }
  };
}

function showPyodideLoading(show) {
  let overlay = document.getElementById('pyodide-overlay');
  if (show) {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'pyodide-overlay';
      overlay.className = 'pyodide-overlay';
      overlay.innerHTML = `
        <div class="pyodide-spinner"></div>
        <div class="pyodide-overlay-text">🔥 正在加载 Python 引擎（首次需下载 ~20MB）...</div>
        <div class="pyodide-overlay-text" style="font-size:.82rem;color:#aaa;">Pyodide — 浏览器端 Python 运行环境</div>`;
      document.body.appendChild(overlay);
    }
    overlay.classList.add('show');
  } else if (overlay) {
    overlay.classList.remove('show');
  }
}

function executePyodideCode(code, outputEl) {
  outputEl.className = 'code-output show';
  outputEl.textContent = '';

  // Redirect stdout/stderr
  state.pyodide.setStdout({ batched: (text) => {
    outputEl.textContent += text;
    outputEl.classList.add('success');
  }});
  state.pyodide.setStderr({ batched: (text) => {
    outputEl.textContent += text;
    outputEl.classList.add('error');
  }});

  try {
    // Run the code in a timeout context
    const result = state.pyodide.runPythonAsync(`
import sys
from io import StringIO

# Capture output
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = StringIO()
sys.stderr = StringIO()

try:
${code.split('\n').map(l => '    ' + l).join('\n')}
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)

_out = sys.stdout.getvalue()
_err = sys.stderr.getvalue()
sys.stdout = _old_stdout
sys.stderr = _old_stderr

if _out: print(_out, end='')
if _err:
    import warnings
    warnings.warn(_err)
`);
    result.then(() => {}).catch((err) => {
      outputEl.textContent += `\n❌ 执行异常: ${err.message}`;
      outputEl.classList.add('error');
    });
  } catch (err) {
    outputEl.textContent = `❌ 执行失败: ${err.message}`;
    outputEl.classList.add('error');
  }
}

function runCode(wrapper, code) {
  const outputEl = wrapper.querySelector('.code-output');
  const runBtn = wrapper.querySelector('.code-btn.run-btn');

  // Clear previous output
  outputEl.textContent = '';
  outputEl.className = 'code-output';

  // Show running state
  runBtn.classList.add('running');

  // Ensure expanded
  wrapper.classList.remove('collapsed');

  window.loadPyodideAndRun(code, outputEl, runBtn).then(() => {
    runBtn.classList.remove('running');
  }).catch(() => {
    runBtn.classList.remove('running');
  });
}

// ============ VIEW MODE TOGGLE (修仙暗色 ⇄ 学术亮色 ⇄ 自动) ============
const VIEW_MODES = ['xiuxian', 'academic', 'auto'];
const MODE_LABELS = { xiuxian: { icon: '⚔️', text: '修仙', title: '暗色修仙风格' }, academic: { icon: '📖', text: '学术', title: '亮色学术风格' }, auto: { icon: '🌓', text: '自动', title: '跟随系统亮/暗色' } };

function initViewModeToggle() {
  const btn = document.getElementById('mode-toggle');
  if (!btn) return;

  const saved = localStorage.getItem('flamesutra-mode') || 'xiuxian';
  state.viewMode = saved;
  applyViewMode(saved);

  btn.addEventListener('click', () => {
    const currentIdx = VIEW_MODES.indexOf(state.viewMode);
    const nextMode = VIEW_MODES[(currentIdx + 1) % VIEW_MODES.length];
    state.viewMode = nextMode;
    applyViewMode(nextMode);
    localStorage.setItem('flamesutra-mode', nextMode);
  });

  // Listen for system color scheme changes (only affects 'auto' mode)
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (state.viewMode === 'auto') applyViewMode('auto');
  });

  document.addEventListener('keydown', (e) => {
    if (e.shiftKey && e.key === 'M') { e.preventDefault(); btn.click(); }
  });
}

function applyViewMode(mode) {
  let effectiveMode = mode;
  if (mode === 'auto') {
    effectiveMode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'xiuxian' : 'academic';
  }

  // Set data-mode: empty string = xiuxian (default dark), 'academic' = light
  document.documentElement.setAttribute('data-mode', effectiveMode === 'xiuxian' ? '' : effectiveMode);

  const btn = document.getElementById('mode-toggle');
  if (!btn) return;

  const icon = btn.querySelector('.mode-icon');
  const text = btn.querySelector('.mode-text');
  const info = MODE_LABELS[mode];

  if (icon) icon.textContent = info.icon;
  if (text) text.textContent = info.text;
  btn.title = `${info.title} → 点击切换 (⇧+M)`;
}

// ============ TOC ============
let tocObserver = null;

window.scrollToHeading = function(id, event) {
  if (event) { event.preventDefault(); event.stopPropagation(); }
  const el = document.getElementById(id);
  if (el) {
    const headerOffset = 80;
    const elementPosition = el.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
    window.scrollTo({ top: offsetPosition, behavior: "smooth" });
    document.querySelectorAll('.toc-item').forEach(item => item.classList.remove('active'));
    const targetItem = document.querySelector(`.toc-item[data-target="${id}"]`);
    if (targetItem) targetItem.classList.add('active');
  }
};

function buildTOC(mdContent) {
  const tocPanel = document.getElementById('toc-panel');
  if (!mdContent) return;
  if (tocObserver) tocObserver.disconnect();

  const headings = mdContent.querySelectorAll('h2, h3, h4');
  if (headings.length < 3) {
    tocPanel.classList.remove('show');
    document.querySelector('.content-area')?.classList.remove('has-toc');
    return;
  }

  let tocHtml = '<div class="toc-title">本卷目录</div>';
  headings.forEach((h, i) => {
    const id = h.id || `heading-${i}`;
    if (!h.id) h.id = id;
    const cls = h.tagName === 'H3' ? 'toc-h3' : h.tagName === 'H4' ? 'toc-h4' : '';
    tocHtml += `<div class="toc-item ${cls}" data-target="${id}" style="cursor:pointer;" onclick="scrollToHeading('${id}', event)">${h.textContent}</div>`;
  });

  tocPanel.innerHTML = tocHtml;
  tocPanel.classList.add('show');
  document.querySelector('.content-area')?.classList.add('has-toc');

  tocObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        document.querySelectorAll('.toc-item').forEach(item => item.classList.remove('active'));
        const activeItem = document.querySelector(`.toc-item[data-target="${id}"]`);
        if (activeItem) {
          activeItem.classList.add('active');
          const panelHeight = tocPanel.clientHeight;
          const itemTop = activeItem.offsetTop;
          if (itemTop < tocPanel.scrollTop || itemTop > tocPanel.scrollTop + panelHeight - 30) {
            tocPanel.scrollTo({ top: itemTop - panelHeight / 2, behavior: "smooth" });
          }
        }
      }
    });
  }, { root: null, rootMargin: '-80px 0px -60% 0px', threshold: 0 });

  headings.forEach(h => tocObserver.observe(h));
}

// ============ SIDEBAR ============
function initSidebar() {
  const btn = document.querySelector('.mobile-menu-btn');
  const overlay = document.querySelector('.sidebar-overlay');
  if (btn) btn.addEventListener('click', toggleSidebar);
  if (overlay) overlay.addEventListener('click', closeSidebar);
}
function toggleSidebar() {
  state.sidebarOpen = !state.sidebarOpen;
  document.querySelector('.sidebar')?.classList.toggle('open', state.sidebarOpen);
  document.querySelector('.sidebar-overlay')?.classList.toggle('show', state.sidebarOpen);
}
function closeSidebar() {
  state.sidebarOpen = false;
  document.querySelector('.sidebar')?.classList.remove('open');
  document.querySelector('.sidebar-overlay')?.classList.remove('show');
}

// ============ SEARCH ============
function initSearch() {
  const input = document.querySelector('.search-input');
  if (!input) return;
  let debounceTimer;
  input.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { performSearch(e.target.value.trim()); }, 200);
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.header-search') && !e.target.closest('.search-results')) {
      document.querySelector('.search-results')?.classList.remove('show');
    }
  });
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); input.focus(); }
    if (e.key === 'Escape') { input.blur(); document.querySelector('.search-results')?.classList.remove('show'); }
  });
}

async function performSearch(query) {
  const resultsPanel = document.querySelector('.search-results');
  if (!query) { resultsPanel?.classList.remove('show'); return; }
  resultsPanel.classList.add('show');
  resultsPanel.innerHTML = '<div class="loading-spinner" style="height:100px;">搜索中</div>';

  const results = [], q = query.toLowerCase();
  for (const page of ALL_PAGES) {
    try {
      const resp = await fetch(`pages/${page.id}.html`);
      if (!resp.ok) continue;
      const text = await resp.text(), lines = text.split('\n');
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].toLowerCase().includes(q) && lines[i].trim().length > 10) {
          const start = Math.max(0, i - 1), end = Math.min(lines.length, i + 3);
          const ctx = lines.slice(start, end).join('\n').substring(0, 200);
          if (results.length < 20 && results.filter(r => r.page.id === page.id).length < 3) {
            results.push({ page, line: lines[i].substring(0, 150), context: ctx, lineNumber: i + 1 });
          }
          if (results.filter(r => r.page.id === page.id).length >= 3) break;
        }
      }
    } catch (e) {}
  }

  if (!results.length) {
    resultsPanel.innerHTML = `<div style="padding:20px;color:var(--text-muted);text-align:center;">未找到相关内容</div>`; return;
  }

  resultsPanel.innerHTML = results.map(r => {
    const hl = r.line.replace(new RegExp(`(${escapeRegex(q)})`, 'gi'), '<mark>$1</mark>');
    return `<div class="search-result-item" onclick="window.location.hash='${r.page.id}';this.closest('.search-results').classList.remove('show');">
      <div class="result-title">${r.page.title}</div>
      <div class="result-preview">${hl}</div>
      <div class="result-context">第 ${r.lineNumber} 行</div></div>`;
  }).join('');
}

function escapeRegex(str) { return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

// ============ SCROLL UTILS ============
function initScrollTop() {
  const btn = document.querySelector('.scroll-top');
  if (!btn) return;
  window.addEventListener('scroll', () => btn.classList.toggle('show', window.scrollY > 400));
  btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
}
function initProgress() {
  const bar = document.querySelector('.progress-bar');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const docH = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = docH > 0 ? `${(window.scrollY / docH) * 100}%` : '0%';
  });
}

// ============ HOME RENDERING ============
function renderHome(container) {
  const totalPages = ALL_PAGES.length;
  const totalLines = ALL_PAGES.reduce((s, p) => s + (p.lines || 0), 0);
  const appendixCount = ALL_PAGES.filter(p => p.isAppendix).length;
  const mainCount = totalPages - appendixCount;

  container.innerHTML = `
    <div class="md-content">
      <div class="home-hero">
        <h1 class="home-title">焚 诀</h1>
        <p class="home-subtitle">A Cultivation Guide for Foundation Models: From Dust to Deity</p>
        <div class="home-quote">"三十年河东，三十年河西，莫欺算法穷！"</div>
        <div class="home-stats">
          <div class="stat-item"><div class="stat-value">${mainCount}</div><div class="stat-label">正篇卷数</div></div>
          <div class="stat-item"><div class="stat-value">${totalLines.toLocaleString()}</div><div class="stat-label">总行数</div></div>
          <div class="stat-item"><div class="stat-value">${appendixCount}</div><div class="stat-label">辅助典籍</div></div>
          <div class="stat-item"><div class="stat-value">10</div><div class="stat-label">修炼境界</div></div>
        </div>
      </div>
      ${renderRealmPyramid()}
      <div class="appendix-section"><h2 class="appendix-title">📖 正篇十卷</h2>
        <div class="volume-grid">${ALL_PAGES.filter(p => !p.isAppendix).map(p => renderVolumeCard(p)).join('')}</div>
      </div>
      <div class="appendix-section"><h2 class="appendix-title">📜 辅助典籍</h2>
        <div class="appendix-grid">${ALL_PAGES.filter(p => p.isAppendix).map(p => renderAppendixCard(p)).join('')}</div>
      </div>
      <div class="appendix-section" style="margin-top:40px;"><h2 class="appendix-title">🌍 术语对照</h2>
        <table><thead><tr><th>玄幻术语</th><th>技术含义</th></tr></thead>
        <tbody>
          <tr><td>焚诀</td><td>本学习手册 / 技术体系</td></tr>
          <tr><td>异火 (Strange Fire)</td><td>GPU / 算力芯片</td></tr>
          <tr><td>丹炉 (Boiler)</td><td>服务器 / 计算集群</td></tr>
          <tr><td>药材 (Herb)</td><td>原始数据集</td></tr>
          <tr><td>丹方 (Recipe)</td><td>训练脚本 / Model Code</td></tr>
          <tr><td>炼丹 (Refining)</td><td>模型训练 (Training)</td></tr>
          <tr><td>反噬 (Backlash)</td><td>Loss NaN / OOM / 崩溃</td></tr>
          <tr><td>斗气 (Dou Qi)</td><td>模型参数 / 权重</td></tr>
          <tr><td>斗技 (Combat Skill)</td><td>算法 / 模型架构</td></tr>
        </tbody></table>
      </div>
    </div>
    <div class="site-footer">
      <p>《焚诀》FlameSutra — 大模型修炼指南 | Apache License 2.0</p>
      <p style="margin-top:8px;">以玄幻修仙之喻，讲解 Foundation Model 全栈技术 | 支持 Pyodide 在线运行 Python</p>
    </div>`;
}

function renderRealmPyramid() {
  return `<div style="text-align:center;margin:40px 0;font-family:var(--font-mono);font-size:.72rem;color:var(--text-muted);line-height:2;overflow-x:auto;">
    <pre style="display:inline-block;text-align:left;">
┌──────────────────────────────────────────────────┐
│ 第十卷·帝境篇（斗帝）│AGI/自进化/大统一架构     │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第九卷·成圣篇（斗圣）│MoE/FlashAttention/Megatron │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第八卷·九转篇（斗尊）│RLHF/PPO/DPO/安全对齐       │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第七卷·空间篇（斗宗）│VLM多模态/CLIP/LLaVA         │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第六卷·凌空篇（斗皇）│SFT/数据清洗/RAG             │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第五卷·化翼篇（斗王）│分布式训练/DeepSpeed/混合精度  │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第四卷·化形篇（斗灵）│PEFT/LoRA/量化/Prompt工程     │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第三卷·凝晶篇（大斗师）│BERT/GPT-2/Tokenizer       │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第二卷·纳灵篇（斗者-斗师）│CNN/RNN/Transformer      │
└─────────────────┬────────────────────────────────┘
┌─────────────────┴────────────────────────────────┐
│ 第一卷·筑基篇（斗之气）│Python/数学/PyTorch基础    │
└──────────────────────────────────────────────────┘</pre></div>`;
}

function renderVolumeCard(page) {
  return `<a class="volume-card" href="#${page.id}">
    <div class="card-header"><span class="card-number">第${page.num}卷</span>
      <span class="realm-badge r-${(page.realmClass||'').replace('realm-','')}">${page.realm}</span></div>
    <div class="card-title">${page.title}</div>
    <div class="card-realm">${page.subtitle}</div>
    <div class="card-desc">${page.description}</div>
    <div class="card-footer"><span class="card-lines">${(page.lines||0).toLocaleString()} 行</span>
      <span class="card-arrow">→</span></div></a>`;
}

function renderAppendixCard(page) {
  return `<a class="appendix-card" href="#${page.id}">
    <div class="card-icon">${page.icon||'📘'}</div>
    <div class="card-title">${page.title}</div>
    <div class="card-desc">${page.description}</div></a>`;
}

// =============================================
// KNOWLEDGE GRAPH + SECTION SPLITTING SYSTEM
// 先看图谱全景 → 再展开具体章节
// =============================================

// CRITICAL: expose functions to global scope for inline onclick handlers
// (app.js is loaded as type="module", so functions are not global by default)
window.scrollToSection = scrollToSection;
window.toggleKnowledgeGraph = toggleKnowledgeGraph;

/**
 * Main entry point: initializes both the cross-volume knowledge graph overview
 * and the intra-volume collapsible section cards for a given page.
 */
function initKnowledgeGraphAndSections(container, pageMeta) {
  const mdContent = container.querySelector('.md-content');
  if (!mdContent) return;

  const sections = pageMeta.sections || [];
  if (sections.length < 2) return;

  // CRITICAL: marked.js v9 does NOT auto-generate heading IDs.
  // We must set them ourselves using the IDs from build.py (sections data).
  // This ensures pills, graph nodes, and section-cards all use the same IDs.
  const h2Elements = Array.from(mdContent.querySelectorAll('h2'));
  h2Elements.forEach((h2El, idx) => {
    if (idx < sections.length) {
      h2El.id = sections[idx].id; // Set the h2 ID from build.py data
    } else {
      h2El.id = h2El.id || `section-extra-${idx}`;
    }
  });

  // 1. Build quick-nav bar (pills for each section, with collapse if too many)
  buildQuickNav(mdContent, sections);

  // 2. Build intra-volume knowledge graph (section dependency flow)
  buildIntraVolumeGraph(mdContent, sections, pageMeta);

  // 3. Split content into collapsible section cards
  splitIntoSectionCards(mdContent, sections);

  // 4. Build cross-volume knowledge graph (at top, only for main volumes)
  if (!pageMeta.isAppendix) {
    buildCrossVolumeGraph(container, pageMeta);
  }
}

// === QUICK NAV BAR (快速导航药丸栏) ===

const QUICK_NAV_MAX_VISIBLE = 10; // Show at most 10 pills before collapsing

function buildQuickNav(mdContent, sections) {
  const bar = document.createElement('div');
  bar.className = 'quick-nav-bar';

  // Label
  const label = document.createElement('span');
  label.className = 'quick-nav-label';
  label.textContent = '📍 快速跳转:';
  bar.appendChild(label);

  // Pill container
  const pillContainer = document.createElement('div');
  pillContainer.className = 'quick-nav-pills';

  sections.forEach((sec, i) => {
    const pill = document.createElement('span');
    pill.className = 'quick-nav-pill';
    if (i >= QUICK_NAV_MAX_VISIBLE) pill.classList.add('quick-nav-hidden');
    pill.dataset.secId = sec.id;
    pill.textContent = `${toChapterLabel(i + 1)} ${truncateText(sec.title, 14)}`;
    pill.addEventListener('click', (e) => scrollToSection(sec.id, e));
    pillContainer.appendChild(pill);
  });

  bar.appendChild(pillContainer);

  // "Show more" toggle if too many sections
  if (sections.length > QUICK_NAV_MAX_VISIBLE) {
    const moreBtn = document.createElement('span');
    moreBtn.className = 'quick-nav-more';
    moreBtn.textContent = `+${sections.length - QUICK_NAV_MAX_VISIBLE} 更多`;
    moreBtn.addEventListener('click', () => {
      const hidden = pillContainer.querySelectorAll('.quick-nav-hidden');
      if (hidden.length > 0) {
        hidden.forEach(p => p.classList.remove('quick-nav-hidden'));
        moreBtn.textContent = '收起';
      } else {
        pillContainer.querySelectorAll('.quick-nav-pill').forEach((p, idx) => {
          if (idx >= QUICK_NAV_MAX_VISIBLE) p.classList.add('quick-nav-hidden');
        });
        moreBtn.textContent = `+${sections.length - QUICK_NAV_MAX_VISIBLE} 更多`;
      }
    });
    bar.appendChild(moreBtn);
  }

  // Insert before first h2 or at beginning
  const firstH2 = mdContent.querySelector('h2');
  if (firstH2) {
    firstH2.parentNode.insertBefore(bar, firstH2);
  } else {
    mdContent.insertBefore(bar, mdContent.firstChild);
  }
}

function scrollToSection(secId, event) {
  if (event) { event.preventDefault(); event.stopPropagation(); }

  // Highlight active pill
  document.querySelectorAll('.quick-nav-pill').forEach(p => p.classList.toggle('active', p.dataset.secId === secId));

  // Expand and scroll to section card
  const card = document.querySelector(`.section-card[data-section-id="${secId}"]`);
  if (card) {
    expandSectionCard(card);
    const headerOffset = 90;
    const elPos = card.getBoundingClientRect().top;
    window.scrollTo({ top: elPos + window.pageYOffset - headerOffset, behavior: 'smooth' });
    return; // Card found, no need for fallback
  }

  // Fallback: scroll to heading directly
  const heading = document.getElementById(secId);
  if (heading) {
    scrollToHeading(secId, null);
  }
}

// === INTRA-VOLUME KNOWLEDGE GRAPH (卷内章节图谱) ===

function buildIntraVolumeGraph(mdContent, sections, pageMeta) {
  if (sections.length < 3) return;
  // Skip for very large section counts (e.g. interview chapter with 44 sections)
  // as the graph becomes unreadable
  if (sections.length > 25) return;

  const canvasW = Math.min(800, mdContent.clientWidth - 40) || 760;
  const canvasH = Math.max(180, Math.min(320, sections.length * 45));

  // Node layout: grid flow
  const cols = Math.ceil(Math.sqrt(sections.length));
  const rows = Math.ceil(sections.length / cols);
  const xGap = canvasW / (cols + 1);
  const yGap = canvasH / (rows + 1);

  const nodes = sections.map((sec, i) => {
    const row = Math.floor(i / cols);
    const col = i % cols;
    return { id: sec.id, title: sec.title, x: xGap * (col + 1), y: yGap * (row + 1), index: i };
  });

  // Edges: sequential chain
  const edges = nodes.slice(0, -1).map((n, i) => ({ source: n.id, target: nodes[i + 1].id }));

  let svgHtml = `<svg viewBox="0 0 ${canvasW} ${canvasH}" xmlns="http://www.w3.org/2000/svg">`;
  svgHtml += `<defs><marker id="iv-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="var(--border-color)" opacity=".4"/></marker></defs>`;

  // Draw edges
  edges.forEach(e => {
    const src = nodes.find(n => n.id === e.source);
    const tgt = nodes.find(n => n.id === e.target);
    if (src && tgt) {
      const midX = (src.x + tgt.x) / 2;
      const midY = Math.min(src.y, tgt.y) - 15;
      svgHtml += `<path class="iv-graph-edge" d="M ${src.x} ${src.y} Q ${midX} ${midY} ${tgt.x} ${tgt.y}" marker-end="url(#iv-arrowhead)"/>`;
    }
  });

  // Draw nodes — use addEventListener via DOM after insertion
  nodes.forEach(n => {
    const label = truncateText(n.title.replace(/^[\d.]+\s*/, ''), 10);
    const isCurrent = n.index === 0;
    svgHtml += `<g class="iv-graph-node" data-target="${n.id}">
      <rect x="${n.x - 50}" y="${n.y - 16}" width="100" height="32" rx="6"
        fill="${isCurrent ? 'rgba(255,107,53,.12)' : 'var(--bg-secondary)'}"
        stroke="${isCurrent ? 'var(--c-fire)' : 'var(--border-color)'}" stroke-width="${isCurrent ? 1.5 : 1}"/>
      <text x="${n.x}" y="${n.y}">${label}</text>
    </g>`;
  });

  svgHtml += '</svg>';

  const graphEl = document.createElement('div');
  graphEl.className = 'iv-graph';
  graphEl.innerHTML = `
    <div class="iv-graph-header">
      <span class="iv-graph-title">🗺️ 本卷知识图谱（${sections.length} 个章节）</span>
      <span class="iv-graph-toggle">收起 ▾</span>
    </div>
    <div class="iv-graph-canvas">${svgHtml}</div>`;

  // Bind header toggle via addEventListener (not inline onclick)
  graphEl.querySelector('.iv-graph-header').addEventListener('click', () => {
    graphEl.classList.toggle('collapsed');
    const toggle = graphEl.querySelector('.iv-graph-toggle');
    toggle.textContent = graphEl.classList.contains('collapsed') ? '展开 ▸' : '收起 ▾';
  });

  // Bind node clicks via addEventListener
  graphEl.querySelectorAll('.iv-graph-node').forEach(nodeEl => {
    nodeEl.style.cursor = 'pointer';
    nodeEl.addEventListener('click', (e) => {
      scrollToSection(nodeEl.dataset.target, e);
    });
  });

  const quickNav = mdContent.querySelector('.quick-nav-bar');
  if (quickNav) {
    quickNav.insertAdjacentElement('afterend', graphEl);
  } else {
    mdContent.insertBefore(graphEl, mdContent.firstChild);
  }
}

// === SECTION CARDS SPLITTING (分节折叠系统) ===

function splitIntoSectionCards(mdContent, sections) {
  const h2Elements = Array.from(mdContent.querySelectorAll('h2'));

  if (h2Elements.length < 2) return; // Not worth splitting

  // Strategy: insert placeholder markers, move content, replace markers with cards.

  // Step 1: Insert placeholder markers BEFORE each h2 (while DOM is intact)
  const markers = [];
  h2Elements.forEach((h2El, idx) => {
    const marker = document.createElement('div');
    marker.className = 'section-placeholder';
    marker.dataset.idx = idx;
    h2El.parentNode.insertBefore(marker, h2El);
    markers.push(marker);
  });

  // Step 2: For each h2, collect elements between it and the next placeholder
  h2Elements.forEach((h2El, idx) => {
    const secId = h2El.getAttribute('id') || `section-${idx}`;
    const sectionData = sections.find(s => s.id === secId) || {};
    const children = sectionData.children || [];
    const secIndex = sections.findIndex(s => s.id === secId);

    // Collect elements: from h2 until next placeholder (or end)
    const elementsToMove = [];
    let sibling = h2El;
    while (sibling) {
      const next = sibling.nextElementSibling;
      if (sibling !== h2El && sibling.classList && sibling.classList.contains('section-placeholder')) break;
      elementsToMove.push(sibling);
      sibling = next;
    }

    // Build card
    const card = document.createElement('div');
    card.className = 'section-card';
    card.dataset.sectionId = secId;

    const childTags = children.slice(0, 5).map(c =>
      `<span class="section-child-tag">${truncateText(c.title, 12)}</span>`
    ).join('') + (children.length > 5 ? `<span class="section-child-tag muted">+${children.length - 5}</span>` : '');

    const cardHeader = document.createElement('div');
    cardHeader.className = 'section-card-header';
    cardHeader.innerHTML = `
      <span class="section-num">${toChapterLabel(secIndex + 1)}</span>
      <div class="section-info">
        <div class="section-title">${h2El.textContent}</div>
        ${children.length > 0 ? `<div class="section-meta">
          <span class="section-child-count">📑 ${children.length} 个小节</span>
          <div class="section-children-preview">${childTags}</div>
        </div>` : ''}
      </div>
      <span class="section-expand-icon">▸</span>`;

    const body = document.createElement('div');
    body.className = 'section-card-body';
    const bodyInner = document.createElement('div');
    bodyInner.className = 'section-card-body-inner';

    elementsToMove.forEach(el => bodyInner.appendChild(el));
    body.appendChild(bodyInner);

    card.appendChild(cardHeader);
    card.appendChild(body);

    // Click to toggle — using addEventListener, not inline onclick
    cardHeader.addEventListener('click', () => toggleSectionCard(card));

    // Replace the placeholder marker with the card
    const marker = markers[idx];
    if (marker && marker.parentNode) {
      marker.parentNode.replaceChild(card, marker);
    }

    // Auto-expand first section by default
    if (idx === 0) {
      card.classList.add('expanded');
      // Use requestAnimationFrame to get correct scrollHeight after DOM update
      requestAnimationFrame(() => {
        body.style.maxHeight = body.scrollHeight + 200 + 'px';
      });
    }
  });
}

function toggleSectionCard(card) {
  const isExpanded = card.classList.contains('expanded');
  const body = card.querySelector('.section-card-body');

  if (isExpanded) {
    card.classList.remove('expanded');
    body.style.maxHeight = '0';
  } else {
    card.classList.add('expanded');
    body.style.maxHeight = body.scrollHeight + 500 + 'px';

    // Update active state on quick nav pills
    const secId = card.dataset.sectionId;
    document.querySelectorAll('.quick-nav-pill').forEach(p => {
      p.classList.toggle('active', p.dataset.secId === secId);
    });
  }
}

function expandSectionCard(card) {
  if (!card.classList.contains('expanded')) {
    card.classList.add('expanded');
    const body = card.querySelector('.section-card-body');
    if (body) body.style.maxHeight = body.scrollHeight + 500 + 'px';
  }
}

// === CROSS-VOLUME KNOWLEDGE GRAPH (跨卷知识图谱总览) ===

function buildCrossVolumeGraph(container, pageMeta) {
  const kgData = SITE_DATA.knowledgeGraph;
  if (!kgData || !kgData.links) return;

  const pages = ALL_PAGES.filter(p => !p.isAppendix);
  const links = kgData.links;

  const canvasW = 700;
  const canvasH = pages.length * 52 + 60;
  const centerX = canvasW / 2;

  const nodePositions = {};
  pages.forEach((p, i) => {
    nodePositions[p.id] = {
      x: centerX + (i % 2 === 0 ? -120 : 120),
      y: 40 + i * 52,
      title: `第${p.num}卷 ${p.title}`,
      shortTitle: `${p.num}.${p.title}`,
      isCurrent: p.id === pageMeta.id,
    };
  });

  let svgHtml = `<svg viewBox="0 0 ${canvasW} ${canvasH}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="kg-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="var(--text-muted)" opacity=".5"/>
      </marker>
      <linearGradient id="kg-glow" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="var(--c-fire)" stop-opacity=".15"/>
        <stop offset="100%" stop-color="var(--c-purple)" stop-opacity=".08"/>
      </linearGradient>
    </defs>`;

  // Draw edges
  links.forEach(link => {
    const src = nodePositions[link.source];
    const tgt = nodePositions[link.target];
    if (src && tgt) {
      const typeClass = link.type === 'prerequisite' ? '' :
                         link.type === 'concept' ? 'kg-concept' : 'kg-app';
      svgHtml += `<g class="kg-link-group ${typeClass}">
        <path class="kg-link" d="M ${src.x} ${src.y} C ${src.x} ${(src.y + tgt.y)/2 - 20}, ${tgt.x} ${(src.y + tgt.y)/2 + 20}, ${tgt.x} ${tgt.y}"
          marker-end="url(#kg-arrowhead)" data-label="${link.label}"/>
      </g>`;
    }
  });

  // Draw nodes — NO inline onclick, will bind via addEventListener
  Object.entries(nodePositions).forEach(([id, pos]) => {
    const cls = pos.isCurrent ? 'kg-node-current' : '';
    const r = pos.isCurrent ? 22 : 18;
    svgHtml += `<g class="kg-node ${cls}" data-vol-id="${id}" ${id !== pageMeta.id ? 'style="cursor:pointer"' : ''}>
      <circle class="kg-node-circle" cx="${pos.x}" cy="${pos.y}" r="${r}"
        fill="${pos.isCurrent ? 'url(#kg-glow)' : 'var(--bg-secondary)'}"
        stroke="${pos.isCurrent ? 'var(--c-fire)' : 'var(--border-color)'}"/>
      <text class="kg-node-text" x="${pos.x}" y="${pos.y - 3}">${pos.shortTitle}</text>
      ${pos.isCurrent ? `<text class="kg-node-sub" x="${pos.x}" y="${pos.y + 11}">← 当前</text>` : ''}
    </g>`;
  });

  svgHtml += '</svg>';

  // Assemble widget
  const kgWidget = document.createElement('div');
  kgWidget.className = state.kgExpanded ? 'kg-overview' : 'kg-overview collapsed';
  kgWidget.innerHTML = `
    <div class="kg-header">
      <div class="kg-header-left">
        <span class="kg-icon">🕸️</span>
        <span class="kg-title">修炼体系 · 知识图谱</span>
        <span class="kg-subtitle">${pages.length} 卷关联 · ${links.length} 条知识链路</span>
      </div>
      <span class="kg-toggle">${state.kgExpanded ? '收起' : '展开'}</span>
    </div>
    <div class="kg-canvas-wrap"><div class="kg-canvas">${svgHtml}</div></div>
    <div class="kg-legend">
      <div class="kg-legend-item"><span class="kg-legend-dot prereq"></span> 前置依赖</div>
      <div class="kg-legend-item"><span class="kg-legend-dot concept"></span> 概念关联</div>
      <div class="kg-legend-item"><span class="kg-legend-dot app"></span> 实战应用</div>
    </div>`;

  // Bind header toggle via addEventListener
  kgWidget.querySelector('.kg-header').addEventListener('click', () => toggleKnowledgeGraph());

  // Bind node clicks via addEventListener
  kgWidget.querySelectorAll('.kg-node').forEach(nodeEl => {
    const volId = nodeEl.dataset.volId;
    if (volId !== pageMeta.id) {
      nodeEl.addEventListener('click', () => { window.location.hash = volId; });
    }
  });

  // Insert before breadcrumb
  const breadcrumb = container.querySelector('.breadcrumb');
  if (breadcrumb) {
    breadcrumb.insertAdjacentElement('beforebegin', kgWidget);
  } else {
    const mdContent = container.querySelector('.md-content');
    if (mdContent) mdContent.insertAdjacentElement('beforebegin', kgWidget);
  }
}

function toggleKnowledgeGraph() {
  state.kgExpanded = !state.kgExpanded;
  const overview = document.querySelector('.kg-overview');
  if (overview) {
    overview.classList.toggle('collapsed', !state.kgExpanded);
    const toggle = overview.querySelector('.kg-toggle');
    if (toggle) toggle.textContent = state.kgExpanded ? '收起' : '展开';
  }
}

// === UTILITY FUNCTIONS ===

/**
 * Slugify a heading text to match GitHub-flavored markdown anchor format.
 * Rules: lowercase, remove punctuation except hyphens, spaces→hyphens, collapse hyphens.
 * Keeps CJK characters, letters, digits, hyphens.
 */
function slugifyHeading(text) {
  return text
    .trim()
    .toLowerCase()
    .replace(/[—–]/g, '-')      // em-dash/en-dash → hyphen (BEFORE removing punctuation)
    .replace(/[^\w\u4e00-\u9fff\u3400-\u4dbf\s-]/g, '') // keep word chars, CJK, spaces, hyphens
    .replace(/\s+/g, '-')       // spaces → hyphens
    .replace(/^-|-$/g, '');     // trim leading/trailing hyphens (but keep interior --)
}

function toChapterLabel(n) {
  if (n <= 0) return String(n);
  // Support numbers up to 99
  const units = ['零','一','二','三','四','五','六','七','八','九'];
  if (n <= 10) return units[n] || String(n);
  if (n < 20) return '十' + (n % 10 === 0 ? '' : units[n % 10]);
  if (n < 100) {
    const tens = Math.floor(n / 10);
    const ones = n % 10;
    return units[tens] + '十' + (ones === 0 ? '' : units[ones]);
  }
  return String(n);
}

// Backward compat alias
function toCnNum(n) { return toChapterLabel(n); }

function truncateText(text, maxLen) {
  if (!text) return '';
  text = text.trim();
  return text.length > maxLen ? text.substring(0, maxLen) + '…' : text;
}
