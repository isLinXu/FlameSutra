// =============================================
// 焚诀 FlameSutra - 核心脚本
// SPA路由 + Markdown渲染 + 搜索 + TOC
// =============================================

// marked and DOMPurify are loaded globally via <script> tags in index.html

// --- State ---
const state = {
  currentPage: null,
  searchQuery: '',
  sidebarOpen: false,
  searchOpen: false,
};

// --- Data (injected at build time) ---
const SITE_DATA = window.__SITE_DATA__ || {};
const ALL_PAGES = SITE_DATA.pages || [];
const HOME_DATA = SITE_DATA.home || {};

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  initRouter();
  initSidebar();
  initSearch();
  initScrollTop();
  initProgress();
});

// --- Router ---
function initRouter() {
  const hash = window.location.hash.slice(1) || 'home';
  navigateTo(hash);
  window.addEventListener('hashchange', () => {
    const h = window.location.hash.slice(1) || 'home';
    if (h !== state.currentPage) navigateTo(h);
  });
}

async function navigateTo(pageId) {
  state.currentPage = pageId;
  state.searchOpen = false;
  document.querySelector('.search-results')?.classList.remove('show');
  closeSidebar();

  // Debug logging
  console.log('Navigate to:', pageId);
  console.log('Available pages:', ALL_PAGES.map(p => p.id));

  // Update active sidebar item
  document.querySelectorAll('.sidebar-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === pageId);
  });

  // Update active nav
  document.querySelectorAll('.header-nav a').forEach(el => {
    el.classList.toggle('active', el.dataset.page === pageId);
  });

  const contentArea = document.getElementById('content');
  
  if (pageId === 'home') {
    renderHome(contentArea);
  } else {
    // Render page content
    const page = ALL_PAGES.find(p => p.id === pageId);
    console.log('Found page:', page);
    if (page) {
      contentArea.innerHTML = '<div class="loading-spinner">加载中</div>';
      try {
        const resp = await fetch(`pages/${pageId}.html`);
        if (!resp.ok) throw new Error('Not found');
        const md = await resp.text();
        renderMarkdownPage(contentArea, md, page);
      } catch (e) {
        console.error('Fetch error:', e);
        contentArea.innerHTML = `<div class="md-content"><h1>404 - 页面未找到</h1><p>该卷的修炼内容尚未加载。</p><a href="#home">返回总览</a></div>`;
      }
    } else {
      contentArea.innerHTML = `<div class="md-content"><h1>404</h1><p>未知页面: ${pageId}</p><a href="#home">返回总览</a></div>`;
    }
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderMarkdownPage(container, md, pageMeta) {
  // Configure marked
  marked.setOptions({
    gfm: true,
    breaks: false,
  });

  let html;
  try {
    const rawHtml = marked.parse(md);
    html = DOMPurify.sanitize(rawHtml);
  } catch (e) {
    console.error('Markdown parsing error:', e);
    html = `<pre style="color:red;white-space:pre-wrap;">解析错误: ${e.message}\n\n原始内容前500字符:\n${md.substring(0, 500)}</pre>`;
  }
  
  // Find page index for prev/next
  const idx = ALL_PAGES.findIndex(p => p.id === pageMeta.id);
  const prevPage = idx > 0 ? ALL_PAGES[idx - 1] : null;
  const nextPage = idx < ALL_PAGES.length - 1 ? ALL_PAGES[idx + 1] : null;

  const breadcrumb = `
    <div class="breadcrumb">
      <a href="#home">焚诀</a>
      <span class="sep">/</span>
      ${pageMeta.category ? `<a href="#${pageMeta.category}">${pageMeta.categoryLabel || pageMeta.category}</a><span class="sep">/</span>` : ''}
      <span class="current">${pageMeta.title}</span>
    </div>
  `;

  const pageNav = `
    <div class="page-nav">
      ${prevPage ? `<a class="page-nav-btn prev" href="#${prevPage.id}"><div class="nav-label">← 上一卷</div><div class="nav-title">${prevPage.title}</div></a>` : '<div></div>'}
      ${nextPage ? `<a class="page-nav-btn next" href="#${nextPage.id}"><div class="nav-label">下一卷 →</div><div class="nav-title">${nextPage.title}</div></a>` : '<div></div>'}
    </div>
  `;

  container.innerHTML = `
    ${breadcrumb}
    <div class="md-content">
      ${html}
    </div>
    ${pageNav}
  `;

  // Force browser to paint
  void container.offsetHeight;

  // Build TOC
  buildTOC(container.querySelector('.md-content'));
}

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

      <div class="appendix-section">
        <h2 class="appendix-title">📖 正篇十卷</h2>
        <div class="volume-grid">
          ${ALL_PAGES.filter(p => !p.isAppendix).map(p => renderVolumeCard(p)).join('')}
        </div>
      </div>

      <div class="appendix-section">
        <h2 class="appendix-title">📜 辅助典籍</h2>
        <div class="appendix-grid">
          ${ALL_PAGES.filter(p => p.isAppendix).map(p => renderAppendixCard(p)).join('')}
        </div>
      </div>

      <div class="appendix-section" style="margin-top: 40px;">
        <h2 class="appendix-title">🌍 术语对照</h2>
        <table>
          <thead><tr><th>玄幻术语</th><th>技术含义</th></tr></thead>
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
          </tbody>
        </table>
      </div>
    </div>

    <div class="site-footer">
      <p>《焚诀》FlameSutra — 大模型修炼指南 | Apache License 2.0</p>
      <p style="margin-top:8px;">以玄幻修仙之喻，讲解 Foundation Model 全栈技术</p>
    </div>
  `;
}

function renderRealmPyramid() {
  return `
    <div style="text-align:center; margin: 40px 0; font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); line-height: 2; overflow-x: auto;">
      <pre style="display:inline-block; text-align:left;">
┌──────────────────────────────────────────────────────┐
│  第十卷 · 帝境篇（斗帝）   │  AGI / 自进化 / 大统一架构  │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第九卷 · 成圣篇（斗圣）   │  MoE / FlashAttention / Megatron  │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第八卷 · 九转篇（斗尊）   │  RLHF / PPO / DPO / 安全对齐     │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第七卷 · 空间篇（斗宗）   │  VLM 多模态 / CLIP / LLaVA      │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第六卷 · 凌空篇（斗皇）   │  SFT / 数据清洗 / RAG             │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第五卷 · 化翼篇（斗王）   │  分布式训练 / DeepSpeed / 混合精度 │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第四卷 · 化形篇（斗灵）   │  PEFT / LoRA / 量化 / Prompt工程  │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第三卷 · 凝晶篇（大斗师） │  BERT / GPT-2 / Tokenizer       │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第二卷 · 纳灵篇（斗者-斗师）│ CNN / RNN / Transformer         │
└──────────────────┬───────────────────────────────────┘
┌──────────────────┴───────────────────────────────────┐
│  第一卷 · 筑基篇（斗之气） │  Python / 数学 / PyTorch 基础   │
└──────────────────────────────────────────────────────┘</pre>
    </div>
  `;
}

function renderVolumeCard(page) {
  return `
    <a class="volume-card" href="#${page.id}">
      <div class="card-header">
        <span class="card-number">第${page.num}卷</span>
        <span class="realm-badge realm-${page.realmClass || ''}">${page.realm}</span>
      </div>
      <div class="card-title">${page.title}</div>
      <div class="card-realm">${page.subtitle}</div>
      <div class="card-desc">${page.description}</div>
      <div class="card-footer">
        <span class="card-lines">${(page.lines || 0).toLocaleString()} 行</span>
        <span class="card-arrow">→</span>
      </div>
    </a>
  `;
}

function renderAppendixCard(page) {
  return `
    <a class="appendix-card" href="#${page.id}">
      <div class="card-icon">${page.icon || '📘'}</div>
      <div class="card-title">${page.title}</div>
      <div class="card-desc">${page.description}</div>
    </a>
  `;
}

// --- TOC ---
let tocObserver = null;

window.scrollToHeading = function(id, event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  const el = document.getElementById(id);
  if (el) {
    const headerOffset = 80;
    const elementPosition = el.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
    window.scrollTo({
         top: offsetPosition,
         behavior: "smooth"
    });
    
    // Manually update active class since observer might take time
    document.querySelectorAll('.toc-item').forEach(item => item.classList.remove('active'));
    const targetItem = document.querySelector(`.toc-item[data-target="${id}"]`);
    if (targetItem) {
      targetItem.classList.add('active');
    }
  } else {
    console.warn('Heading not found:', id);
  }
};

function buildTOC(mdContent) {
  const tocPanel = document.getElementById('toc-panel');
  if (!mdContent) return;

  // Cleanup old observer
  if (tocObserver) {
    tocObserver.disconnect();
  }

  const headings = mdContent.querySelectorAll('h2, h3, h4');
  if (headings.length < 3) {
    tocPanel.classList.remove('show');
    return;
  }

  let tocHtml = '<div class="toc-title">本卷目录</div>';
  headings.forEach((h, i) => {
    // Generate a robust ID if the heading doesn't have one
    const id = h.id || `heading-${i}`;
    if (!h.id) {
      h.id = id;
    }
    const cls = h.tagName === 'H3' ? 'toc-h3' : h.tagName === 'H4' ? 'toc-h4' : '';
    // Avoid href="#id" which triggers the hashchange router. 
    // Use href="javascript:void(0)" and onclick instead.
    // Also use button styling/behavior to be absolutely safe from router interference
    tocHtml += `<div class="toc-item ${cls}" data-target="${id}" style="cursor:pointer;" onclick="scrollToHeading('${id}', event)">${h.textContent}</div>`;
  });

  tocPanel.innerHTML = tocHtml;
  tocPanel.classList.add('show');

  // Setup IntersectionObserver for TOC highlighting
  const observerOptions = {
    root: null,
    rootMargin: '-80px 0px -60% 0px',
    threshold: 0
  };

  tocObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        document.querySelectorAll('.toc-item').forEach(item => item.classList.remove('active'));
        const activeItem = document.querySelector(`.toc-item[data-target="${id}"]`);
        if (activeItem) {
          activeItem.classList.add('active');
          // Scroll TOC panel to keep active item in view
          const panelHeight = tocPanel.clientHeight;
          const itemTop = activeItem.offsetTop;
          if (itemTop < tocPanel.scrollTop || itemTop > tocPanel.scrollTop + panelHeight - 30) {
             tocPanel.scrollTo({
                top: itemTop - panelHeight / 2,
                behavior: 'smooth'
             });
          }
        }
      }
    });
  }, observerOptions);

  headings.forEach(h => tocObserver.observe(h));
}

// --- Sidebar ---
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

// --- Search ---
function initSearch() {
  const input = document.querySelector('.search-input');
  if (!input) return;

  let debounceTimer;
  input.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      state.searchQuery = e.target.value.trim();
      performSearch(state.searchQuery);
    }, 200);
  });

  // Close on click outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.header-search') && !e.target.closest('.search-results')) {
      document.querySelector('.search-results')?.classList.remove('show');
      state.searchOpen = false;
    }
  });

  // Keyboard shortcut
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      input.focus();
    }
    if (e.key === 'Escape') {
      input.blur();
      document.querySelector('.search-results')?.classList.remove('show');
    }
  });
}

async function performSearch(query) {
  const resultsPanel = document.querySelector('.search-results');
  if (!query) {
    resultsPanel?.classList.remove('show');
    return;
  }

  resultsPanel.classList.add('show');
  resultsPanel.innerHTML = '<div class="loading-spinner" style="height:100px;">搜索中</div>';

  const results = [];
  const q = query.toLowerCase();

  for (const page of ALL_PAGES) {
    try {
      const resp = await fetch(`pages/${page.id}.html`);
      if (!resp.ok) continue;
      const text = await resp.text();
      const lines = text.split('\n');
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].toLowerCase();
        if (line.includes(q) && lines[i].trim().length > 10) {
          // Extract context
          const start = Math.max(0, i - 1);
          const end = Math.min(lines.length, i + 3);
          const context = lines.slice(start, end).join('\n').substring(0, 200);
          
          if (results.length < 20) {
            results.push({
              page,
              line: lines[i].substring(0, 150),
              context,
              lineNumber: i + 1,
            });
          }
          if (results.filter(r => r.page.id === page.id).length >= 3) break;
        }
      }
    } catch (e) { /* skip */ }
  }

  if (results.length === 0) {
    resultsPanel.innerHTML = `<div style="padding: 20px; color: var(--text-muted); text-align: center;">未找到相关内容</div>`;
    return;
  }

  resultsPanel.innerHTML = results.map(r => {
    const highlighted = r.line.replace(new RegExp(`(${escapeRegex(query)})`, 'gi'), '<mark>$1</mark>');
    return `
      <div class="search-result-item" onclick="window.location.hash='${r.page.id}'; document.querySelector('.search-results')?.classList.remove('show');">
        <div class="result-title">${r.page.title}</div>
        <div class="result-preview">${highlighted}</div>
        <div class="result-context">第 ${r.lineNumber} 行</div>
      </div>
    `;
  }).join('');
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// --- Scroll to Top ---
function initScrollTop() {
  const btn = document.querySelector('.scroll-top');
  if (!btn) return;
  window.addEventListener('scroll', () => {
    btn.classList.toggle('show', window.scrollY > 400);
  });
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// --- Progress Bar ---
function initProgress() {
  const bar = document.querySelector('.progress-bar');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    bar.style.width = progress + '%';
  });
}
