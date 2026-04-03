#!/usr/bin/env python3
"""
焚诀 FlameSutra - GitHub Pages 构建脚本
将所有 Markdown 卷转换为静态站点
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
PAGES_DIR = SITE_DIR / "pages"
INDEX_HTML = SITE_DIR / "index.html"

# --- Volume definitions ---
VOLUMES = [
    {
        "id": "vol01",
        "dir": "01-筑基篇-斗之气",
        "title": "筑基篇",
        "subtitle": "凝聚气旋",
        "realm": "斗之气",
        "realm_class": "dou-zhi-qi",
        "num": "一",
        "description": "Python、线性代数、微积分、概率论、PyTorch 基础",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol02",
        "dir": "02-纳灵篇-斗者到斗师",
        "title": "纳灵篇",
        "subtitle": "炼气入体",
        "realm": "斗者→斗师",
        "realm_class": "dou-zhe",
        "num": "二",
        "description": "CNN、RNN/LSTM、Transformer 架构深度拆解",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol03",
        "dir": "03-凝晶篇-大斗师",
        "title": "凝晶篇",
        "subtitle": "斗气固化",
        "realm": "大斗师",
        "realm_class": "da-dou-shi",
        "num": "三",
        "description": "BERT、GPT-2 预训练、Tokenizer、Embedding",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol04",
        "dir": "04-化形篇-斗灵",
        "title": "化形篇",
        "subtitle": "丹药化形",
        "realm": "斗灵",
        "realm_class": "dou-ling",
        "num": "四",
        "description": "PEFT（LoRA/DoRA/AdaLoRA）、量化、Prompt Engineering",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol05",
        "dir": "05-化翼篇-斗王",
        "title": "化翼篇",
        "subtitle": "斗气化翼",
        "realm": "斗王",
        "realm_class": "dou-wang",
        "num": "五",
        "description": "DDP、DeepSpeed、混合精度、梯度累加",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol06",
        "dir": "06-凌空篇-斗皇",
        "title": "凌空篇",
        "subtitle": "凌空而行",
        "realm": "斗皇",
        "realm_class": "dou-huang",
        "num": "六",
        "description": "SFT 体系、数据清洗、RAG 与向量数据库",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol07",
        "dir": "07-空间篇-斗宗",
        "title": "空间篇",
        "subtitle": "空间之力",
        "realm": "斗宗",
        "realm_class": "dou-zong",
        "num": "七",
        "description": "VLM 多模态、CLIP、Vision Encoder + LLM 融合",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol08",
        "dir": "08-九转篇-斗尊",
        "title": "九转篇",
        "subtitle": "空间粉碎",
        "realm": "斗尊",
        "realm_class": "dou-zun",
        "num": "八",
        "description": "RLHF、PPO、DPO、安全对齐",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol09",
        "dir": "09-成圣篇-斗圣",
        "title": "成圣篇",
        "subtitle": "半步成神",
        "realm": "斗圣",
        "realm_class": "dou-sheng",
        "num": "九",
        "description": "MoE、Flash Attention、Megatron-LM、长文本",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "vol10",
        "dir": "10-帝境篇-斗帝",
        "title": "帝境篇",
        "subtitle": "破碎虚空",
        "realm": "斗帝",
        "realm_class": "dou-di",
        "num": "十",
        "description": "自进化算法、多模态原生大统一、逻辑推理突破",
        "is_appendix": False,
        "icon": "📖",
    },
    {
        "id": "huofire",
        "dir": "万火录-GPU异火榜",
        "title": "万火录",
        "subtitle": "GPU 异火榜",
        "realm": "附录",
        "realm_class": "",
        "num": "",
        "description": "GPU 参数全录：从 RTX 3060 到 H200",
        "is_appendix": True,
        "icon": "🔥",
    },
    {
        "id": "debug",
        "dir": "反噬录-Debug宝典",
        "title": "反噬录",
        "subtitle": "Debug 宝典",
        "realm": "附录",
        "realm_class": "",
        "num": "",
        "description": "炼丹失败案例集：Loss 震荡、模型坍塌、OOM 复盘",
        "is_appendix": True,
        "icon": "⚡",
    },
    {
        "id": "dataset",
        "dir": "药材库-数据集指南",
        "title": "药材库",
        "subtitle": "数据集指南",
        "realm": "附录",
        "realm_class": "",
        "num": "",
        "description": "开源数据集收录与洗炼工具",
        "is_appendix": True,
        "icon": "🌿",
    },
    {
        "id": "interview",
        "dir": "11-面经篇-天劫试炼",
        "title": "面经篇",
        "subtitle": "天劫试炼",
        "realm": "附录",
        "realm_class": "",
        "num": "",
        "description": "算法面经与面试通关指南：Python/ML/DL/Transformer/NLP/LLM/RAG/系统设计九重天劫",
        "is_appendix": True,
        "icon": "⚔️",
    },
]


def count_lines(filepath):
    """Count lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def extract_sections(md_content):
    """
    Extract section structure from markdown content.
    Returns a list of section dicts with id, level, title, and children.
    This enables knowledge-graph + collapsible section rendering.
    """
    sections = []
    lines = md_content.split('\n')
    current_h2 = None

    for line in lines:
        if line.startswith('## '):
            title = line[3:].strip()
            sec_id = generate_id_from_text(title)
            section = {
                "id": sec_id,
                "title": title,
                "level": 2,
                "children": [],
            }
            sections.append(section)
            current_h2 = section
        elif line.startswith('### ') and current_h2 is not None:
            title = line[4:].strip()
            child_id = generate_id_from_text(title)
            child = {
                "id": child_id,
                "title": title,
                "level": 3,
            }
            current_h2["children"].append(child)

    return sections


def extract_knowledge_links(all_volumes_data):
    """
    Build cross-volume knowledge graph links based on known concept dependencies.
    These links show how concepts flow between volumes.
    """
    # Define key concepts per volume and their forward/backward references
    link_rules = [
        # (from_vol, to_vol, relation_label, relation_type)
        ("vol01", "vol02", "PyTorch基础 → 架构入门", "prerequisite"),
        ("vol02", "vol03", "Transformer架构 → 预训练", "prerequisite"),
        ("vol03", "vol04", "预训练模型 → 高效微调", "prerequisite"),
        ("vol04", "vol05", "LoRA微调 → 分布式扩展", "prerequisite"),
        ("vol05", "vol06", "分布式训练 → 工业级流水线", "prerequisite"),
        ("vol06", "vol07", "SFT/RAG → 多模态融合", "prerequisite"),
        ("vol07", "vol08", "VLM多模态 → 对齐训练", "prerequisite"),
        ("vol08", "vol09", "RLHF对齐 → 大规模预训练", "prerequisite"),
        ("vol09", "vol10", "基础架构 → AGI前沿", "prerequisite"),
        # Cross-volume concept links
        ("vol01", "vol05", "Python基础 → 混合精度", "concept"),
        ("vol02", "vol07", "CNN → 视觉编码器", "concept"),
        ("vol02", "vol08", "Transformer → PPO对齐", "concept"),
        ("vol03", "vol06", "Tokenizer → 数据清洗", "concept"),
        ("vol04", "vol08", "PEFT微调 → DPO优化", "concept"),
        ("vol05", "vol09", "DeepSpeed → Megatron-LM", "concept"),
        ("vol06", "vol11", "SFT工程 → 面经实战", "application"),
        ("vol07", "vol10", "多模态 → 原生统一", "concept"),
        ("vol09", "vol10", "MoE → 自进化Agent", "concept"),
    ]

    return [
        {"source": s, "target": t, "label": l, "type": typ}
        for s, t, l, typ in link_rules
    ]


def generate_id_from_text(text):
    """Generate a URL-safe ID from heading text."""
    # Remove markdown formatting
    text = re.sub(r'[#*`_\[\]()]', '', text)
    # Replace Chinese punctuation
    text = text.replace('—', '-').replace('：', ':').replace('，', ',')
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace(' ', '-')
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\-_.]', '', text)
    return text.strip('-_')


def build_site():
    """Main build function."""
    print("🔥 焚诀 GitHub Pages 构建开始...\n")
    
    # Ensure pages directory exists
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    pages_data = []
    all_sections = {}  # vol_id -> sections list

    # Process each volume
    for vol in VOLUMES:
        src_file = ROOT / vol["dir"] / "README.md"
        dst_file = PAGES_DIR / f"{vol['id']}.html"
        
        if not src_file.exists():
            print(f"  ⚠️  {vol['dir']}/README.md 不存在，跳过")
            continue
        
        # Read markdown
        with open(src_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Extract section structure for knowledge graph + collapsible sections
        sections = extract_sections(md_content)
        all_sections[vol["id"]] = sections

        # Copy as .html (will be rendered client-side as markdown)
        with open(dst_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # Get line count
        lines = count_lines(src_file)

        # Build page data
        page_data = {
            "id": vol["id"],
            "title": vol["title"],
            "subtitle": vol["subtitle"],
            "realm": vol["realm"],
            "realmClass": vol["realm_class"],
            "num": vol["num"],
            "description": vol["description"],
            "isAppendix": vol["is_appendix"],
            "icon": vol["icon"],
            "lines": lines,
            "sections": sections,
            "sectionCount": len(sections),
        }
        pages_data.append(page_data)
        
        status = "✅"
        print(f"  {status} {vol['id']:10s} | {vol['title']:8s} | {lines:>6,} 行")
    
    # Build site data JSON
    # Extract cross-volume knowledge links
    knowledge_links = extract_knowledge_links(pages_data)

    site_data = {
        "pages": pages_data,
        "home": {
            "title": "焚诀",
            "subtitle": "A Cultivation Guide for Foundation Models: From Dust to Deity",
        },
        "knowledgeGraph": {
            "links": knowledge_links,
        },
    }
    
    site_data_json = json.dumps(site_data, ensure_ascii=False, indent=2)
    
    # Update index.html with site data
    with open(INDEX_HTML, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    if '__SITE_DATA_PLACEHOLDER__' in html_content:
        html_content = html_content.replace('__SITE_DATA_PLACEHOLDER__', site_data_json)
    else:
        # Escape backslashes in json for regex substitution
        safe_json = site_data_json.replace('\\', '\\\\')
        html_content = re.sub(
            r'(window\.__SITE_DATA__\s*=\s*)\{.*?\};',
            r'\g<1>' + safe_json + ';',
            html_content,
            flags=re.DOTALL
        )
    
    with open(INDEX_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Generate sidebar HTML
    sidebar_html = generate_sidebar(pages_data)
    
    with open(INDEX_HTML, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Replace sidebar placeholder
    html_content = re.sub(
        r'(<aside class="sidebar" id="sidebar">).*?(</aside>)',
        r'\1\n' + generate_sidebar(pages_data) + r'\n  \2',
        html_content,
        flags=re.DOTALL
    )
    
    # Replace nav placeholder  
    nav_html = generate_nav(pages_data)
    html_content = re.sub(
        r'(<nav class="header-nav" id="header-nav">).*?(</nav>)',
        r'\1\n    ' + nav_html + r'\n  \2',
        html_content,
        flags=re.DOTALL
    )
    
    with open(INDEX_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n📊 统计:")
    print(f"   总页数: {len(pages_data)}")
    print(f"   总行数: {sum(p['lines'] for p in pages_data):,}")
    print(f"\n📁 输出目录: {SITE_DIR}")
    print(f"\n✅ 构建完成！")
    print(f"\n💡 预览: cd {SITE_DIR} && python -m http.server 8080")
    print(f"   或: npx serve {SITE_DIR}")


def generate_sidebar(pages_data):
    """Generate sidebar HTML."""
    main_pages = [p for p in pages_data if not p["isAppendix"]]
    appendix_pages = [p for p in pages_data if p["isAppendix"]]
    
    html = ''
    
    # Home link
    html += f'    <a class="sidebar-item" data-page="home" href="#home">\n'
    html += f'      <span class="item-icon">🏠</span>\n'
    html += f'      <span class="item-label">功法总览</span>\n'
    html += f'    </a>\n'
    
    # Main volumes
    html += '    <div class="sidebar-section">\n'
    html += '      <div class="sidebar-section-title">正篇十卷</div>\n'
    for p in main_pages:
        badge = f'<span class="realm-badge realm-{p["realmClass"]}">{p["realm"]}</span>' if p["realmClass"] else ''
        html += f'    <a class="sidebar-item" data-page="{p["id"]}" href="#{p["id"]}">\n'
        html += f'      <span class="item-icon">📖</span>\n'
        html += f'      <span class="item-label">第{p["num"]}卷 {p["title"]}</span>\n'
        html += f'      {badge}\n'
        html += f'    </a>\n'
    html += '    </div>\n'
    
    # Appendix
    html += '    <div class="sidebar-section">\n'
    html += '      <div class="sidebar-section-title">辅助典籍</div>\n'
    for p in appendix_pages:
        html += f'    <a class="sidebar-item" data-page="{p["id"]}" href="#{p["id"]}">\n'
        html += f'      <span class="item-icon">{p["icon"]}</span>\n'
        html += f'      <span class="item-label">{p["title"]}</span>\n'
        html += f'    </a>\n'
    html += '    </div>\n'
    
    return html


def generate_nav(pages_data):
    """Generate header nav HTML."""
    main_pages = [p for p in pages_data if not p["isAppendix"]]
    
    html = ''
    for p in main_pages:
        label = f"第{p['num']}卷"
        html += f'<a data-page="{p["id"]}" href="#{p["id"]}">{label}</a>\n    '
    
    return html


if __name__ == '__main__':
    build_site()
