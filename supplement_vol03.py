#!/usr/bin/env python3
"""第03卷(凝晶篇-大斗师)补充脚本"""

import os

filepath = "03-凝晶篇-大斗师/README.md"
content_path = "supplement_vol03_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\r\n', '\n')

# Read supplement content
with open(content_path, 'r', encoding='utf-8') as f:
    new_content = f.read()

# Find insertion point: before 修炼总结
insert_marker = "## 修炼总结与境界突破条件"
if insert_marker not in content:
    print(f"ERROR: Cannot find '{insert_marker}'")
    exit(1)

insert_idx = content.index(insert_marker)
content = content[:insert_idx] + new_content + "\n\n" + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done! File: {filepath}")
print(f"New lines: {len(new_content.splitlines())}")
print(f"Total lines: {len(content.splitlines())}")
