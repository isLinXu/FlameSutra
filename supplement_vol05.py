#!/usr/bin/env python3
"""为第五卷(斗王)补充ZeRO-3深度实战、Colossal-AI和MoE并行内容"""

filepath = "/Users/gatilin/PycharmProjects/FlameSutra-books/05-化翼篇-斗王/README.md"
content_file = "/Users/gatilin/PycharmProjects/FlameSutra-books/supplement_vol05_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

with open(content_file, 'r', encoding='utf-8') as f:
    new_content = f.read()

insert_marker = "## 修炼总结"
insert_idx = content.index(insert_marker)

content = content[:insert_idx] + new_content + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

new_lines = len(new_content.splitlines())
total_lines = len(content.splitlines())
print("OK: added %d lines, total now %d" % (new_lines, total_lines))
