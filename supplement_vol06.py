#!/usr/bin/env python3
"""为第六卷(斗皇)补充量化部署、RAG进阶和附录内容
将新内容从一个单独的文件读取并插入。
"""
import sys

filepath = "/Users/gatilin/PycharmProjects/FlameSutra-books/06-凌空篇-斗皇/README.md"
content_file = "/Users/gatilin/PycharmProjects/FlameSutra-books/supplement_vol06_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

with open(content_file, 'r', encoding='utf-8') as f:
    new_content = f.read()

insert_marker = "## 修炼总结与境界突破条件"
insert_idx = content.index(insert_marker)

content = content[:insert_idx] + new_content + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

new_lines = len(new_content.splitlines())
total_lines = len(content.splitlines())
print(f"OK: added {new_lines} lines, total now {total_lines}")
