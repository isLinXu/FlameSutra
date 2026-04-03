#!/usr/bin/env python3
"""为第九卷(斗圣)补充Ring Attention和Sequence Parallelism内容"""

filepath = "/Users/gatilin/PycharmProjects/FlameSutra-books/09-成圣篇-斗圣/README.md"
content_file = "/Users/gatilin/PycharmProjects/FlameSutra-books/supplement_vol09_content.txt"

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
print("OK: added %d lines, total now %d" % (new_lines, total_lines))
