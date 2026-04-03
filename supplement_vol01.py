#!/usr/bin/env python3
"""为第一卷(斗气)补充NumPy进阶和Matplotlib可视化内容"""

filepath = "/Users/gatilin/PycharmProjects/FlameSutra-books/01-筑基篇-斗之气/README.md"
content_file = "/Users/gatilin/PycharmProjects/FlameSutra-books/supplement_vol01_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

with open(content_file, 'r', encoding='utf-8') as f:
    new_content = f.read()

insert_marker = "## 附录：推荐修炼资源"
insert_idx = content.index(insert_marker)

content = content[:insert_idx] + new_content + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

new_lines = len(new_content.splitlines())
total_lines = len(content.splitlines())
print("OK: added %d lines, total now %d" % (new_lines, total_lines))
