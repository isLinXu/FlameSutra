#!/usr/bin/env python3
"""第07卷(空间篇-斗宗)补充脚本"""

filepath = "07-空间篇-斗宗/README.md"
content_path = "supplement_vol07_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\r\n', '\n')

with open(content_path, 'r', encoding='utf-8') as f:
    new_content = f.read()

insert_marker = "## 修炼总结与境界突破条件"
if insert_marker not in content:
    print("ERROR: Cannot find insertion point")
    exit(1)

insert_idx = content.index(insert_marker)
content = content[:insert_idx] + new_content + "\n\n" + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done! New lines: {len(new_content.splitlines())}")
print(f"Total lines: {len(content.splitlines())}")
