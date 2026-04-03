#!/usr/bin/env python3
"""万火录补充脚本"""

filepath = "万火录-GPU异火榜/README.md"
content_path = "supplement_huofire_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\r\n', '\n')

with open(content_path, 'r', encoding='utf-8') as f:
    new_content = f.read()

# Insert before the final closing line
insert_marker = "异火选购新趋势（2025）：能效比优先、显存为王、国产替代加速、云端按需租赁。"
if insert_marker not in content:
    print("ERROR: Cannot find insertion point")
    exit(1)

insert_idx = content.index(insert_marker) + len(insert_marker)
content = content[:insert_idx] + "\n\n" + new_content + "\n" + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done! New lines: {len(new_content.splitlines())}")
print(f"Total lines: {len(content.splitlines())}")
