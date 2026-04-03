#!/usr/bin/env python3
"""药材库补充脚本"""

filepath = "药材库-数据集指南/README.md"
content_path = "supplement_dataset_content.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\r\n', '\n')

with open(content_path, 'r', encoding='utf-8') as f:
    new_content = f.read()

# Insert before the final closing line
insert_marker = "数据质量铁律：1万条高质量 > 100万条低质量；多样性比数量更重要；GPT-4 标注效果最佳。"
if insert_marker not in content:
    print("ERROR: Cannot find insertion point")
    exit(1)

insert_idx = content.index(insert_marker) + len(insert_marker)
content = content[:insert_idx] + "\n" + new_content + "\n" + content[insert_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done! New lines: {len(new_content.splitlines())}")
print(f"Total lines: {len(content.splitlines())}")
