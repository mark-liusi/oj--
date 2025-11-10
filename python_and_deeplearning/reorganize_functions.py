#!/usr/bin/env python3
"""重组 calculate_scientific.py，将 run_best_list_pipeline 相关函数移到 main() 之前"""

import re

# 读取原文件
with open("calculate_scientific.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 找到需要移动的函数块（从 MaterialCandidate 定义开始到 run_best_list_pipeline 结束）
# 使用正则表达式提取整个模块

# 找到 MaterialCandidate 类定义的起始位置
match_start = re.search(r'^@dataclass\s*\nclass MaterialCandidate:', content, re.MULTILINE)
if not match_start:
    print("错误：找不到 MaterialCandidate 类定义")
    exit(1)

start_pos = match_start.start()
print(f"✓ 找到 MaterialCandidate 起始位置: {start_pos}")

# 找到 run_best_list_pipeline 函数的结束位置（下一个模块分隔符或文件结尾）
# 查找函数定义后的返回语句
match_end = re.search(r'(def run_best_list_pipeline\(.*?\n(?:.*?\n)*?    return results\s*\n)', content[start_pos:], re.DOTALL)
if not match_end:
    print("错误：找不到 run_best_list_pipeline 函数结束")
    exit(1)

end_pos = start_pos + match_end.end()
print(f"✓ 找到 run_best_list_pipeline 结束位置: {end_pos}")

# 提取要移动的代码块
functions_to_move = content[start_pos:end_pos]
print(f"✓ 提取代码块长度: {len(functions_to_move)} 字符")

# 2. 找到插入位置（# ------------------------------ 主流程 之前）
insert_match = re.search(r'\n(# -{20,}\n# 主流程\n# -{20,}\n)', content)
if not insert_match:
    print("错误：找不到主流程分隔符")
    exit(1)

insert_pos = insert_match.start()
print(f"✓ 找到插入位置（主流程分隔符之前）: {insert_pos}")

# 3. 构建新内容
new_content = (
    content[:insert_pos] + 
    "\n\n# ============================================================\n" +
    "# 追加模块：每个系列（箱子）\"最佳下级（主料）清单\"\n" +
    "# ============================================================\n\n" +
    functions_to_move + 
    "\n\n" +
    content[insert_pos:start_pos] + 
    content[end_pos:]
)

# 4. 清理重复的模块标题
new_content = re.sub(
    r'\n# ={60,}\n# 追加模块：每个系列（箱子）\"最佳下级（主料）清单\"\n# ={60,}\n+@dataclass',
    '\n@dataclass',
    new_content
)

# 5. 写回文件
with open("calculate_scientific.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("✓ 成功重组文件，run_best_list_pipeline 已移动到 main() 之前")
