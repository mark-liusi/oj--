#!/usr/bin/env python3
"""将 run_best_list_pipeline 相关函数移到 main() 之前"""
import sys

# 读取文件
with open("calculate_scientific.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"总行数: {len(lines)}", flush=True)
sys.stdout.flush()

# 提取第539-991行（Python的list是0-based，所以是538:991）
code_block = lines[538:991]  # 第539行到第991行
print(f"提取代码块: {len(code_block)} 行")

# 删除第539-991行
del lines[538:991]
print(f"删除后总行数: {len(lines)}")

# 在第265行之后插入（第266行之前，即index=265）
# 添加分隔符和代码块
separator = [
    "\n",
    "# ============================================================\n",
    "# 追加模块：每个系列（箱子）\"最佳下级（主料）清单\"\n",
    "# ============================================================\n",
    "\n"
]

insert_pos = 265  # 在第266行之前插入
lines[insert_pos:insert_pos] = separator + code_block
print(f"插入后总行数: {len(lines)}")

# 写回文件
with open("calculate_scientific.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✓ 成功重组文件！")
print(f"  - 原 main() 在第269行")
print(f"  - 代码块已移动到第266-{266+len(separator)+len(code_block)-1}行之间")
print(f"  - 新 main() 在第{266+len(separator)+len(code_block)+3}行左右")
