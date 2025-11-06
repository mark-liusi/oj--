# 测试修复后的解析函数
import sys
sys.path.insert(0, '.')
# 导入修改后的函数
exec(open('2.py', 'r', encoding='utf-8').read())

# 读取测试 HTML
html = open('test_case.html', 'r', encoding='utf-8').read()

# 解析
items, errors = parse_case_html(html)

print(f'✅ 找到 {len(items)} 个物品')
if errors:
    print(f'⚠️ 错误: {errors}')

print('\n按稀有度统计:')
from collections import Counter
rarity_count = Counter(it['rarity_en'] for it in items)
for rarity in ['Covert', 'Classified', 'Restricted', 'Gold']:
    print(f'  {rarity}: {rarity_count.get(rarity, 0)} 个')

print('\n前 15 个物品:')
for it in items[:15]:
    print(f'  [{it["rarity_en"]}] {it["weapon"]} | {it["finish"]}')
