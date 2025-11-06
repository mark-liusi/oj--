from bs4 import BeautifulSoup

html = open('test_case.html', 'r', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

# 查找所有表格
tables = soup.find_all('table')
print(f'找到 {len(tables)} 个表格\n')

for i, t in enumerate(tables[:3]):
    cls = t.get('class', [])
    print(f'表格 {i}: class={cls}')
    # 查找表格中的行
    rows = t.find_all('tr')[:3]
    for j, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        if cells:
            print(f'  行 {j}: {[c.get_text(strip=True)[:30] for c in cells[:3]]}')
    print()

# 查找包含武器名的元素
print('\n查找包含 "AK-47" 或 "AWP" 的元素:')
for elem in soup.find_all(string=lambda t: t and ('AK-47' in t or 'AWP' in t or 'Desert Eagle' in t))[:5]:
    parent = elem.parent
    print(f'{parent.name}: {elem.strip()[:60]}')
