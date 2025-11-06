from bs4 import BeautifulSoup

html = open('test_case.html', 'r', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

# 查找所有包含武器名的链接
weapons = []
for a in soup.find_all('a'):
    text = a.get_text(strip=True)
    if '|' in text or any(w in text for w in ['AK-47', 'AWP', 'Desert Eagle', 'M4A1-S', 'M4A4']):
        weapons.append(text)
        # 查看父元素结构
        parent_chain = []
        p = a.parent
        for _ in range(5):
            if p:
                parent_chain.append(f"{p.name}({p.get('class', [])})")
                p = p.parent
        if len(weapons) <= 5:
            print(f"武器: {text}")
            print(f"  父元素链: {' <- '.join(parent_chain)}")
            print()

print(f'\n总共找到 {len(weapons)} 个武器相关链接')
print('\n前 20 个:')
for w in weapons[:20]:
    print(f'  - {w}')
