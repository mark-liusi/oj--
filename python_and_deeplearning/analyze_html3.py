from bs4 import BeautifulSoup

html = open('test_case.html', 'r', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

# 找到所有 gallery 项
galleries = soup.find_all('div', class_='wikia-gallery-item')
print(f'找到 {len(galleries)} 个图库项目\n')

rarity_map = {
    'rare': 'Covert',
    'mythical': 'Classified',  
    'ancient': 'Restricted',
    'legendary': 'Gold'
}

items = []
for g in galleries:
    # 查找稀有度 span
    rarity_span = g.find('span', class_=lambda x: x and any(r in x for r in ['rare', 'mythical', 'ancient', 'legendary']))
    if not rarity_span:
        continue
    
    rarity_class = [c for c in rarity_span.get('class', []) if c in rarity_map]
    rarity = rarity_map.get(rarity_class[0] if rarity_class else '', 'Unknown')
    
    # 查找武器名 - 可能在 <a> 标签中
    name_link = g.find('a', class_='image')
    if name_link:
        # 武器名通常在标题中
        title = name_link.get('title', '')
        items.append((title, rarity))

print('找到的物品:')
for name, rarity in items:
    print(f'  [{rarity}] {name}')
