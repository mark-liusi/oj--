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
for i, g in enumerate(galleries[:10]):  # 只看前10个
    print(f'--- 项目 {i} ---')
    # 查找整个 caption div
    caption = g.find('div', class_='lightbox-caption')
    if caption:
        print(f'Caption HTML: {caption}')
        # 查找所有文本
        text = caption.get_text(' ', strip=True)
        print(f'Caption 文本: {text}')
    print()
