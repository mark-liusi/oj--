# -*- coding: utf-8 -*-
"""
从 prices_all_min.csv 提取外观信息，生成适合科学版计算的格式
"""
import pandas as pd
import re

# 读取价格数据
df = pd.read_csv('prices_all_min.csv')

# 从 marketHashName_used 中提取物品名和外观
def extract_name_and_exterior(market_hash_name):
    """
    从 'AK-47 | Elite Build (Factory New)' 提取:
    - name: 'AK-47 | Elite Build'
    - exterior: 'FN'
    """
    if pd.isna(market_hash_name):
        return None, None
    
    s = str(market_hash_name).strip()
    
    # 外观映射
    exterior_map = {
        'Factory New': 'FN',
        'Minimal Wear': 'MW',
        'Field-Tested': 'FT',
        'Well-Worn': 'WW',
        'Battle-Scarred': 'BS'
    }
    
    # 尝试匹配外观
    exterior = None
    for full, abbr in exterior_map.items():
        if f'({full})' in s:
            exterior = abbr
            # 去掉外观部分得到物品名
            name = s.replace(f' ({full})', '').strip()
            return name, exterior
    
    # 没有外观（可能是刀具或其他）
    return s, None

# 提取信息
df[['name', 'exterior']] = df['marketHashName_used'].apply(
    lambda x: pd.Series(extract_name_and_exterior(x))
)

# 重命名列以适配 calculate.py
df_for_calc = df.rename(columns={
    'rarity_en': 'tier',
    'case_name_en': 'series'
})

# 选择需要的列
df_for_calc = df_for_calc[['name', 'series', 'tier', 'price', 'exterior', 'platform']]

# 保存
df_for_calc.to_csv('prices_with_exterior.csv', index=False, encoding='utf-8-sig')
print(f"✅ 已生成 prices_with_exterior.csv ({len(df_for_calc)} 行)")
print(f"\n前5行预览:")
print(df_for_calc.head())

# 统计外观分布
print(f"\n外观分布:")
print(df_for_calc['exterior'].value_counts())
