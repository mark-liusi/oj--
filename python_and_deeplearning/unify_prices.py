"""
统一处理prices_all.csv（全平台价格数据），确保所有数据都有必需的列
支持从多平台数据中自动选择最低价
"""
import pandas as pd
import re

# 读取数据（优先读取全平台数据，兜底使用min版本）
try:
    df = pd.read_csv('prices_all.csv')
    print(f"✅ 读取全平台数据: prices_all.csv")
except FileNotFoundError:
    df = pd.read_csv('prices_all_min.csv')
    print(f"⚠️  使用备用数据: prices_all_min.csv")

print(f"原始数据: {len(df)}行")

# 删除没有价格的行
df = df[df['price'].notna() & (df['price'] > 0)]
print(f"过滤无效价格后: {len(df)}行")

# 添加缺失的列
if 'name' not in df.columns:
    df['name'] = None
if 'exterior' not in df.columns:
    df['exterior'] = None
if 'market_hash_name' not in df.columns:
    df['market_hash_name'] = None
if 'series' not in df.columns:
    df['series'] = None
if 'tier' not in df.columns:
    df['tier'] = None

# 外观映射
exterior_map = {
    'Factory New': 'FN',
    'Minimal Wear': 'MW',
    'Field-Tested': 'FT',
    'Well-Worn': 'WW',
    'Battle-Scarred': 'BS'
}

# 处理缺失name和exterior的行(从marketHashName_used提取)
for idx, row in df.iterrows():
    if pd.isna(row['name']) and pd.notna(row['marketHashName_used']):
        hash_name = str(row['marketHashName_used']).strip()
        
        # 提取name和exterior
        extracted_name = hash_name
        extracted_exterior = None
        
        for full, abbr in exterior_map.items():
            if f'({full})' in hash_name:
                extracted_name = hash_name.replace(f' ({full})', '').strip()
                extracted_exterior = abbr
                break
        
        df.at[idx, 'name'] = extracted_name
        df.at[idx, 'exterior'] = extracted_exterior
        df.at[idx, 'market_hash_name'] = hash_name

# 处理缺失series和tier的行(从rarity_en和case_name_en补充)
for idx, row in df.iterrows():
    if pd.isna(row['series']) and pd.notna(row['case_name_en']):
        df.at[idx, 'series'] = row['case_name_en']
    
    if pd.isna(row['tier']) and pd.notna(row['rarity_en']):
        df.at[idx, 'tier'] = row['rarity_en']

# 再次过滤:必须有name, price, platform
df = df[df['name'].notna() & df['price'].notna() & df['platform'].notna()]
print(f"过滤缺失关键字段后: {len(df)}行")

# ===== 新增：从多平台数据中选择每个物品的最低价 =====
print(f"\n开始选择最低价...")
# 按 name+exterior 分组，每组选择价格最低的一条
df_grouped = df.groupby(['name', 'exterior'], as_index=False).apply(
    lambda x: x.loc[x['price'].idxmin()]
).reset_index(drop=True)

print(f"去重后物品数: {len(df_grouped)}行 (原始{len(df)}行)")
print(f"  平均每个物品有 {len(df)/len(df_grouped):.1f} 个平台的价格")

# 选择需要的列并重命名
df_final = df_grouped[['name', 'series', 'tier', 'price', 'exterior', 'platform']].copy()

# 保存
df_final.to_csv('prices_with_exterior.csv', index=False)
print(f"\n✅ 已生成 prices_with_exterior.csv ({len(df_final)} 行)")

# 统计
print(f"\n统计信息:")
print(f"  唯一物品数: {df_final['name'].nunique()}")
print(f"  各外观分布:")
if 'exterior' in df_final.columns:
    print(df_final['exterior'].value_counts())
print(f"  各平台分布:")
print(df_final['platform'].value_counts())

# 显示8个修正物品
items = [
    'Galil AR | Connexion',
    'Glock-18 | Neo-Noir',
    'P250 | Contaminant',
    'M4A4 | Griffin',
    'XM1014 | Incinegator',
    'P90 | Neoqueen',
    'AWP | Duality',
    'R8 Revolver | Grip'
]
result = df_final[df_final['name'].isin(items)]
print(f"\n8个修正物品数据: {len(result)}条")
for item in items:
    count = len(result[result['name'] == item])
    print(f"  {item}: {count}条")
