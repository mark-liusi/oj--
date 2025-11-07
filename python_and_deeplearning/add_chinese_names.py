# -*- coding: utf-8 -*-
"""测试从 SteamDT 获取中文名称"""

import os
import pandas as pd
import requests

os.environ["STEAMDT_API_KEY"] = "376ad07a755d4e1bb2ce192c47c51028"
API_KEY = "376ad07a755d4e1bb2ce192c47c51028"
STEAMDT_BASE = "https://open.steamdt.com"

# 读取现有的数据
df = pd.read_csv("cs2_case_items_full.csv", encoding="utf-8-sig")
print(f"总共 {len(df)} 个物品")
print("\n原始数据（前5行）:")
print(df[['item_name_en', 'weapon', 'finish', 'rarity_zh']].head())

# 获取 SteamDT 基础数据
print("\n正在获取 SteamDT 数据...")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

r = requests.get(f"{STEAMDT_BASE}/open/cs2/v1/base", headers=headers, timeout=60)
data = r.json()

if data.get("code") == 200:
    items = data.get("data", [])
    print(f"✅ 获取到 {len(items)} 个 SteamDT 物品数据")
    
    # 建立映射：英文市场名 -> 中文名
    name_map = {}
    for item in items:
        mhn = item.get("marketHashName", "")
        name_zh = item.get("name", "")
        if mhn and name_zh:
            # 提取基础名称（去掉磨损度）
            base_mhn = mhn.split(" (")[0] if " (" in mhn else mhn
            if base_mhn not in name_map:
                name_map[base_mhn] = name_zh.split(" (")[0] if " (" in name_zh else name_zh
    
    print(f"建立了 {len(name_map)} 个映射关系")
    print("\n示例映射:")
    for i, (k, v) in enumerate(list(name_map.items())[:5]):
        print(f"  {k} -> {v}")
    
    # 为 DataFrame 添加中文名
    def get_chinese_name(row):
        weapon = str(row.get("weapon", "")).strip()
        finish = str(row.get("finish", "")).strip()
        
        if weapon and finish:
            base_name = f"{weapon} | {finish}"
        elif weapon:
            base_name = weapon
        else:
            base_name = ""
        
        return name_map.get(base_name, "")
    
    df["item_name_zh"] = df.apply(get_chinese_name, axis=1)
    
    # 统计匹配情况
    matched = df["item_name_zh"].notna() & (df["item_name_zh"] != "")
    print(f"\n匹配结果: {matched.sum()}/{len(df)} ({matched.sum()/len(df)*100:.1f}%)")
    
    print("\n添加中文名后（前10行）:")
    print(df[['item_name_en', 'item_name_zh', 'rarity_zh']].head(10).to_string(index=False))
    
    # 保存结果
    df.to_csv("cs2_case_items_full_zh.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存到 cs2_case_items_full_zh.csv")
else:
    print(f"❌ API 错误: {data}")
