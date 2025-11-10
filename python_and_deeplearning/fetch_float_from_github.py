# -*- coding: utf-8 -*-
"""
从GitHub公开数据源获取CS2皮肤浮漂区间
使用: https://github.com/ByMykel/CSGO-API (CS2皮肤数据API)
"""

import argparse
import json
import pandas as pd
import requests
from tqdm import tqdm

def fetch_csgo_skins_data():
    """
    从GitHub公开API获取所有CS2皮肤数据
    """
    print("正在从GitHub API获取CS2皮肤数据...")
    
    urls = [
        "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/skins.json",
        "https://bymykel.github.io/CSGO-API/api/en/skins.json",
    ]
    
    for url in urls:
        try:
            print(f"尝试: {url}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                print(f"成功获取数据!")
                return data
        except Exception as e:
            print(f"  失败: {e}")
            continue
    
    return None

def normalize_name(name: str) -> str:
    """标准化皮肤名称用于匹配"""
    # 移除特殊字符,统一大小写
    return name.lower().strip()

def match_skin(item_name: str, skins_data: list) -> dict:
    """
    在皮肤数据中匹配物品
    """
    item_normalized = normalize_name(item_name)
    
    for skin in skins_data:
        if not isinstance(skin, dict):
            continue
            
        # 构造皮肤全名
        weapon = skin.get('weapon', {})
        pattern = skin.get('pattern', {})
        
        if weapon and isinstance(weapon, dict) and pattern and isinstance(pattern, dict):
            weapon_name = weapon.get('name', '')
            pattern_name = pattern.get('name', '')
            
            if weapon_name and pattern_name:
                skin_name = f"{weapon_name} | {pattern_name}"
                if normalize_name(skin_name) == item_normalized:
                    return skin
        
        # 也尝试直接匹配name字段
        if 'name' in skin:
            if normalize_name(skin['name']) == item_normalized:
                return skin
    
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", "-i", required=True, help="物品CSV文件")
    ap.add_argument("--output", "-o", default="skins_meta_complete.csv", help="输出文件")
    args = ap.parse_args()
    
    # 获取GitHub数据
    skins_data = fetch_csgo_skins_data()
    if not skins_data:
        print("ERROR: 无法获取皮肤数据")
        return 1
    
    print(f"成功获取 {len(skins_data)} 个皮肤的数据")
    
    # 读取物品列表
    df = pd.read_csv(args.items)
    
    # 查找name列
    name_col = None
    for col in ['name', 'item_name_en', '物品名称']:
        if col in df.columns:
            name_col = col
            break
    
    if not name_col:
        print("ERROR: 未找到name列")
        return 1
    
    # 获取唯一物品
    unique_names = df[name_col].dropna().unique()
    print(f"需要处理 {len(unique_names)} 个唯一物品")
    
    # 匹配并提取浮漂区间
    results = []
    matched_count = 0
    
    for name in tqdm(unique_names, desc="匹配皮肤"):
        matched_skin = match_skin(name, skins_data)
        
        if matched_skin and 'min_float' in matched_skin and 'max_float' in matched_skin:
            results.append({
                'name': name,
                'float_min': matched_skin['min_float'],
                'float_max': matched_skin['max_float'],
                'source': 'csgo_api_github'
            })
            matched_count += 1
        else:
            # 未匹配到,使用默认值
            # 判断物品类型
            if any(x in name for x in ['★', 'Knife', 'Gloves']):
                min_f, max_f = 0.06, 0.80  # 刀具/手套
            elif 'Asiimov' in name:
                min_f, max_f = 0.18, 1.00  # Asiimov系列
            else:
                min_f, max_f = 0.00, 1.00  # 默认范围
            
            results.append({
                'name': name,
                'float_min': min_f,
                'float_max': max_f,
                'source': 'default_estimate'
            })
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    
    print(f"\n完成!")
    print(f"总计: {len(result_df)} 个皮肤")
    print(f"成功匹配: {matched_count} ({matched_count/len(result_df)*100:.1f}%)")
    print(f"使用默认: {len(result_df)-matched_count}")
    print(f"已保存到: {args.output}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
