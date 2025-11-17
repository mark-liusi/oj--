#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成完整的中英文物品名称映射表
从 cs2_case_items_full.csv 读取所有物品，通过 Steam 社区市场或 BUFF API 获取中文名称

使用方法：
    python generate_cn_name_mapping.py --source buff    # 从 BUFF 获取（需要 cookies）
    python generate_cn_name_mapping.py --source steam   # 从 Steam 社区市场获取（推荐）
    python generate_cn_name_mapping.py --manual         # 使用手动维护的映射表
"""

import argparse
import pandas as pd
import requests
import time
from pathlib import Path
from typing import Dict, Optional, List
import json
from tqdm import tqdm

# Steam 社区市场 API（支持多语言）
STEAM_MARKET_SEARCH = "https://steamcommunity.com/market/search/render/"

# 外观英文->中文映射
EXTERIOR_CN = {
    "Factory New": "崭新出厂",
    "Minimal Wear": "略有磨损", 
    "Field-Tested": "久经沙场",
    "Well-Worn": "破损不堪",
    "Battle-Scarred": "战痕累累"
}

WEARS = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]


def get_cn_name_from_steam(item_name_en: str, max_retries: int = 3) -> Optional[str]:
    """
    通过 Steam 社区市场搜索接口获取中文名称
    Steam 支持通过 l=schinese 参数获取简体中文结果
    """
    # 尝试不同的搜索策略
    search_queries = [
        item_name_en,  # 完整名称
        item_name_en.split(" | ")[0] if " | " in item_name_en else item_name_en,  # 只搜武器名
    ]
    
    for query in search_queries:
        for attempt in range(max_retries):
            try:
                params = {
                    "query": query,
                    "start": 0,
                    "count": 10,
                    "search_descriptions": 0,
                    "sort_column": "popular",
                    "sort_dir": "desc",
                    "appid": 730,  # CS2
                    "norender": 1,
                    "l": "schinese"  # 简体中文
                }
                
                response = requests.get(STEAM_MARKET_SEARCH, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # 查找匹配的结果
                    for result in results:
                        hash_name = result.get("hash_name", "")
                        name = result.get("name", "")
                        
                        # 如果英文名称匹配，返回中文名称
                        if item_name_en.lower() in hash_name.lower():
                            # name 字段是中文名称
                            if name and name != hash_name:
                                return name
                    
                    # 如果没有完全匹配，返回第一个结果的名称
                    if results and len(results) > 0:
                        first_name = results[0].get("name", "")
                        if first_name:
                            return first_name
                
                # 429 Too Many Requests - 等待后重试
                if response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print(f"  ⚠ 获取失败 {item_name_en}: {e}")
    
    return None


def get_cn_name_from_buff(item_name_en: str, cookies: Dict[str, str]) -> Optional[str]:
    """
    通过 BUFF API 获取中文名称（需要登录 cookies）
    """
    try:
        # BUFF 搜索 API
        url = "https://buff.163.com/api/market/search"
        params = {
            "game": "csgo",
            "search": item_name_en,
            "page_num": 1,
            "page_size": 10
        }
        
        response = requests.get(url, params=params, cookies=cookies, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("data", {}).get("items", [])
            
            for item in items:
                # BUFF 的 name 字段是中文名
                cn_name = item.get("name", "")
                en_name = item.get("market_hash_name", "")
                
                # 匹配英文名称
                if item_name_en.lower() in en_name.lower():
                    return cn_name
        
    except Exception as e:
        print(f"  ⚠ BUFF API 错误 {item_name_en}: {e}")
    
    return None


def generate_manual_mapping() -> Dict[str, str]:
    """
    生成手动维护的常见物品映射表
    这些是最常见的交易物品，可以手动维护
    """
    manual_map = {
        # AK-47 系列
        "AK-47 | Redline": "AK-47 | 红线",
        "AK-47 | Asiimov": "AK-47 | 二西莫夫",
        "AK-47 | Fire Serpent": "AK-47 | 火蛇",
        "AK-47 | Vulcan": "AK-47 | 火神",
        "AK-47 | Case Hardened": "AK-47 | 案例硬化",
        
        # AWP 系列
        "AWP | Asiimov": "AWP | 二西莫夫",
        "AWP | Dragon Lore": "AWP | 龙狙",
        "AWP | Lightning Strike": "AWP | 闪电突击",
        "AWP | Hyper Beast": "AWP | 超凡野兽",
        
        # M4A4 系列
        "M4A4 | Asiimov": "M4A4 | 二西莫夫",
        "M4A4 | Howl": "M4A4 | 嚎叫",
        "M4A4 | The Emperor": "M4A4 | 帝王",
        
        # M4A1-S 系列
        "M4A1-S | Hot Rod": "M4A1-S | 烈焰神驹",
        "M4A1-S | Golden Coil": "M4A1-S | 黄金圈",
        
        # Desert Eagle 系列
        "Desert Eagle | Blaze": "沙漠之鹰 | 烈焰",
        "Desert Eagle | Code Red": "沙漠之鹰 | 暗红代码",
        
        # Glock-18 系列
        "Glock-18 | Water Elemental": "格洛克18型 | 水元素",
        "Glock-18 | Fade": "格洛克18型 | 渐变之色",
        
        # USP-S 系列
        "USP-S | Kill Confirmed": "USP-S | 确认击杀",
        "USP-S | Neo-Noir": "USP-S | 黑色影像",
    }
    
    return manual_map


def main():
    parser = argparse.ArgumentParser(description="生成中英文物品名称映射表")
    parser.add_argument("--source", choices=["steam", "buff", "manual"], default="steam",
                        help="数据来源：steam（推荐）/ buff（需要cookies）/ manual（手动映射）")
    parser.add_argument("--input", default="data/cs2_case_items_full.csv",
                        help="输入文件路径")
    parser.add_argument("--output", default="data/name_mapping.csv",
                        help="输出文件路径")
    parser.add_argument("--buff-cookies", default=None,
                        help="BUFF cookies JSON 文件路径（仅 --source buff 时需要）")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="请求间隔（秒），避免被限流")
    
    args = parser.parse_args()
    
    # 读取物品清单
    print(f"📖 读取物品清单: {args.input}")
    df = pd.read_csv(args.input, encoding="utf-8")
    
    # 提取唯一的英文物品名
    if "item_name_en" not in df.columns:
        print("❌ 错误：输入文件缺少 item_name_en 列")
        return
    
    items = df["item_name_en"].dropna().unique()
    print(f"✅ 找到 {len(items)} 个唯一物品")
    
    # 生成映射
    mapping = {}
    
    if args.source == "manual":
        print("📝 使用手动维护的映射表")
        mapping = generate_manual_mapping()
        
    elif args.source == "steam":
        print("🌐 从 Steam 社区市场获取中文名称...")
        print(f"   请求间隔: {args.rate_limit} 秒")
        
        for item in tqdm(items, desc="获取中文名称"):
            cn_name = get_cn_name_from_steam(item)
            if cn_name:
                mapping[item] = cn_name
            time.sleep(args.rate_limit)
    
    elif args.source == "buff":
        if not args.buff_cookies:
            print("❌ 错误：--source buff 需要提供 --buff-cookies 参数")
            return
        
        print(f"🌐 从 BUFF 获取中文名称（使用 cookies: {args.buff_cookies}）...")
        
        with open(args.buff_cookies, "r") as f:
            cookies = json.load(f)
        
        for item in tqdm(items, desc="获取中文名称"):
            cn_name = get_cn_name_from_buff(item, cookies)
            if cn_name:
                mapping[item] = cn_name
            time.sleep(args.rate_limit)
    
    # 生成完整的映射表（包含外观变体）
    print("\n📋 生成完整映射表（包含外观变体）...")
    full_mapping = []
    
    for item_en, item_cn in mapping.items():
        # 基础名称映射
        full_mapping.append({
            "name": item_en,
            "market_hash_name": item_cn
        })
        
        # 为每个外观生成映射（用于市场搜索）
        for wear_en, wear_cn in EXTERIOR_CN.items():
            full_name_en = f"{item_en} ({wear_en})"
            full_name_cn = f"{item_cn} ({wear_cn})"
            full_mapping.append({
                "name": full_name_en,
                "market_hash_name": full_name_cn
            })
    
    # 保存映射表
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_mapping = pd.DataFrame(full_mapping)
    df_mapping.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n✅ 已生成映射表: {output_path}")
    print(f"   基础物品: {len(mapping)} 个")
    print(f"   总映射数: {len(full_mapping)} 条（含外观变体）")
    
    # 显示示例
    print("\n📝 映射示例（前10条）:")
    print(df_mapping.head(10).to_string(index=False))
    
    # 统计未映射的物品
    unmapped = set(items) - set(mapping.keys())
    if unmapped:
        print(f"\n⚠️  {len(unmapped)} 个物品未获取到中文名称:")
        for item in list(unmapped)[:10]:
            print(f"   - {item}")
        if len(unmapped) > 10:
            print(f"   ... 还有 {len(unmapped) - 10} 个")


if __name__ == "__main__":
    main()
