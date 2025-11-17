#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-fetch CS2 skin prices using SteamDT OpenAPI.

Usage:
  export STEAMDT_API_KEY=xxxxx
  python fetch_prices_with_steamdt.py --items-csv data/cs2_case_items_full.csv --out prices_today.csv --platform steam --mode current
  python fetch_prices_with_steamdt.py --items-csv data/cs2_case_items_full.csv --out prices_avg7d.csv --platform steam --mode avg7d

Columns expected in --items-csv:
  - item_name_en, weapon, finish, steamdt_market_hash_name (optional)

Notes:
  * We generate marketHashName candidates as "Weapon | Finish (WEAR)".
  * SteamDT endpoints used:
      - GET  /open/cs2/v1/price/single?marketHashName=...           (current listings) 
      - POST /open/cs2/v1/price/batch {"marketHashNames":[...]}     (current listings batch)
      - GET  /open/cs2/v1/price/avg?marketHashName=...              (last 7 days average)
    See docs: https://doc.steamdt.com/  (requires Authorization Bearer key)
"""
import os, sys, time, argparse, re
from typing import List, Dict, Any
import pandas as pd
import requests
from tqdm import tqdm

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = "https://open.steamdt.com"
def HEADERS(api_key): 
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

WEARS = ["Factory New","Minimal Wear","Field-Tested","Well-Worn","Battle-Scarred"]

def is_gold_row(row: pd.Series) -> bool:
    """判断是否为刀具/手套（Gold 稀有度）"""
    name = str(row.get("item_name_en", ""))
    rarity = str(row.get("rarity_en", ""))
    return ("★" in name) or ("Knife" in name) or ("Gloves" in name) or (rarity == "Gold")

def normalize_finish_for_hash(finish: str) -> str:
    """
    处理多普勒相位/宝石问题
    Steam 的 market_hash_name 是 "★ Bayonet | Doppler (Factory New)"
    不包含 (Sapphire/Ruby/Phase 1) 等相位信息
    """
    if "Doppler" in finish:
        # 去掉 (Sapphire/Ruby/Black Pearl/Phase N) 等相位
        finish = re.sub(r"Doppler\s*\([^)]+\)", "Doppler", finish).strip()
    return finish

def norm_base_name(weapon: str, finish: str) -> str:
    # 处理 NaN 或其他非字符串类型
    weapon = str(weapon) if pd.notna(weapon) else ""
    finish = str(finish) if pd.notna(finish) else ""
    weapon = weapon.strip()
    finish = finish.strip()
    if weapon and finish:
        return f"{weapon} | {finish}"
    return weapon or finish

def build_candidates(row: pd.Series) -> List[str]:
    """
    生成 marketHashName 候选列表
    关键修复：
    1. Gold（刀/手套）必须带磨损后缀
    2. Doppler 去掉相位/宝石信息
    """
    mh = row.get("steamdt_market_hash_name")
    if pd.notna(mh) and str(mh).strip():
        return [str(mh).strip()]
    
    # 获取武器和涂装
    weapon = row.get("weapon", "")
    finish = row.get("finish", "")
    
    # 从 item_name_en 提取武器和涂装（兜底）
    if not weapon or not finish:
        item = row.get("item_name_en")
        if pd.notna(item):
            item = str(item).strip()
            if " | " in item:
                try:
                    weapon, finish = item.split(" | ", 1)
                except ValueError:
                    pass
    
    # 处理 NaN
    weapon = str(weapon) if pd.notna(weapon) else ""
    finish = str(finish) if pd.notna(finish) else ""
    weapon = weapon.strip()
    finish = finish.strip()
    
    if not weapon or not finish:
        return []
    
    # 多普勒相位处理：去掉 (Sapphire/Ruby/Phase N)
    finish_normalized = normalize_finish_for_hash(finish)
    
    # 生成候选列表（所有物品都需要磨损后缀）
    candidates = []
    for wear in WEARS:
        candidates.append(f"{weapon} | {finish_normalized} ({wear})")
    
    # 也尝试不带磨损的版本（某些特殊情况）
    candidates.append(f"{weapon} | {finish_normalized}")
    
    return candidates

def price_from_single(api_key: str, market_hash: str, platform: str):
    """
    若 platform == 'all'：返回该 market_hash 下所有平台的价格列表
    否则：只在返回列表里精确匹配目标平台；匹配不到就返回 price=None（不再回落到第一条）。
    这样主循环可以继续尝试下一个候选 marketHashName。
    """
    url = f"{BASE}/open/cs2/v1/price/single"
    params = {"marketHashName": market_hash}
    r = requests.get(url, params=params, headers=HEADERS(api_key), timeout=25)
    if r.status_code != 200:
        base = {"marketHashName": market_hash, "source": "single", "status": r.status_code}
        return [dict(base, price=None, platform=platform)] if platform.lower()=="all" else dict(base, price=None, platform=platform)

    j = r.json()
    data = j.get("data", []) or []

    # 平台统一小写用于比较，但输出保留原样
    if platform.lower().strip() == "all":
        rows = []
        for d in data:
            rows.append({
                "marketHashName": market_hash,
                "price": d.get("sellPrice"),
                "platform": d.get("platform", ""),
                "source": "single",
                "status": 200,
            })
        return rows
    else:
        target = platform.lower().strip()
        # 只在真正匹配到目标平台时返回价格；否则返回 price=None
        for d in data:
            plat = str(d.get("platform",""))
            if plat.lower() == target or target in plat.lower():
                return {"marketHashName": market_hash, "price": d.get("sellPrice"),
                        "platform": plat, "source": "single", "status": 200}
        # 没匹配到就显式为空，主循环会继续尝试下一个候选
        return {"marketHashName": market_hash, "price": None,
                "platform": platform, "source": "single", "status": 200}

def price_avg7d(api_key: str, market_hash: str, platform: str):
    """
    若 platform == 'all'：返回 dataList 里每个平台的7日均价
    否则：只返回目标平台的均价；匹配不到则 price=None（不再用总均价兜底）
    """
    url = f"{BASE}/open/cs2/v1/price/avg"
    params = {"marketHashName": market_hash}
    r = requests.get(url, params=params, headers=HEADERS(api_key), timeout=25)
    if r.status_code != 200:
        base = {"marketHashName": market_hash, "source": "avg7d", "status": r.status_code}
        return [dict(base, price=None, platform=platform)] if platform.lower()=="all" else dict(base, price=None, platform=platform)

    j = r.json()
    data = j.get("data", {}) or {}
    data_list = data.get("dataList", []) or []

    if platform.lower().strip() == "all":
        rows = []
        for d in data_list:
            rows.append({
                "marketHashName": market_hash,
                "price": d.get("avgPrice"),
                "platform": d.get("platform", ""),
                "source": "avg7d",
                "status": 200,
            })
        return rows
    else:
        target = platform.lower().strip()
        for d in data_list:
            plat = str(d.get("platform",""))
            if plat.lower() == target or target in plat.lower():
                return {"marketHashName": market_hash, "price": d.get("avgPrice"),
                        "platform": plat, "source": "avg7d", "status": 200}
        # 没匹配到就显式为空
        return {"marketHashName": market_hash, "price": None,
                "platform": platform, "source": "avg7d", "status": 200}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", default="steam",
                    help="价格平台：steam / buff / skinport ... 或 all（收集所有平台）")
    ap.add_argument("--mode", choices=["current","avg7d"], default="current")
    ap.add_argument("--min-only", action="store_true",
                    help="若指定，则仅输出每个物品在所选范围内的最低价一行")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    api_key = os.getenv("STEAMDT_API_KEY","" ).strip()
    if not api_key:
        print("ERROR: Set STEAMDT_API_KEY in env."); raise SystemExit(2)

    df = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    if args.limit>0: df = df.head(args.limit)

    total = len(df)
    print(f"开始获取 {total} 个物品的价格...")
    
    def _pick_min(rows):
        """从多个价格记录中选择最低价"""
        cand = [r for r in rows if r.get("price")]
        if not cand: return None
        # 价格按数值比较
        cand.sort(key=lambda r: float(r.get("price", 1e99)))
        return cand[0]
    
    out_rows = []
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="获取价格", unit="item"), 1):
        cands = build_candidates(row)
        hit_rows = []  # 收集所有外观的价格
        
        # 遍历所有外观候选，不要提前 break
        for mh in cands:
            try:
                if args.mode == "current":
                    res = price_from_single(api_key, mh, args.platform)
                else:
                    res = price_avg7d(api_key, mh, args.platform)
                
                # 统一为 list
                if isinstance(res, dict):
                    res = [res]
                elif isinstance(res, list):
                    pass
                else:
                    res = []
                
                # 检查是否有有效价格
                if args.platform.lower().strip() == "all":
                    # all 模式：只要有任何价格就算命中
                    has_price = any(r.get("price") for r in res)
                else:
                    # 指定平台模式：只有当"目标平台价格"非空，才算命中
                    has_price = any(
                        (r.get("price") is not None) and (
                            str(r.get("platform","")).lower()==args.platform.lower().strip()
                            or args.platform.lower().strip() in str(r.get("platform","")).lower()
                        )
                        for r in res
                    )
                
                # 将有价格的结果添加到 hit_rows，继续查询其他外观
                if has_price:
                    hit_rows.extend(res)
                    
            except Exception as e:
                # 记录错误但继续查询其他外观
                pass
        
        # "只保留全网底价"开关 - 按外观分组，每个外观选最便宜平台
        if args.min_only and hit_rows:
            # 按 marketHashName 分组
            from collections import defaultdict
            wear_groups = defaultdict(list)
            for hr in hit_rows:
                mh = hr.get("marketHashName", "")
                wear_groups[mh].append(hr)
            
            # 每个外观选最便宜的平台
            hit_rows = []
            for mh, group in wear_groups.items():
                picked = _pick_min(group)
                if picked:
                    hit_rows.append(picked)
        
        # 如果没有命中任何价格，添加一个空记录
        if not hit_rows:
            hit_rows = [{"marketHashName": "", "price": None, "platform": args.platform, 
                        "source": args.mode, "status": ""}]
        
        # 把（可能多条）命中记录写进输出（只保留6个关键字段）
        for hr in hit_rows:
            out_rows.append({
                "marketHashName_used": hr.get("marketHashName",""),
                "rarity_en": row.get("rarity_en",""),
                "case_name_en": row.get("case_name_en",""),
                "price": hr.get("price"),
                "platform": hr.get("platform", args.platform),
                "status": hr.get("status", -1),  # 200=成功，-1=失败
            })
        
        time.sleep(0.12)
    
    # 组装 DataFrame
    df = pd.DataFrame(out_rows)
    
    # 只保留 6 列（容错：若某列不存在就先补空列）
    for col in ["marketHashName_used","rarity_en","case_name_en","price","platform","status"]:
        if col not in df.columns:
            df[col] = None
    
    df = df[["marketHashName_used","rarity_en","case_name_en","price","platform","status"]]
    
    # 写出
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成! 已保存 {len(out_rows)} 条记录到 {args.out}")
    
    # 统计各平台数量和价格优势
    if len(df) > 0 and "platform" in df.columns:
        print("\n" + "="*80)
        print("📊 价格数据统计")
        print("="*80)
        
        # 过滤掉无价格的记录和0价格
        df_valid = df[df["price"].notna() & (df["price"] != "")]
        try:
            df_valid = df_valid[pd.to_numeric(df_valid["price"], errors='coerce') > 0]
        except:
            pass
        
        if len(df_valid) > 0:
            # 平台分布统计
            platform_counts = df_valid["platform"].value_counts()
            total_valid = len(df_valid)
            
            print(f"\n总计获取 {total_valid} 条有效价格记录（{len(df)} 条总记录）：")
            print("-" * 80)
            
            # 按数量排序展示
            for platform, count in platform_counts.items():
                percentage = (count / total_valid) * 100
                bar_length = int(percentage / 2)  # 每2%一个字符
                bar = "█" * bar_length
                print(f"  {platform:10s} : {count:4d} 条  ({percentage:5.1f}%)  {bar}")
            
            # 价格优势分析（不管是否是 min-only 模式）
            print("\n" + "-" * 80)
            print("💰 价格对比分析（基准：STEAM）：")
            print("-" * 80)
            
            # 按商品分组，找出每个商品的 Steam 价格
            steam_prices = {}
            grouped = df_valid.groupby("marketHashName_used")
            
            for mh, group in grouped:
                mh = str(mh)
                steam_rows = group[group["platform"].str.upper() == "STEAM"]
                if len(steam_rows) > 0:
                    try:
                        steam_prices[mh] = float(steam_rows.iloc[0]["price"])
                    except:
                        pass
            
            # 计算每个平台相对于Steam的平均节省
            platform_stats = {}
            for platform in platform_counts.index:
                if platform.upper() != "STEAM":
                    platform_df = df_valid[df_valid["platform"] == platform]
                    savings_list = []
                    total_saved = 0
                    
                    for _, row in platform_df.iterrows():
                        mh = str(row.get("marketHashName_used", ""))
                        if mh in steam_prices:
                            try:
                                other_price = float(row["price"])
                                steam_price = steam_prices[mh]
                                if steam_price > 0 and other_price > 0:
                                    saving_amount = steam_price - other_price
                                    saving_pct = (saving_amount / steam_price) * 100
                                    savings_list.append(saving_pct)
                                    total_saved += saving_amount
                            except:
                                pass
                    
                    if savings_list:
                        avg_saving = sum(savings_list) / len(savings_list)
                        platform_stats[platform] = {
                            "count": len(savings_list),
                            "avg_saving": avg_saving,
                            "total_saved": total_saved
                        }
            
            # 按平均节省百分比排序
            sorted_platforms = sorted(platform_stats.items(), 
                                    key=lambda x: x[1]["avg_saving"], 
                                    reverse=True)
            
            if sorted_platforms:
                for platform, stats in sorted_platforms:
                    avg_save = stats["avg_saving"]
                    total_save = stats["total_saved"]
                    sample_count = stats["count"]
                    
                    if avg_save > 0:
                        emoji = "✅"
                        sign = ""
                    else:
                        emoji = "⚠️"
                        sign = ""
                    
                    print(f"  {emoji} {platform:10s} : 平均 {sign}{avg_save:+6.1f}%  "
                          f"(累计省 ¥{total_save:.2f}, {sample_count}个商品)")
            else:
                print("  ⚠️  无法对比（缺少 STEAM 价格参考）")
        else:
            print("\n⚠️  未获取到有效价格数据")
        
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
