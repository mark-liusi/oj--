#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-fetch CS2 skin prices using SteamDT OpenAPI.

Usage:
  export STEAMDT_API_KEY=xxxxx
  python fetch_prices_with_steamdt.py --items-csv cs2_case_items_full.csv --out prices_today.csv --platform steam --mode current
  python fetch_prices_with_steamdt.py --items-csv cs2_case_items_full.csv --out prices_avg7d.csv --platform steam --mode avg7d

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
    否则：返回单个平台的一条记录（与原行为兼容）
    """
    url = f"{BASE}/open/cs2/v1/price/single"
    params = {"marketHashName": market_hash}
    r = requests.get(url, params=params, headers=HEADERS(api_key), timeout=25)
    if r.status_code != 200:
        base = {"marketHashName": market_hash, "source": "single", "status": r.status_code}
        return [dict(base, price=None, platform=platform)] if platform.lower() == "all" else dict(base, price=None, platform=platform)
    
    j = r.json()
    data = j.get("data", [])

    # 平台名统一小写比较但原样输出
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
        platform_lower = platform.lower().strip()
        price, plat = None, platform
        for d in data:
            if str(d.get("platform","")).lower() == platform_lower:
                price = d.get("sellPrice")
                plat = d.get("platform","")
                break
        if price is None and data:
            price = data[0].get("sellPrice")
            plat = data[0].get("platform","")
        return {"marketHashName": market_hash, "price": price, "platform": plat, "source": "single", "status": 200}

def price_avg7d(api_key: str, market_hash: str, platform: str):
    """
    若 platform == 'all'：返回该 market_hash 下所有平台的7日均价列表
    否则：返回单个平台的一条记录（与原行为兼容）
    """
    url = f"{BASE}/open/cs2/v1/price/avg"
    params = {"marketHashName": market_hash}
    r = requests.get(url, params=params, headers=HEADERS(api_key), timeout=25)
    if r.status_code != 200:
        base = {"marketHashName": market_hash, "source": "avg7d", "status": r.status_code}
        return [dict(base, price=None, platform=platform)] if platform.lower() == "all" else dict(base, price=None, platform=platform)
    
    j = r.json()
    data = j.get("data", {})
    
    if platform.lower().strip() == "all":
        rows = []
        data_list = data.get("dataList", [])
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
        price = data.get("avgPrice")
        plat = platform
        if data.get("dataList"):
            for d in data["dataList"]:
                if str(d.get("platform","")).lower() == platform.lower():
                    price = d.get("avgPrice", price)
                    plat = d.get("platform", platform)
                    break
        return {"marketHashName": market_hash, "price": price, "platform": plat, "source": "avg7d", "status": 200}

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
        hit_rows = []  # 可能是多平台
        
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
                
                # 找到有价格的就停止
                has_price = any(r.get("price") for r in res)
                if has_price:
                    hit_rows = res
                    break
            except Exception as e:
                hit_rows = [{"marketHashName": mh, "price": None, "platform": args.platform, 
                            "source": args.mode, "status": -1, "error": str(e)}]
        
        # "只保留全网底价"开关
        if args.min_only and hit_rows:
            picked = _pick_min(hit_rows)
            hit_rows = [picked] if picked else []
        
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

if __name__ == "__main__":
    main()
