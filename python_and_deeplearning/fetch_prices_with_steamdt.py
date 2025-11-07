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
import os, sys, time, argparse
from typing import List, Dict, Any
import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://open.steamdt.com"
def HEADERS(api_key): 
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

WEARS = ["Factory New","Minimal Wear","Field-Tested","Well-Worn","Battle-Scarred"]

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
    mh = row.get("steamdt_market_hash_name")
    if pd.notna(mh) and str(mh).strip():
        return [str(mh).strip()]
    weapon, finish = row.get("weapon",""), row.get("finish","")
    base = norm_base_name(weapon, finish)
    if not base:
        item = row.get("item_name_en")
        if pd.notna(item):
            item = str(item).strip()
            if " | " in item:
                weapon, finish = item.split(" | ", 1)
                base = f"{weapon} | {finish}"
    if not base:
        return []
    return [f"{base} ({w})" for w in WEARS] + [base]

def price_from_single(api_key: str, market_hash: str, platform: str) -> Dict[str, Any]:
    url = f"{BASE}/open/cs2/v1/price/single"
    params = {"marketHashName": market_hash}
    r = requests.get(url, params=params, headers=HEADERS(api_key), timeout=25)
    if r.status_code != 200:
        return {"marketHashName": market_hash, "price": None, "platform": platform, "source": "single", "status": r.status_code}
    j = r.json()
    data = j.get("data", [])
    price = None
    platform_lower = platform.lower().strip()
    for d in data:
        if str(d.get("platform","")).lower() == platform_lower:
            price = d.get("sellPrice")
            break
    if price is None and data:
        price = data[0].get("sellPrice")
        platform_lower = data[0].get("platform","")
    return {"marketHashName": market_hash, "price": price, "platform": platform_lower, "source": "single", "status": 200}

def price_avg7d(api_key: str, market_hash: str, platform: str) -> Dict[str, Any]:
    url = f"{BASE}/open/cs2/v1/price/avg"
    params = {"marketHashName": market_hash}
    r = requests.get(url, params=params, headers=HEADERS(api_key), timeout=25)
    if r.status_code != 200:
        return {"marketHashName": market_hash, "price": None, "platform": platform, "source": "avg7d", "status": r.status_code}
    j = r.json()
    data = j.get("data", {})
    price = data.get("avgPrice")
    if data.get("dataList"):
        for d in data["dataList"]:
            if str(d.get("platform","")) .lower()== platform.lower():
                price = d.get("avgPrice", price)
                break
    return {"marketHashName": market_hash, "price": price, "platform": platform, "source": "avg7d", "status": 200}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", default="steam")
    ap.add_argument("--mode", choices=["current","avg7d"], default="current")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    api_key = os.getenv("STEAMDT_API_KEY","" ).strip()
    if not api_key:
        print("ERROR: Set STEAMDT_API_KEY in env."); raise SystemExit(2)

    df = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    if args.limit>0: df = df.head(args.limit)

    total = len(df)
    print(f"开始获取 {total} 个物品的价格...")
    
    out_rows = []
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="获取价格", unit="item"), 1):
        cands = build_candidates(row)
        hit = None
        for mh in cands:
            try:
                res = price_from_single(api_key, mh, args.platform) if args.mode=="current" else price_avg7d(api_key, mh, args.platform)
                if res.get("price"): hit = res; break
            except Exception as e:
                hit = {"marketHashName": mh, "price": None, "platform": args.platform, "source": args.mode, "status": -1, "error": str(e)}
        out_rows.append({
            "item_name_en": row.get("item_name_en",""),
            "weapon": row.get("weapon",""),
            "finish": row.get("finish",""),
            "rarity_en": row.get("rarity_en",""),
            "case_name_en": row.get("case_name_en",""),
            "marketHashName_used": (hit or {}).get("marketHashName",""),
            "platform": (hit or {}).get("platform", args.platform),
            "price": (hit or {}).get("price"),
            "source": (hit or {}).get("source", args.mode),
            "status": (hit or {}).get("status", ""),
        })
        time.sleep(0.12)
    pd.DataFrame(out_rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成! 已保存 {len(out_rows)} 条记录到 {args.out}")

if __name__ == "__main__":
    main()
