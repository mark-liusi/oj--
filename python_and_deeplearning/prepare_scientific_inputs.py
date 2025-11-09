# -*- coding: utf-8 -*-
"""
prepare_scientific_inputs.py
============================
把“科学版 EV”所需的两张辅助表自动准备出来：
1) 皮肤浮漂区间表（name,float_min,float_max）——优先从 Inspect API 读（响应里自带 min/max）。
2) 外观价格表（name,exterior,price）——若主CSV已有 exterior 列，则按 (name,exterior) 分组取均价。

另外：若主CSV含 inspect_link（或类似列），本脚本会批量请求 Inspect API，把实例浮漂写回到主CSV的 float 列。

兼容的 Inspect API：CSFloat/CSGOFloat 自建/公有实例（GET /?url=... 返回 iteminfo.floatvalue/min/max）。
默认基址：--inspect-base https://api.csfloat.com  （可改为你自建实例或其它兼容地址）

用法示例：
python prepare_scientific_inputs.py \
  --input items.csv \
  --inspect-base https://api.csfloat.com \
  --meta-out skins_meta.csv \
  --prices-out skin_prices.csv \
  --items-out items_with_float.csv

备注：不会删除/覆盖原列；仅在必要时新增 float 列与写出新文件。
"""

import argparse, sys, time, math, json
from typing import Optional, Dict
from urllib.parse import quote_plus

import pandas as pd
import numpy as np
import requests

EXTERIOR_ALIASES = {
    "崭新出厂":"FN","略有磨损":"MW","久经沙场":"FT","破损不堪":"WW","战痕累累":"BS",
    "factory new":"FN","minimal wear":"MW","field-tested":"FT","well-worn":"WW","battle-scarred":"BS",
    "fn":"FN","mw":"MW","ft":"FT","ww":"WW","bs":"BS"
}

def normalize_exterior(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return None
    s = str(x).strip()
    s_low = s.lower()
    if s in EXTERIOR_ALIASES: return EXTERIOR_ALIASES[s]
    if s_low in EXTERIOR_ALIASES: return EXTERIOR_ALIASES[s_low]
    abbr = s.upper()
    return abbr if abbr in {"FN","MW","FT","WW","BS"} else None

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    cand = {
        "name": ["name","item","物品","物品名称","皮肤","枪皮","名称"],
        "series": ["series","collection","case","箱子","系列","套"],
        "tier": ["tier","rarity","grade","quality","稀有度","品质","等级"],
        "price": ["price","当前价格","价格","steam_price","现价"],
        "float": ["float","wear","float_value","磨损","浮漂"],
        "exterior": ["exterior","外观"],
        "inspect": ["inspect","inspect_link","inspectUrl","inspectURL","inspect link","csgo_econ_action_preview"]
    }
    out = {}
    cols_low = {c.lower():c for c in df.columns}
    for key, names in cand.items():
        for nm in names:
            if nm in df.columns: out[key]=nm; break
            nm_low = nm.lower()
            if nm_low in cols_low: out[key]=cols_low[nm_low]; break
    # 必需列检查
    req = {"name","price"}
    miss = [k for k in req if k not in out]
    if miss:
        raise ValueError(f"缺少必要列：{miss}; 你的列: {list(df.columns)}")
    return out

def call_inspect(inspect_base: str, link: str, timeout: float=15.0) -> Optional[dict]:
    """
    兼容 csfloat/inspect 的 GET /?url=... 返回 JSON。
    典型字段：iteminfo.floatvalue / iteminfo.paintwear, iteminfo.min, iteminfo.max
    """
    url = inspect_base.rstrip("/") + "/?url=" + quote_plus(link)
    try:
        r = requests.get(url, timeout=timeout, headers={"Accept":"application/json"})
        if r.status_code != 200: return None
        return r.json()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input","-i", required=True, help="主CSV：至少包含 name, price；若含 inspect_link 列则会拉取实例浮漂")
    ap.add_argument("--inspect-base", default="https://api.csfloat.com", help="Inspect API 基址（默认 https://api.csfloat.com ）")
    ap.add_argument("--sleep", type=float, default=1.0, help="每次 Inspect 调用之间的间隔秒数（避免速率限制。默认 1s）")
    ap.add_argument("--meta-out", default="skins_meta.csv", help="输出：皮肤 min/max 表")
    ap.add_argument("--prices-out", default="skin_prices.csv", help="输出：外观均价表")
    ap.add_argument("--items-out", default=None, help="输出：写回 float 后的主CSV（可选）")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cols = detect_columns(df)

    # 如果有外观列，先标准化，便于后面聚合价格表
    if "exterior" in cols:
        df["__exterior_std__"] = df[cols["exterior"]].map(normalize_exterior)
    else:
        df["__exterior_std__"] = None

    # 1) 若有 inspect_link，则批量请求，把 float 写回；同时收集 min/max
    meta_records = {}  # name -> (min,max)
    if "inspect" in cols:
        new_float = []
        for i, row in df.iterrows():
            link = row[cols["inspect"]]
            if not isinstance(link, str) or "csgo_econ_action_preview" not in link:
                new_float.append(np.nan); continue
            j = call_inspect(args.inspect_base, link)
            if not j or "iteminfo" not in j:
                new_float.append(np.nan); continue
            info = j["iteminfo"]
            # 实例浮漂：floatvalue 或 paintwear
            fv = info.get("floatvalue", info.get("paintwear"))
            try:
                fv = float(fv) if fv is not None else np.nan
            except Exception:
                fv = np.nan
            new_float.append(fv)
            # 皮肤 min/max：有些实现带有 "min","max"
            fmin = info.get("min"); fmax = info.get("max")
            if fmin is not None and fmax is not None:
                try:
                    fmin = float(fmin); fmax = float(fmax)
                    nm = str(row[cols["name"]])
                    if nm not in meta_records: meta_records[nm] = (fmin,fmax)
                except Exception:
                    pass
            time.sleep(max(0.0, args.sleep))
        # 写回 float 列（若原本没有则新增）
        if "float" in cols:
            df[cols["float"]] = new_float
        else:
            df["float"] = new_float

    # 2) 构建/补全皮肤 min/max 表（只依赖 inspect 结果；若没有 inspect 列则可能为空）
    if meta_records:
        meta_df = pd.DataFrame([{"name":k,"float_min":v[0],"float_max":v[1]} for k,v in meta_records.items()])
    else:
        meta_df = pd.DataFrame(columns=["name","float_min","float_max"])
    meta_df.to_csv(args.meta_out, index=False, encoding="utf-8-sig")
    print(f"[OK] 写出皮肤区间表：{args.meta_out}（{len(meta_df)} 行）")

    # 3) 外观价格表（若主CSV有外观信息才会生成；否则导出空表，科学版会自动回退到主表均价）
    price_rows = []
    if df["__exterior_std__"].notna().any():
        grp = df.groupby([cols["name"],"__exterior_std__"], dropna=True)[cols["price"]].mean().reset_index()
        grp.columns = ["name","exterior","price"]
        price_rows = grp.to_dict("records")
    prices_df = pd.DataFrame(price_rows, columns=["name","exterior","price"])
    prices_df.to_csv(args.prices_out, index=False, encoding="utf-8-sig")
    print(f"[OK] 写出外观价格表：{args.prices_out}（{len(prices_df)} 行）")

    # 4) 可选：写回 items 带 float 的文件
    if args.items_out:
        df.drop(columns=["__exterior_std__"], inplace=True)
        df.to_csv(args.items_out, index=False, encoding="utf-8-sig")
        print(f"[OK] 写出写回后的主CSV：{args.items_out}")
    else:
        df.drop(columns=["__exterior_std__"], inplace=True)

if __name__ == "__main__":
    main()
