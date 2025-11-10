# -*- coding: utf-8 -*-
"""
CS2 炼金期望（科学版，考虑磨损传导/浮漂计算）
================================================
保持你原有“变量名与思路”，只替换“期望产出价 avg_out_next”的求法：
- 由“同系列下一档的均价”，改为“同系列下一档候选皮肤的【按磨损外观落档】的期望价 EV_out”。
- 其它变量与流程保持一致：value_per_item、buy_cost、margin、profit_ratio（ROI）等。

核心规则（实现与注释全在下面函数里）：
1) 稀有度跨档：Consumer→Industrial→Mil-Spec→Restricted→Classified→Covert→Gold。
   - 普通跨档 K = 10；红→金（Covert→Gold）K = 5。
2) 候选皮肤：与材料“同系列 series”的“下一档 next_tier”所有皮肤，等概率。
3) 浮漂传导：
   - 把每个输入材料的浮漂 f_in 先按其“输入皮肤的 (f_min_in, f_max_in)”做归一化，得到 ~f ∈ [0,1]。
   - 再把 ~f 映射到“目标皮肤的 (f_min_out, f_max_out)”得到 f_out；
   - 用统一外观阈值（FN/MW/FT/WW/BS）判定外观 → 取对应外观价格。
4) 单件线性期望：
   - 单件价值 value_per_item = EV_out * (1 - sell_fee) / K
   - buy_cost = price * (1 + buy_fee)
   - margin = value_per_item - buy_cost；profit_ratio = margin / buy_cost

输入与数据：
- 必需：主CSV，包含列：名称(name) / 系列(series) / 稀有度(tier) / 价格(price)。列名可中英混合，自动识别。
- 可选：
  a) 浮漂元数据 CSV（--meta），列：name,item, float_min, float_max（目标/输入皮肤通用）
  b) 外观价格 CSV（--prices），列：name,item, exterior(FN/MW/FT/WW/BS), price
  c) 主CSV若包含 float 或 exterior（外观）列，也会参与计算。
- 若缺元数据：采用稳健的保底策略：输入或输出皮肤的 (f_min, f_max) 缺失 → 退化为 [0,1]；
  输入没有 float → 若有外观，用该外观区间的中点作 f_in；再按 [0,1] 归一化；都没有 → 用 0.50。

用法示例：
python calculate_scientific.py --input items.csv --meta skins_meta.csv --prices skin_prices.csv --sell-fee 0.15 --buy-fee 0.00

输出：
- 明细榜（按 margin 降序），含：name, series, tier, next_tier, K, price, avg_out_next, value_per_item, buy_cost, margin, profit_ratio
- 可选导出 CSV（--out-csv）
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------------
# 常量：稀有度、跨档映射、外观阈值
# ------------------------------
TIER_ALIASES = {
    # 中文
    "消费级":"Consumer", "工业级":"Industrial", "军规级":"Mil-Spec", "受限":"Restricted",
    "保密":"Classified", "隐秘":"Covert", "金":"Gold",
    # 英文/别名
    "consumer":"Consumer", "industrial":"Industrial", "mil-spec":"Mil-Spec", "milspec":"Mil-Spec",
    "restricted":"Restricted", "classified":"Classified", "covert":"Covert", "red":"Covert",
    "gold":"Gold",
}
TIER_ORDER = ["Consumer","Industrial","Mil-Spec","Restricted","Classified","Covert","Gold"]
NEXT_TIER = {
    "Consumer": ("Industrial", 10),
    "Industrial": ("Mil-Spec", 10),
    "Mil-Spec": ("Restricted", 10),
    "Restricted": ("Classified", 10),
    "Classified": ("Covert", 10),
    "Covert": ("Gold", 5),  # 红→金：K=5
    # "Gold": (None, None), # 终点
}

# 外观阈值(全球统一档):FN, MW, FT, WW, BS
EXTERIOR_THRESHOLDS = [
    ("FN", 0.00, 0.07),
    ("MW", 0.07, 0.15),
    ("FT", 0.15, 0.38),
    ("WW", 0.38, 0.45),
    ("BS", 0.45, 1.00),
]
EXTERIOR_ALIASES = {
    "崭新出厂":"FN","略有磨损":"MW","久经沙场":"FT","破损不堪":"WW","战痕累累":"BS",
    "factory new":"FN","minimal wear":"MW","field-tested":"FT","well-worn":"WW","battle-scarred":"BS",
    "fn":"FN","mw":"MW","ft":"FT","ww":"WW","bs":"BS"
}

# 外观英文→中文映射(用于输出)
EXTERIOR_CN = {
    "FN": "崭新",
    "MW": "略磨",
    "FT": "久经",
    "WW": "破损",
    "BS": "战痕"
}

# ------------------------------
# 工具函数
# ------------------------------
def normalize_tier(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    s_low = s.lower()
    if s in TIER_ALIASES: return TIER_ALIASES[s]
    if s_low in TIER_ALIASES: return TIER_ALIASES[s_low]
    # 已经是标准？
    if s in TIER_ORDER: return s
    return s  # 保留原值，后续再过滤

def normalize_exterior(x: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    s_low = s.lower()
    if s in EXTERIOR_ALIASES: return EXTERIOR_ALIASES[s]
    if s_low in EXTERIOR_ALIASES: return EXTERIOR_ALIASES[s_low]
    abbr = s.upper()
    if abbr in {"FN","MW","FT","WW","BS"}: return abbr
    return None

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """自动识别主CSV的列名，返回标准键：name/series/tier/price/float/exterior/stattrak"""
    candidates = {
        "name": ["name","item","物品","物品名称","皮肤","枪皮","名称"],
        "series": ["series","collection","case","箱子","系列","套"],
        "tier": ["tier","rarity","grade","quality","稀有度","品质","等级"],
        "price": ["price","当前价格","价格","steam_price","现价"],
        "float": ["float","wear","float_value","磨损","浮漂"],
        "exterior": ["exterior","外观"],
        "stattrak": ["stattrak","st","是否st","stat","是否StatTrak"]
    }
    out = {}
    cols_low = {c.lower():c for c in df.columns}
    for key, names in candidates.items():
        hit = None
        for nm in names:
            if nm in df.columns:
                hit = nm; break
            if nm.lower() in cols_low:
                hit = cols_low[nm.lower()]; break
        if hit: out[key] = hit
    required = {"name","series","tier","price"}
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"缺少必需列：{missing}；你的列有：{list(df.columns)}")
    return out

def exterior_from_float(f: float) -> str:
    f = max(0.0, min(1.0, float(f)))
    for label, lo, hi in EXTERIOR_THRESHOLDS:
        if lo <= f < hi or (label=="BS" and abs(f-hi)<1e-12):
            return label
    return "BS"

def midpoint(lo: float, hi: float) -> float:
    return (float(lo)+float(hi))/2.0

# ------------------------------
# EV(科学版)：对“同系列+下一档”的候选皮肤，按浮漂传导→外观→价格求均值
# ------------------------------
def compute_ev_out_for_series(
    series: str,
    next_tier: str,
    df_all: pd.DataFrame,
    col_name: str,
    col_series: str,
    col_tier: str,
    # 输入端（当前材料）信息：
    in_name: str,
    f_in: Optional[float],
    exterior_in: Optional[str],
    # 浮漂/外观元数据：
    meta_df: Optional[pd.DataFrame],
    prices_df: Optional[pd.DataFrame],
    # 参数：若缺 min/max，是否回退 [0,1]
    assume_full_range: bool = True,
) -> Optional[float]:
    """
    返回：该 series 在 next_tier 的“期望产出价 EV_out”（不含手续费），若找不到候选皮肤则返回 None。
    说明：
      - 候选皮肤集合 S = {df_all 中 (series,next_tier) 的所有 name，去重}
      - 等概率：每个皮肤 1/|S|
      - 每个皮肤 s 的价格取决于 f_out(s) 落到的外观：
          (1) 先得到输入的归一化 ~f：用输入皮肤 in_name 的 (f_min_in,f_max_in)
              若 meta 缺失：
                • 有 f_in 则直接当作 [0,1] 范围的值（~f = f_in，若 assume_full_range=True）；
                • 没 f_in 但有 exterior_in：取该外观区间中点作为 f_in，再当作 [0,1]；
                • 都没有：~f = 0.50
          (2) 把 ~f 映射到目标皮肤 s 的 (f_min_out,f_max_out)：f_out
          (3) 用全局阈值判定外观 → 在 prices_df 里查该(s,exterior)价格；若缺，则回退到 df_all 中该 s 的“price”列均值；仍缺则跳过。
    """
    # 1) 候选皮肤集合 S
    cand = df_all[(df_all[col_series]==series) & (df_all[col_tier]==next_tier)]
    if cand.empty:
        return None
    # 使用去重名字
    S = cand.groupby(col_name, as_index=False)["price"].mean()  # 保留可回退的均价
    # 2) 输入侧：计算 ~f
    ftilde = None
    f_in_val = None
    if f_in is not None and not (isinstance(f_in, float) and math.isnan(f_in)):
        f_in_val = float(f_in)
    elif exterior_in is not None:
        # 用外观阈值中点作为 f_in
        ex = normalize_exterior(exterior_in)
        for label, lo, hi in EXTERIOR_THRESHOLDS:
            if label == ex:
                f_in_val = midpoint(lo,hi); break
    # 输入皮肤 min/max（如果能拿到）
    fmin_in = None; fmax_in = None
    if meta_df is not None:
        row_in = meta_df[meta_df["name"].str.casefold()==str(in_name).casefold()]
        if not row_in.empty:
            fmin_in = float(row_in.iloc[0]["float_min"])
            fmax_in = float(row_in.iloc[0]["float_max"])
    # 归一化
    if f_in_val is not None and (fmin_in is not None and fmax_in is not None and fmax_in>fmin_in):
        ftilde = (f_in_val - fmin_in) / (fmax_in - fmin_in)
        ftilde = max(0.0, min(1.0, ftilde))
    else:
        # 回退策略
        if f_in_val is not None and assume_full_range:
            ftilde = max(0.0, min(1.0, f_in_val))  # 把 f_in 当作 [0,1] 值
        else:
            ftilde = 0.50  # 完全缺信息
    # 3) 对每个 s 计算 f_out(s) → exterior → price
    prices_sum = 0.0
    cnt = 0
    for _, r in S.iterrows():
        s_name = r[col_name]
        # 目标皮肤的 min/max
        fmin_out = None; fmax_out = None
        if meta_df is not None:
            row_out = meta_df[meta_df["name"].str.casefold()==str(s_name).casefold()]
            if not row_out.empty:
                fmin_out = float(row_out.iloc[0]["float_min"])
                fmax_out = float(row_out.iloc[0]["float_max"])
        if (fmin_out is None or fmax_out is None or fmax_out<=fmin_out) and assume_full_range:
            fmin_out, fmax_out = 0.0, 1.0
        # f_out
        f_out = fmin_out + ftilde * (fmax_out - fmin_out)
        ex_out = exterior_from_float(f_out)
        # 查价格
        price_s = None
        if prices_df is not None:
            mask = (prices_df["name"].str.casefold()==str(s_name).casefold()) & (prices_df["exterior"]==ex_out)
            rowp = prices_df[mask]
            if not rowp.empty:
                price_s = float(rowp.iloc[0]["price"])
        if price_s is None:
            # 回退：用主表中该皮肤的 price 均值
            price_s = float(r["price"]) if "price" in S.columns and not math.isnan(r["price"]) else None
        if price_s is None:
            continue  # 实在没法估价，跳过该候选
        prices_sum += price_s
        cnt += 1
    if cnt==0: return None
    return prices_sum / cnt

# ------------------------------
# 主流程
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="CS2 炼金期望（科学版，考虑磨损传导）")
    ap.add_argument("--input","-i", required=True, help="主CSV：包含 name/series/tier/price；可选 float/exterior/stattrak")
    ap.add_argument("--meta","-m", default=None, help="皮肤浮漂元数据 CSV：name,float_min,float_max")
    ap.add_argument("--prices","-p", default=None, help="外观价格 CSV：name,exterior(FN/MW/FT/WW/BS),price")
    ap.add_argument("--sell-fee", type=float, default=0.15, help="卖出手续费比例（默认 0.15）")
    ap.add_argument("--buy-fee", type=float, default=0.00, help="买入手续费比例（默认 0.00）")
    ap.add_argument("--filter-tier", default=None, nargs="*", help="仅计算这些输入稀有度（可多选），支持中英别名")
    ap.add_argument("--out-csv", default=None, help="导出结果 CSV 路径")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"找不到输入文件：{inp}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(inp)
    cols = detect_columns(df)

    # 规范化
    df["tier"] = df[cols["tier"]].map(normalize_tier)
    df["series"] = df[cols["series"]]
    df["name"] = df[cols["name"]]
    df["price"] = pd.to_numeric(df[cols["price"]], errors="coerce")

    if "float" in cols:
        df["float"] = pd.to_numeric(df[cols["float"]], errors="coerce")
    else:
        df["float"] = np.nan
    if "exterior" in cols:
        df["exterior"] = df[cols["exterior"]].map(normalize_exterior)
    else:
        df["exterior"] = None
    if "stattrak" in cols:
        df["stattrak"] = df[cols["stattrak"]].astype(str).str.lower().isin(["1","true","yes","y","是"])
    else:
        df["stattrak"] = False

    # 过滤无效 tier 与价格
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price"])

    # 元数据与外观价格表
    meta_df = None
    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.exists():
            print(f"警告：未找到 meta：{meta_path}（将采用回退策略）")
        else:
            meta_df = pd.read_csv(meta_path)
            # 规范字段
            meta_cols = {c.lower():c for c in meta_df.columns}
            def colpick(*opts):
                for o in opts:
                    if o in meta_df.columns: return o
                    if o.lower() in meta_cols: return meta_cols[o.lower()]
                return None
            c_name = colpick("name","item","皮肤","名称")
            c_min = colpick("float_min","min_float","fmin","最小浮漂")
            c_max = colpick("float_max","max_float","fmax","最大浮漂")
            if not all([c_name, c_min, c_max]):
                print("警告：meta列不完整，需包含 name,float_min,float_max；已忽略 meta。")
                meta_df = None
            else:
                meta_df = meta_df.rename(columns={c_name:"name", c_min:"float_min", c_max:"float_max"})
                meta_df = meta_df[["name","float_min","float_max"]]

    prices_df = None
    if args.prices:
        p_path = Path(args.prices)
        if not p_path.exists():
            print(f"警告：未找到 prices：{p_path}（必要时将用主表均价回退）")
        else:
            prices_df = pd.read_csv(p_path)
            # 规范字段
            pcols = {c.lower():c for c in prices_df.columns}
            def pcol(*opts):
                for o in opts:
                    if o in prices_df.columns: return o
                    if o.lower() in pcols: return pcols[o.lower()]
                return None
            c_name = pcol("name","item","皮肤","名称")
            c_ex = pcol("exterior","wear","外观")
            c_pr = pcol("price","价格")
            if not all([c_name,c_ex,c_pr]):
                print("警告：prices列不完整，需包含 name,exterior,price；已忽略 prices。")
                prices_df = None
            else:
                prices_df = prices_df.rename(columns={c_name:"name", c_ex:"exterior", c_pr:"price"})
                prices_df["exterior"] = prices_df["exterior"].map(normalize_exterior)
                prices_df = prices_df.dropna(subset=["exterior"])

    # 可选：仅计算某些输入稀有度
    if args.filter_tier:
        allow = {normalize_tier(t) for t in args.filter_tier}
        df = df[df["tier"].isin(allow)].copy()

    # 计算 next_tier 与 K
    df["next_tier"] = df["tier"].map(lambda t: NEXT_TIER[t][0] if t in NEXT_TIER else None)
    df["K"] = df["tier"].map(lambda t: NEXT_TIER[t][1] if t in NEXT_TIER else np.nan)

    # 科学版：对每一行材料，计算其同系列+下一档的 EV_out（未含手续费）
    # 同时找出该材料的最优外观
    ev_list = []
    best_exterior_list = []
    
    for idx, row in df.iterrows():
        t = row["tier"]
        series = row["series"]
        name = row["name"]
        next_t = row["next_tier"]
        K = row["K"]
        
        if not isinstance(next_t, str) or not isinstance(K, (int,float)):
            ev_list.append(np.nan)
            best_exterior_list.append("")
            continue
        
        # 计算期望值
        ev = compute_ev_out_for_series(
            series=series, next_tier=next_t,
            df_all=df, col_name="name", col_series="series", col_tier="tier",
            in_name=name,
            f_in=(row["float"] if not pd.isna(row["float"]) else None),
            exterior_in=(row["exterior"] if isinstance(row["exterior"], str) else None),
            meta_df=meta_df, prices_df=prices_df,
            assume_full_range=True
        )
        ev_list.append(ev if ev is not None else np.nan)
        
        # 找出该物品最便宜的外观(作为推荐购买外观)
        best_ext = ""
        if prices_df is not None:
            item_prices = prices_df[prices_df["name"].str.casefold() == str(name).casefold()]
            if not item_prices.empty:
                cheapest = item_prices.loc[item_prices["price"].idxmin()]
                best_ext = cheapest["exterior"]
        
        # 如果没有外观价格数据,使用主表的exterior字段
        if not best_ext and isinstance(row.get("exterior"), str):
            best_ext = row["exterior"]
        
        # 转换为中文外观名称
        best_ext_cn = EXTERIOR_CN.get(best_ext, best_ext)
        
        best_exterior_list.append(best_ext_cn)
    
    df["avg_out_next"] = ev_list  # 保持原变量名：期望产出价（候选均值）
    df["best_exterior"] = best_exterior_list  # 新增：推荐外观

    # 价值与利润（保持原管线公式与变量名）
    sell_fee = float(args.sell_fee)
    buy_fee = float(args.buy_fee)
    df["value_per_item"] = (df["avg_out_next"] * (1.0 - sell_fee)) / df["K"]
    df["buy_cost"] = df["price"] * (1.0 + buy_fee)
    df["margin"] = df["value_per_item"] - df["buy_cost"]
    df["profit_ratio"] = df["margin"] / df["buy_cost"]

    # 排序与预览
    work_sorted = df.sort_values(by=["margin","profit_ratio"], ascending=[False, False])
    cols_show = ["name","series","tier","next_tier","K","price","avg_out_next","value_per_item","buy_cost","margin","profit_ratio"]
    print("\n=== 科学版：材料明细（按期望利润降序）Top 20 预览 ===")
    print(work_sorted[cols_show].head(20).to_string(index=False))

    # 导出 CSV
    if args.out_csv:
        outp = Path(args.out_csv)
        work_sorted.to_csv(outp, index=False)
        print(f"\n已导出 CSV：{outp}")
    
    # ===== 新增：生成格式化 TXT 文件 =====
    from calculate import format_to_txt
    
    # 1. 按利润排序
    winners_by_margin = work_sorted[work_sorted["margin"] > 0].head(100)
    losers_by_margin = work_sorted[work_sorted["margin"] < 0].tail(100).iloc[::-1]
    
    # 2. 按 ROI 排序
    winners_by_roi = work_sorted[work_sorted["profit_ratio"] > 0].sort_values("profit_ratio", ascending=False).head(100)
    losers_by_roi = work_sorted[work_sorted["profit_ratio"] < 0].sort_values("profit_ratio", ascending=True).head(100)
    
    # 3. 生成 TXT 文件
    output_dir = Path(args.out_csv).parent if args.out_csv else Path(".")
    
    txt_files = {
        "科学版_盈利TOP100_按利润.txt": (winners_by_margin, "科学版 Trade-Up 盈利 TOP 100 - 按期望利润排序"),
        "科学版_亏损TOP100_按利润.txt": (losers_by_margin, "科学版 Trade-Up 亏损 TOP 100 - 按期望利润排序"),
        "科学版_盈利TOP100_按ROI.txt": (winners_by_roi, "科学版 Trade-Up 盈利 TOP 100 - 按 ROI 排序"),
        "科学版_亏损TOP100_按ROI.txt": (losers_by_roi, "科学版 Trade-Up 亏损 TOP 100 - 按 ROI 排序"),
    }
    
    for filename, (data, title) in txt_files.items():
        if len(data) > 0:
            txt_path = output_dir / filename
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(format_to_txt(data, title, max_rows=100))
            print(f"已生成格式化文本：{txt_path}")
    
    print("\n✅ 所有文件生成完成！")

if __name__ == "__main__":
    main()
