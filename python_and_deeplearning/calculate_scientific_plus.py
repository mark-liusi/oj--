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
python calculate_scientific.py --input items.csv --meta data/skins_meta_complete.csv --prices data/prices_with_exterior.csv --sell-fee 0.15 --buy-fee 0.00

输出：
- 明细榜（按 margin 降序），含：name, series, tier, next_tier, K, price, avg_out_next, value_per_item, buy_cost, margin, profit_ratio
- 可选导出 CSV（--out-csv）
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np



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
    ("FN", 0.00, 0.069999),
    ("MW", 0.07, 0.149999),
    ("FT", 0.15, 0.379999),
    ("WW", 0.38, 0.449999),
    ("BS", 0.45, 1.00),
]

# 外观上限阈值字典（用于目标外观计算）
EXTERIOR_UPPER_THRESHOLDS = {
    "FN": 0.069999,
    "MW": 0.149999,
    "FT": 0.379999,
    "WW": 0.449999,
    "BS": 1.00,
}

# 更好外观映射
_BETTER = {"MW":"FN", "FT":"MW", "WW":"FT", "BS":"WW"}

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
# OR-Tools 软依赖
# ------------------------------
try:
    from ortools.sat.python import cp_model  # type: ignore
except Exception:
    cp_model = None

# ====== 运行期缓存（加速） ======
PRICE_ANCHOR_MAP: Optional[Dict[Tuple[str,str], float]] = None  # (name_cf, exterior)->min price
PRICE_SOURCE_MAP: Optional[Dict[Tuple[str,str], str]] = None    # (name_cf, exterior)->platform source
FALLBACK_PRICE_MAP: Optional[Dict[str, float]] = None           # name_cf->min price (fallback)
F_RANGE_MAP: Optional[Dict[str, Tuple[float,float]]] = None     # name_cf->(fmin,fmax)
CAND_MAP: Optional[Dict[Tuple[str,str], List[Tuple[str,float,float,float]]]] = None
# (series_cf, tier)->[(name, fallback_price, fmin_out, fmax_out), ...]
CHEAPEST_EX_MAP: Optional[Dict[str,str]] = None                 # name_cf->cheapest exterior

def _cf(s: Any) -> str:
    return str(s).strip().casefold()

def init_runtime_caches(df_ctx: pd.DataFrame,
                        meta_df: Optional[pd.DataFrame],
                        prices_df: Optional[pd.DataFrame]) -> None:
    """为本数据集建立加速用的查找表（ST/Normal 各自独立）"""
    global PRICE_ANCHOR_MAP, PRICE_SOURCE_MAP, FALLBACK_PRICE_MAP, F_RANGE_MAP, CAND_MAP, CHEAPEST_EX_MAP

    # 1) 浮漂区间
    F_RANGE_MAP = {}
    if meta_df is not None and len(meta_df):
        mm = meta_df.copy()
        mm["name_cf"] = mm["name"].astype(str).str.strip().str.casefold()
        for _, r in mm.iterrows():
            F_RANGE_MAP[r["name_cf"]] = (float(r["float_min"]), float(r["float_max"]))

    # 2) 外观锚点价 & 最便宜外观
    PRICE_ANCHOR_MAP = {}
    PRICE_SOURCE_MAP = {}
    if prices_df is not None and len(prices_df):
        pp = prices_df.copy()
        pp["name_cf"] = pp["name"].astype(str).str.strip().str.casefold()
        # 按(name_cf, exterior)分组，取最低价及其来源平台
        for (name_cf, exterior), grp_data in pp.groupby(["name_cf","exterior"]):
            # 找最低价的行
            idx_min = grp_data["price"].idxmin()
            min_price = float(grp_data.loc[idx_min, "price"])
            source_platform = str(grp_data.loc[idx_min, "platform"])
            PRICE_ANCHOR_MAP[(name_cf, exterior)] = min_price
            PRICE_SOURCE_MAP[(name_cf, exterior)] = source_platform
        idx = pp.groupby("name_cf")["price"].idxmin()
        CHEAPEST_EX_MAP = {pp.loc[i,"name_cf"]: pp.loc[i,"exterior"] for i in idx}
    else:
        CHEAPEST_EX_MAP = {}

    # 3) 回退价（优先外观表聚合，否则主表）
    FALLBACK_PRICE_MAP = {}
    if prices_df is not None and len(prices_df):
        p2 = prices_df.copy()
        p2["name_cf"] = p2["name"].astype(str).str.strip().str.casefold()
        gg = p2.groupby("name_cf")["price"].min().reset_index()
        for _, r in gg.iterrows():
            FALLBACK_PRICE_MAP[r["name_cf"]] = float(r["price"])
    else:
        gg = df_ctx.groupby("name")["price"].min().reset_index()
        gg["name_cf"] = gg["name"].astype(str).str.strip().str.casefold()
        for _, r in gg.iterrows():
            FALLBACK_PRICE_MAP[r["name_cf"]] = float(r["price"])

    # 4) 候选集合：(series,tier)->该 tier 的 name 列表（含回退价与 f 区间）
    CAND_MAP = {}
    df_tmp = df_ctx.copy()
    df_tmp["_series_cf"] = df_tmp["series"].astype(str).str.strip().str.casefold()
    for (ser_cf, tier), g in df_tmp.groupby(["_series_cf","tier"]):
        lst = []
        for nm, gg in g.groupby("name"):
            ncf = _cf(nm)
            fmin, fmax = F_RANGE_MAP.get(ncf, (0.0, 1.0))
            fallback = FALLBACK_PRICE_MAP.get(ncf, float(gg["price"].min()))
            lst.append((str(nm), float(fallback), float(fmin), float(fmax)))
        CAND_MAP[(ser_cf, tier)] = lst

# ------------------------------
# 工具函数（优化版：统一处理逻辑，避免重复）
# ------------------------------
def _normalize_with_aliases(x: str, aliases: dict, valid_set: set = None) -> Optional[str]:
    """通用的规范化函数，处理别名映射"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    # 优先匹配原始字符串
    if s in aliases:
        return aliases[s]
    # 再尝试小写匹配
    s_low = s.lower()
    if s_low in aliases:
        return aliases[s_low]
    # 如果有有效集合，检查是否已经是标准值
    if valid_set and s in valid_set:
        return s
    # 外观特殊处理：支持缩写
    if valid_set == {"FN","MW","FT","WW","BS"}:
        abbr = s.upper()
        if abbr in valid_set:
            return abbr
    return s if not valid_set else None

def normalize_tier(x: str) -> str:
    """规范化稀有度，使用统一的别名映射逻辑"""
    result = _normalize_with_aliases(x, TIER_ALIASES, set(TIER_ORDER))
    return result if result else str(x).strip()

def normalize_exterior(x: str) -> Optional[str]:
    """规范化外观，使用统一的别名映射逻辑"""
    return _normalize_with_aliases(x, EXTERIOR_ALIASES, {"FN","MW","FT","WW","BS"})

def detect_columns(df: pd.DataFrame, required: tuple = ("name", "series", "tier", "price")) -> Dict[str, str]:
    """自动识别主CSV的列名，返回标准键：name/series/tier/price/float/exterior/stattrak
    
    Args:
        df: DataFrame
        required: 必需列元组，默认 ("name", "series", "tier", "price")
    """
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
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"缺少必需列：{missing}；你的列有：{list(df.columns)}")
    return out

def detect_price_columns(df: pd.DataFrame) -> Dict[str, str]:
    """专用于价格表的列检测，只要求 name/price/exterior 三列"""
    return detect_columns(df, required=("name", "price", "exterior"))

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

# 辅助函数：减少重复代码和None检查
def _get_float_value(f_in, exterior_in):
    if f_in is not None and not (isinstance(f_in, float) and math.isnan(f_in)):
        return float(f_in)
    if exterior_in is not None:
        ex = normalize_exterior(exterior_in)
        for label, lo, hi in EXTERIOR_THRESHOLDS:
            if label == ex:
                # 当只有外观信息时，使用该外观的最高磨损度下界（上界-ε）
                # 因为区间是 [lo, hi)，所以应该稍小于hi
                # 例如：MW [0.07, 0.15) → 返回 0.15 - 1e-6 = 0.149999
                epsilon = 1e-6
                return float(hi - epsilon)
    return None

def _get_skin_float_range(name, meta_df, assume_full=True):
    if meta_df is not None:
        row = meta_df[meta_df["name"].str.casefold()==str(name).casefold()]
        if not row.empty:
            fmin = float(row.iloc[0]["float_min"])
            fmax = float(row.iloc[0]["float_max"])
            if fmax > fmin:
                return fmin, fmax
    return (0.0, 1.0) if assume_full else (None, None)

def _normalize_float(f_val, fmin, fmax, assume_full=True):
    if f_val is not None and fmin is not None and fmax is not None and fmax > fmin:
        ftilde = (f_val - fmin) / (fmax - fmin)
        return max(0.0, min(1.0, ftilde))
    if f_val is not None and assume_full:
        return max(0.0, min(1.0, f_val))
    return 0.50

def _get_price_for_skin(name, exterior, prices_df, fallback_price):
    if prices_df is not None:
        mask = (prices_df["name"].str.casefold()==str(name).casefold()) & (prices_df["exterior"]==exterior)
        rowp = prices_df[mask]
        if not rowp.empty:
            return float(rowp["price"].median())
    return fallback_price
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
    # 使用去重名字，取最低价作为回退价
    S = cand.groupby(col_name, as_index=False)["price"].min()  # 保留最低价作为回退价
    # 2) 输入侧：计算 ~f
    ftilde = None
    f_in_val = None
    if f_in is not None and not (isinstance(f_in, float) and math.isnan(f_in)):
        f_in_val = float(f_in)
    elif exterior_in is not None:
        # 用外观的最高磨损度（上界-ε）
        ex = normalize_exterior(exterior_in)
        for label, lo, hi in EXTERIOR_THRESHOLDS:
            if label == ex:
                f_in_val = hi - 1e-6; break  # 取上界-ε，确保落在该外观区间
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
        # 查价格：根据外观从prices_df中查询最低价
        price_s = None
        if prices_df is not None:
            mask = (prices_df["name"].str.casefold()==str(s_name).casefold()) & (prices_df["exterior"]==ex_out)
            rowp = prices_df[mask]
            if not rowp.empty:
                # 使用最低价而非中位数
                price_s = float(rowp["price"].min())
        if price_s is None:
            # 回退：用主表中该皮肤的最低价
            price_s = float(r["price"]) if "price" in S.columns and not math.isnan(r["price"]) else None
        if price_s is None:
            continue  # 实在没法估价，跳过该候选
        prices_sum += price_s
        cnt += 1
    if cnt==0: return None
    return prices_sum / cnt

def compute_ev_out_for_series_fast(series: str, next_tier: str,
                                   in_name: str,
                                   f_in: Optional[float],
                                   exterior_in: Optional[str],
                                   assume_full_range: bool = True) -> Optional[float]:
    """快速 EV 计算：使用运行期缓存"""
    # 候选缓存
    if 'CAND_MAP' not in globals() or CAND_MAP is None:
        return None
    S = CAND_MAP.get((str(series).strip().casefold(), next_tier))
    if not S:
        return None

    # 输入 ~f
    f_in_val = None
    if f_in is not None and not (isinstance(f_in, float) and math.isnan(f_in)):
        f_in_val = float(f_in)
    elif isinstance(exterior_in, str):
        exn = normalize_exterior(exterior_in)
        for label, lo, hi in EXTERIOR_THRESHOLDS:
            if label == exn:
                f_in_val = hi - 1e-6; break  # 取上界-ε
    fmin_in, fmax_in = (0.0, 1.0)
    if 'F_RANGE_MAP' in globals() and F_RANGE_MAP is not None:
        fmin_in, fmax_in = F_RANGE_MAP.get(str(in_name).strip().casefold(), (0.0, 1.0))
    if f_in_val is not None and (fmax_in > fmin_in):
        ftilde = (f_in_val - fmin_in) / (fmax_in - fmin_in)
        ftilde = max(0.0, min(1.0, ftilde))
    else:
        ftilde = max(0.0, min(1.0, f_in_val)) if (f_in_val is not None and assume_full_range) else 0.50

    # 价格聚合：根据浮漂传导后的外观，从PRICE_ANCHOR_MAP查询该外观的最低价
    total, cnt = 0.0, 0
    for s_name, fallback_price, fmin_out, fmax_out in S:
        # 步骤1：浮漂传导
        f_out = fmin_out + ftilde * (fmax_out - fmin_out)
        # 步骤2：判断外观
        ex_out = exterior_from_float(f_out)
        
        # 步骤3：查询该物品在该外观下的最低价
        s_name_cf = str(s_name).strip().casefold()
        p = None
        if 'PRICE_ANCHOR_MAP' in globals() and PRICE_ANCHOR_MAP is not None:
            p = PRICE_ANCHOR_MAP.get((s_name_cf, ex_out))
        
        # 步骤4：回退策略（如果该外观没有数据）
        if p is None:
            p = fallback_price
        
        if p is None:
            continue
        total += float(p); cnt += 1
    
    if cnt == 0:
        return None
    return total / cnt

def _get_price_anchor(prices_df, name: str, ex: str):
    # 先查缓存
    if 'PRICE_ANCHOR_MAP' in globals() and PRICE_ANCHOR_MAP is not None:
        v = PRICE_ANCHOR_MAP.get((str(name).strip().casefold(), ex))
        if v is not None:
            return float(v)
    if prices_df is None: 
        return None
    m = prices_df[(prices_df["name"].str.casefold()==str(name).strip().casefold())
                  & (prices_df["exterior"]==ex)]["price"]
    return float(m.median()) if len(m) else None

def _ext_bounds(ex: str):
    for label, lo, hi in EXTERIOR_THRESHOLDS:
        if label == ex: return float(lo), float(hi)
    return None, None

def price_from_float(name: str, f_allow: float, prices_df,
                     gamma=None, kappa=None, eps=0.01) -> Optional[float]:
    """获取物品的最低价格（与主流程一致）。
    
    参数 gamma, kappa, eps 已弃用，保留只为向后兼容。
    现在直接返回该物品的最低价，确保整个系统（包括辅助优化）一致使用最低价。
    """
    # 获取该物品的最低价
    name_cf = str(name).strip().casefold()
    
    # 优先使用缓存的最低价
    if FALLBACK_PRICE_MAP is not None:
        fallback_price = FALLBACK_PRICE_MAP.get(name_cf)
        if fallback_price is not None:
            return float(fallback_price)
    
    # 回退到直接查询prices_df
    if prices_df is not None and len(prices_df) > 0:
        m = prices_df[prices_df["name"].str.casefold() == name_cf]["price"]
        if len(m) > 0:
            return float(m.min())
    
    return None


# =========================================
# 在 avgF 下的“系列下一档期望价”（候选均值）
# 沿用你现有的落档、候选均值逻辑
# =========================================
def EV_out_at_avgF(series: str, next_tier: str, avgF: float,
                   df_all: pd.DataFrame, meta_df: Optional[pd.DataFrame],
                   prices_df: Optional[pd.DataFrame]) -> Optional[float]:
    # 快路径：使用 CAND_MAP/PRICE_ANCHOR_MAP
    if 'CAND_MAP' in globals() and CAND_MAP is not None:
        S = CAND_MAP.get((str(series).strip().casefold(), next_tier))
        if not S:
            return None
        acc = []
        for s_name, fallback_price, fmin, fmax in S:
            fout = min(max(avgF, fmin), fmax - 1e-6)
            ex = exterior_from_float(fout)
            p = None
            if 'PRICE_ANCHOR_MAP' in globals() and PRICE_ANCHOR_MAP is not None:
                p = PRICE_ANCHOR_MAP.get((str(s_name).strip().casefold(), ex))
            if p is None:
                p = fallback_price
            if p is not None:
                acc.append(float(p))
        return (sum(acc)/len(acc)) if acc else None

    # —— 回退实现：使用最低价确保一致性 ——
    S = df_all[(df_all["series"].str.casefold()==str(series).casefold())
               & (df_all["tier"]==next_tier)]
    if S.empty: return None
    acc = []
    # 按 name 聚合，避免同名多行重复计入
    for sname, g in S.groupby("name"):
        price = None
        if prices_df is not None:
            rowp = prices_df[prices_df["name"].str.casefold()==str(sname).casefold()]
            if not rowp.empty:
                price = float(rowp["price"].min())  # 使用最低价而非中位数
        if price is None and "price" in g:
            price = float(g["price"].min())  # 使用最低价而非中位数
        if price is not None:
            acc.append(price)
    return (sum(acc)/len(acc)) if acc else None


# =========================================
# 简化版最佳主料推荐：为每个目标物品找价格最低的下级主料
# =========================================
def find_best_anchor_for_target(
    target_name: str, target_series: str, target_tier: str, target_exterior: str,
    df_all: pd.DataFrame, prices_df: pd.DataFrame, meta_df: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    为指定目标物品找最佳主料（同系列、下一级稀有度、价格最低）。
    
    返回：
    {
        "anchor_name": str,        # 最佳主料名称
        "anchor_price": float,     # 最佳主料价格
        "anchor_float": float,     # 推荐磨损值（尽量低）
        "source_tier": str         # 来源稀有度
    }
    """
    # 1) 找到下级稀有度（source_tier）
    source_tier = None
    for src, nxt in NEXT_TIER.items():
        if target_tier in nxt:
            source_tier = src
            break
    if source_tier is None:
        return None
    
    # 2) 筛选同系列、下级稀有度的物品
    pool = df_all[
        (df_all["series"].str.casefold() == str(target_series).casefold()) &
        (df_all["tier"] == source_tier)
    ].copy()
    
    if pool.empty:
        return None
    
    # 3) 找价格最低的物品
    best = None
    for _, row in pool.iterrows():
        name = str(row["name"])
        price = float(row["price"])
        
        # 获取该物品的最低可达磨损
        fmin = 0.0
        if meta_df is not None:
            rowm = meta_df[meta_df["name"].str.casefold() == name.casefold()]
            if not rowm.empty:
                fmin = float(rowm.iloc[0]["float_min"])
        
        # 获取最低价格
        min_price = price_from_float(name, fmin, prices_df)
        if min_price is None:
            min_price = price
        
        if best is None or min_price < best["anchor_price"]:
            best = {
                "anchor_name": name,
                "anchor_price": float(min_price),
                "anchor_float": float(fmin),
                "source_tier": source_tier
            }
    
    return best


# =========================================
# 为"每个目标物品"生成"最优主料清单"（按箱/系列分组）
# =========================================
def build_best_lower_map_for_targets(
    df_all: pd.DataFrame, prices_df: pd.DataFrame, meta_df: pd.DataFrame,
    target_tier: str, target_exterior: str = "FN"
) -> pd.DataFrame:
    """生成所有目标物品的最佳主料推荐表"""
    rows = []
    targets = df_all[df_all["tier"] == target_tier].copy()
    
    for _, t in targets.iterrows():
        tgt_name = str(t["name"])
        tgt_series = str(t["series"])
        tgt_tier = str(t["tier"])
        
        best = find_best_anchor_for_target(
            target_name=tgt_name,
            target_series=tgt_series,
            target_tier=tgt_tier,
            target_exterior=target_exterior,
            df_all=df_all,
            prices_df=prices_df,
            meta_df=meta_df
        )
        
        if best:
            rows.append({
                "target_name": tgt_name,
                "target_series": tgt_series,
                "target_tier": tgt_tier,
                "target_exterior": target_exterior,
                "best_anchor_name": best["anchor_name"],
                "best_anchor_price": best["anchor_price"],
                "best_anchor_float": best["anchor_float"],
                "source_tier": best["source_tier"]
            })
    
    return pd.DataFrame(rows)

# 主流程
def main():
    ap = argparse.ArgumentParser(description="CS2 炼金期望（科学版，考虑磨损传导）")
    ap.add_argument("--input","-i", required=True, help="主CSV：包含 name/series/tier/price；可选 float/exterior/stattrak")
    ap.add_argument("--meta","-m", default=None, help="皮肤浮漂元数据 CSV：name,float_min,float_max")
    ap.add_argument("--prices","-p", default=None, help="外观价格 CSV：name,exterior(FN/MW/FT/WW/BS),price")
    ap.add_argument("--sell-fee", type=float, default=0.00, help="卖出手续费比例（默认 0.00）")
    ap.add_argument("--buy-fee", type=float, default=0.00, help="买入手续费比例（默认 0.00）")
    ap.add_argument("--filter-tier", default=None, nargs="*", help="仅计算这些输入稀有度（可多选），支持中英别名")
    ap.add_argument("--out-csv", default=None, help="导出结果 CSV 路径")
    ap.add_argument("--split-st", action="store_true",
                    help="分离 StatTrak™：生成 ST/非ST 两份独立榜单，避免混算（ST 溢价通常巨大）")
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
            # 使用专用价格表列检测函数
            try:
                pcol_map = detect_price_columns(prices_df)
                prices_df = prices_df.rename(columns={
                    pcol_map["name"]: "name",
                    pcol_map["exterior"]: "exterior",
                    pcol_map["price"]: "price"
                })
                prices_df["exterior"] = prices_df["exterior"].map(normalize_exterior)
                prices_df = prices_df.dropna(subset=["exterior"])
            except ValueError as e:
                print(f"警告：prices列检测失败（{e}），已忽略 prices。")
                prices_df = None

    # 可选：仅计算某些输入稀有度
    if args.filter_tier:
        allow = {normalize_tier(t) for t in args.filter_tier}
        df = df[df["tier"].isin(allow)].copy()

    # === 新增：StatTrak™ 分离处理 ===
    split_st = args.split_st
    datasets_to_process = []
    
    if split_st:
        # 分离模式：生成 ST 和非 ST 两份数据集
        df_st = df[df["stattrak"] == True].copy()
        df_normal = df[df["stattrak"] == False].copy()
        
        if len(df_st) > 0:
            datasets_to_process.append(("ST", df_st))
            print(f"\n[+] 检测到 {len(df_st)} 个 StatTrak 物品")
        
        if len(df_normal) > 0:
            datasets_to_process.append(("Normal", df_normal))
            print(f"[+] 检测到 {len(df_normal)} 个普通物品")
        
        if len(datasets_to_process) == 0:
            print("错误：启用 --split-st 但没有找到任何数据", file=sys.stderr)
            sys.exit(1)
    else:
        # 混算模式（默认）：将所有数据当作一个整体
        datasets_to_process.append(("All", df))
    
    # 对每个数据集分别计算
    all_results = {}
    
    for label, df_subset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"开始处理：{label} 数据集 ({len(df_subset)} 条记录)")
        print(f"{'='*60}")
        
        df_work = df_subset.copy()

        # 计算 next_tier 与 K
        df_work["next_tier"] = df_work["tier"].map(lambda t: NEXT_TIER[t][0] if t in NEXT_TIER else None)
        df_work["K"] = df_work["tier"].map(lambda t: NEXT_TIER[t][1] if t in NEXT_TIER else np.nan)

        # === 建立运行期缓存，加速后续计算 ===
        init_runtime_caches(df_work, meta_df, prices_df)

        # 科学版：对每一行材料，计算其同系列+下一档的 EV_out（未含手续费）
        # 同时找出该材料的最优外观
        ev_list = []
        best_exterior_list = []
        
        for idx, row in df_work.iterrows():
            t = row["tier"]
            series = row["series"]
            name = row["name"]
            next_t = row["next_tier"]
            K = row["K"]
            
            if not isinstance(next_t, str) or not isinstance(K, (int,float)):
                ev_list.append(np.nan)
                best_exterior_list.append("")
                continue
            
            # 计算期望值 - 使用快速路径
            ev = compute_ev_out_for_series_fast(
                series=series, next_tier=next_t, in_name=name,
                f_in=(row["float"] if not pd.isna(row["float"]) else None),
                exterior_in=(row["exterior"] if isinstance(row["exterior"], str) else None),
                assume_full_range=True
            )
            # 回退到慢路径（如果缓存未初始化）
            if ev is None:
                ev = compute_ev_out_for_series(
                    series=series, next_tier=next_t,
                    df_all=df_work, col_name="name", col_series="series", col_tier="tier",
                    in_name=name,
                    f_in=(row["float"] if not pd.isna(row["float"]) else None),
                    exterior_in=(row["exterior"] if isinstance(row["exterior"], str) else None),
                    meta_df=meta_df, prices_df=prices_df,
                    assume_full_range=True
                )
            ev_list.append(ev if ev is not None else np.nan)
            
            # 找出该物品最便宜的外观(作为推荐购买外观) - 使用缓存
            best_ext = ""
            if 'CHEAPEST_EX_MAP' in globals() and CHEAPEST_EX_MAP is not None:
                best_ext = CHEAPEST_EX_MAP.get(str(name).strip().casefold())
            if not best_ext and prices_df is not None:
                item_prices = prices_df[prices_df["name"].str.casefold() == str(name).casefold()]
                if not item_prices.empty:
                    cheapest = item_prices.loc[item_prices["price"].idxmin()]
                    best_ext = cheapest["exterior"]
            
            # 如果没有外观价格数据,使用主表的exterior字段
            if not best_ext and isinstance(row.get("exterior"), str):
                best_ext = row["exterior"]
            
            # 转换为中文外观名称
            best_ext_cn = EXTERIOR_CN.get(best_ext, best_ext) if best_ext else ""
            
            best_exterior_list.append(best_ext_cn)
        
        df_work["avg_out_next"] = ev_list  # 保持原变量名：期望产出价（候选均值）
        df_work["best_exterior"] = best_exterior_list  # 新增：推荐外观

        # 价值与利润（保持原管线公式与变量名）
        sell_fee = float(args.sell_fee)
        buy_fee = float(args.buy_fee)
        df_work["value_per_item"] = (df_work["avg_out_next"] * (1.0 - sell_fee)) / df_work["K"]
        df_work["buy_cost"] = df_work["price"] * (1.0 + buy_fee)
        df_work["margin"] = df_work["value_per_item"] - df_work["buy_cost"]
        df_work["profit_ratio"] = np.where(df_work["buy_cost"] > 0,
                                           df_work["margin"] / df_work["buy_cost"],
                                           np.nan)

        # 排序与预览
        work_sorted = df_work.sort_values(by=["margin","profit_ratio"], ascending=[False, False])
        cols_show = ["name","series","tier","next_tier","K","price","avg_out_next","value_per_item","buy_cost","margin","profit_ratio"]
        print(f"\n=== {label} 科学版：材料明细（按期望利润降序）Top 20 预览 ===")
        print(work_sorted[cols_show].head(20).to_string(index=False))

        # 导出 CSV
        if args.out_csv:
            outp = Path(args.out_csv)
            # 如果是分离模式，为文件名添加后缀
            if split_st and label != "All":
                outp = outp.with_stem(f"{outp.stem}_{label}")
            work_sorted.to_csv(outp, index=False)
            print(f"\n已导出 CSV：{outp}")
        
        # ===== 生成固定的5份核心报告 =====
        output_dir = Path(args.out_csv).parent if args.out_csv else Path(".")
        suffix = f"_{label}" if split_st and label != "All" else ""
        
        # 调用整合后的固定报告生成函数（生成5个文件）
        emit_fixed_reports_for_dataset(
            df_work=work_sorted, meta_df=meta_df, prices_df=prices_df,
            out_dir=output_dir, label_suffix=suffix
        )
        
        # 保存本次结果
        all_results[label] = work_sorted
    
    print("\n所有文件生成完成！")


# 反向映射：目标稀有度 -> 下级稀有度
PREV_TIER = {nxt: src for src, (nxt, _) in NEXT_TIER.items()}  # 例如 Classified<-Restricted, Covert<-Classified 等

def _get_fminmax(meta_df: Optional[pd.DataFrame], name: str) -> Tuple[float, float]:
    """读某皮肤可达浮漂区间；缺失时回退 [0,1]。"""
    # 先查缓存
    if 'F_RANGE_MAP' in globals() and F_RANGE_MAP is not None:
        key = str(name).strip().casefold()
        v = F_RANGE_MAP.get(key)
        if v is not None: 
            return float(v[0]), float(v[1])
    if meta_df is None:
        return 0.0, 1.0
    row = meta_df[meta_df["name"].str.casefold() == str(name).strip().casefold()]
    if row.empty:
        return 0.0, 1.0
    fmin = float(row.iloc[0]["float_min"]); fmax = float(row.iloc[0]["float_max"])
    if not (fmax > fmin):
        return 0.0, 1.0
    return fmin, fmax

def _exteriors_for_item(name: str, meta_df: Optional[pd.DataFrame]) -> List[str]:
    """该目标皮“可达”的外观集合（与自身 f 区间相交的外观才输出）。"""
    fmin, fmax = _get_fminmax(meta_df, name)
    ex_list = []
    for lab, lo, hi in EXTERIOR_THRESHOLDS:
        if (fmax > lo) and (fmin < hi):   # 区间有交
            ex_list.append(lab)
    return ex_list

def _theta_for_item_and_ex(name: str, target_exterior: str, meta_df: Optional[pd.DataFrame]) -> float:
    """
    针对“某个目标皮 + 外观”的归一化阈值 θ_item：
    令 f_out = fmin_out + ~f * (fmax_out - fmin_out) ≤ Thr(ex) → ~f ≤ θ_item
    """
    fmin_out, fmax_out = _get_fminmax(meta_df, name)
    thr = EXTERIOR_UPPER_THRESHOLDS.get(target_exterior, 0.07)
    theta = (thr - fmin_out) / max(1e-9, (fmax_out - fmin_out))
    return float(np.clip(theta, 0.0, 1.0))

def _estimate_price_at_allow(name: str, f_allow: float, prices_df: Optional[pd.DataFrame]) -> Optional[float]:
    """
    估算在 f_allow 处的最低可成交价：优先价格–磨损曲线；失败则“本地均价×智能溢价”兜底。
    """
    p = price_from_float(name, float(f_allow), prices_df)  # 可单调、贴近更好外观的曲线估价
    if p is not None:
        return float(p)
    return None


def emit_fixed_reports_for_dataset(
    df_work: pd.DataFrame,        # 已规范化 & 含 value_per_item/margin/profit_ratio 的数据表
    meta_df: Optional[pd.DataFrame],
    prices_df: Optional[pd.DataFrame],
    out_dir: Path,
    label_suffix: str = ""
):
    """
    生成5份固定核心报告：
      1) 每箱/每物品/各外观 的最优下级（最高磨损→最低价）
      2) 炼金期望 正收益 TOP100 - 按利润排序（含 ROI）
      3) 炼金期望 负收益 TOP100 - 按利润排序（含 ROI）
      4) 炼金期望 正收益 TOP100 - 按 ROI 排序（含绝对利润）
      5) 炼金期望 负收益 TOP100 - 按 ROI 排序（含绝对利润）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- (1) 每箱/每物品/各外观 的最优下级 ----------
    lines = []
    title = "每箱/每物品/各外观 的最优下级（最高磨损→最低价）"
    lines.append("=" * 120)
    lines.append(title)
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"{'系列(箱子)':<28}{'目标物品':<40}{'目标稀有度':<10}{'外观':<6}"
                 f"{'最优下级(材料名)':<40}{'f_allow(上限)':>14}{'估算最低价':>14}")
    lines.append("-" * 120)

    # 目标集合：tier ∈ PREV_TIER.keys()（有下级可炼）
    targets = df_work[df_work["tier"].isin(PREV_TIER.keys())][["name","series","tier"]].drop_duplicates()
    targets = targets.sort_values(by=["series","tier","name"])

    # pool 缓存优化
    pool_cache: Dict[Tuple[str,str], pd.DataFrame] = {}

    for _, t in targets.iterrows():
        tgt_name = str(t["name"]); series = str(t["series"]); tgt_tier = str(t["tier"])
        src_tier = PREV_TIER.get(tgt_tier)
        if not src_tier:
            continue
        # 同系列下级材料池 - 使用缓存
        key = (series, src_tier)
        if key not in pool_cache:
            pool_cache[key] = df_work[(df_work["series"]==series) & (df_work["tier"]==src_tier)]
        pool = pool_cache[key]
        if pool.empty:
            # 没有下级可选
            ex_list = _exteriors_for_item(tgt_name, meta_df)
            for ex in ex_list:
                lines.append(f"{series:<28}{tgt_name:<40}{tgt_tier:<10}{ex:<6}"
                             f"{'无下级可选':<40}{'':>14}{'':>14}")
            continue

        # 逐外观给出“最高磨损→最低价”的最优下级
        for ex in _exteriors_for_item(tgt_name, meta_df):
            theta = _theta_for_item_and_ex(tgt_name, ex, meta_df)
            best = None  # (price, name, f_allow)
            for _, r in pool.iterrows():
                in_name = str(r["name"])
                fmin_in, fmax_in = _get_fminmax(meta_df, in_name)
                f_allow = np.clip(fmin_in + theta * (fmax_in - fmin_in), fmin_in, fmax_in)
                est = _estimate_price_at_allow(in_name, f_allow, prices_df)
                if est is None:
                    continue
                cand = (float(est), in_name, float(f_allow))
                if (best is None) or (cand[0] < best[0]):
                    best = cand
            if best is None:
                lines.append(f"{series:<28}{tgt_name:<40}{tgt_tier:<10}{ex:<6}"
                             f"{'无法估价':<40}{'':>14}{'':>14}")
            else:
                est_price, best_name, f_allow = best
                lines.append(f"{series:<28}{tgt_name:<40}{tgt_tier:<10}{ex:<6}"
                             f"{best_name:<40}{f_allow:>14.5f}{est_price:>14.2f}")

    path_best_lower = out_dir / f"固定_每箱每物品_各外观_最优下级{label_suffix}.txt"
    with open(path_best_lower, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---------- (2) 正收益 TOP100 - 按利润排序（含 ROI） ----------
    winners_margin = df_work[df_work["margin"] > 0].sort_values(["margin","profit_ratio"], ascending=[False, False]).head(100)
    lines2 = []
    lines2.append("=" * 150)
    lines2.append("炼金期望 正收益 TOP100 - 按利润排序（含 ROI）")
    lines2.append("=" * 150)
    lines2.append("")
    lines2.append(f"{'排名':<6}{'物品名称':<40}{'系列':<28}{'稀有度':<10}{'目标':<10}{'外观':<6}"
                  f"{'单价':>10}{'期望产出':>12}{'单件价值':>12}{'利润':>12}{'ROI':>10}")
    lines2.append("-" * 150)
    for rank, (_, r) in enumerate(winners_margin.iterrows(), 1):
        exterior_cn = str(r['best_exterior']) if 'best_exterior' in r and pd.notna(r['best_exterior']) else ''
        lines2.append(f"{rank:<6}{str(r['name'])[:38]:<40}{str(r['series'])[:26]:<28}{str(r['tier'])[:10]:<10}"
                      f"{str(r['next_tier'])[:10]:<10}{exterior_cn:<6}{float(r['price']):>10.2f}{float(r['avg_out_next']):>12.2f}"
                      f"{float(r['value_per_item']):>12.2f}{float(r['margin']):>12.2f}{(float(r['profit_ratio'])*100):>9.2f}%")
    path_win_margin = out_dir / f"固定_炼金期望_正收益TOP100_按利润{label_suffix}.txt"
    with open(path_win_margin, "w", encoding="utf-8") as f:
        f.write("\n".join(lines2))

    # ---------- (3) 负收益 TOP100 - 按利润排序（含 ROI） ----------
    losers_margin = df_work[df_work["margin"] < 0].sort_values(["margin","profit_ratio"], ascending=[True, True]).head(100)
    lines3 = []
    lines3.append("=" * 150)
    lines3.append("炼金期望 负收益 TOP100 - 按利润排序（含 ROI）")
    lines3.append("=" * 150)
    lines3.append("")
    lines3.append(f"{'排名':<6}{'物品名称':<40}{'系列':<28}{'稀有度':<10}{'目标':<10}{'外观':<6}"
                  f"{'单价':>10}{'期望产出':>12}{'单件价值':>12}{'亏损':>12}{'ROI':>10}")
    lines3.append("-" * 150)
    for rank, (_, r) in enumerate(losers_margin.iterrows(), 1):
        exterior_cn = str(r['best_exterior']) if 'best_exterior' in r and pd.notna(r['best_exterior']) else ''
        lines3.append(f"{rank:<6}{str(r['name'])[:38]:<40}{str(r['series'])[:26]:<28}{str(r['tier'])[:10]:<10}"
                      f"{str(r['next_tier'])[:10]:<10}{exterior_cn:<6}{float(r['price']):>10.2f}{float(r['avg_out_next']):>12.2f}"
                      f"{float(r['value_per_item']):>12.2f}{float(r['margin']):>12.2f}{(float(r['profit_ratio'])*100):>9.2f}%")
    path_lose_margin = out_dir / f"固定_炼金期望_负收益TOP100_按利润{label_suffix}.txt"
    with open(path_lose_margin, "w", encoding="utf-8") as f:
        f.write("\n".join(lines3))

    # ---------- (4) 正收益 TOP100 - 按 ROI 排序（含绝对利润） ----------
    winners_roi = df_work[df_work["profit_ratio"] > 0].sort_values(["profit_ratio","margin"], ascending=[False, False]).head(100)
    lines4 = []
    lines4.append("=" * 150)
    lines4.append("炼金期望 正收益 TOP100 - 按 ROI 排序（含绝对利润）")
    lines4.append("=" * 150)
    lines4.append("")
    lines4.append(f"{'排名':<6}{'物品名称':<40}{'系列':<28}{'稀有度':<10}{'目标':<10}{'外观':<6}"
                  f"{'单价':>10}{'期望产出':>12}{'单件价值':>12}{'利润':>12}{'ROI':>10}")
    lines4.append("-" * 150)
    for rank, (_, r) in enumerate(winners_roi.iterrows(), 1):
        exterior_cn = str(r['best_exterior']) if 'best_exterior' in r and pd.notna(r['best_exterior']) else ''
        lines4.append(f"{rank:<6}{str(r['name'])[:38]:<40}{str(r['series'])[:26]:<28}{str(r['tier'])[:10]:<10}"
                      f"{str(r['next_tier'])[:10]:<10}{exterior_cn:<6}{float(r['price']):>10.2f}{float(r['avg_out_next']):>12.2f}"
                      f"{float(r['value_per_item']):>12.2f}{float(r['margin']):>12.2f}{(float(r['profit_ratio'])*100):>9.2f}%")
    path_win_roi = out_dir / f"固定_炼金期望_正收益TOP100_按ROI{label_suffix}.txt"
    with open(path_win_roi, "w", encoding="utf-8") as f:
        f.write("\n".join(lines4))

    # ---------- (5) 负收益 TOP100 - 按 ROI 排序（含绝对利润） ----------
    losers_roi = df_work[df_work["profit_ratio"] < 0].sort_values(["profit_ratio","margin"], ascending=[True, False]).head(100)
    lines5 = []
    lines5.append("=" * 150)
    lines5.append("炼金期望 负收益 TOP100 - 按 ROI 排序（含绝对利润）")
    lines5.append("=" * 150)
    lines5.append("")
    lines5.append(f"{'排名':<6}{'物品名称':<40}{'系列':<28}{'稀有度':<10}{'目标':<10}{'外观':<6}"
                  f"{'单价':>10}{'期望产出':>12}{'单件价值':>12}{'亏损':>12}{'ROI':>10}")
    lines5.append("-" * 150)
    for rank, (_, r) in enumerate(losers_roi.iterrows(), 1):
        exterior_cn = str(r['best_exterior']) if 'best_exterior' in r and pd.notna(r['best_exterior']) else ''
        lines5.append(f"{rank:<6}{str(r['name'])[:38]:<40}{str(r['series'])[:26]:<28}{str(r['tier'])[:10]:<10}"
                      f"{str(r['next_tier'])[:10]:<10}{exterior_cn:<6}{float(r['price']):>10.2f}{float(r['avg_out_next']):>12.2f}"
                      f"{float(r['value_per_item']):>12.2f}{float(r['margin']):>12.2f}{(float(r['profit_ratio'])*100):>9.2f}%")
    path_lose_roi = out_dir / f"固定_炼金期望_负收益TOP100_按ROI{label_suffix}.txt"
    with open(path_lose_roi, "w", encoding="utf-8") as f:
        f.write("\n".join(lines5))

    # 控制台提示
    print(f"\n[固定输出] 已生成5份核心报告：")
    print(f"  1. {path_best_lower.name}")
    print(f"  2. {path_win_margin.name}")
    print(f"  3. {path_lose_margin.name}")
    print(f"  4. {path_win_roi.name}")
    print(f"  5. {path_lose_roi.name}")


if __name__ == "__main__":
    main()


