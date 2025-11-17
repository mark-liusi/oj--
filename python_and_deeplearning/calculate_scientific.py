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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tools.connectors_market import MarketPriceFetcher, FetcherConfig

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

# 外观上限阈值字典（用于目标外观计算）
EXTERIOR_UPPER_THRESHOLDS = {
    "FN": 0.07,
    "MW": 0.15,
    "FT": 0.38,
    "WW": 0.45,
    "BS": 1.00,
}

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
                # 使用中位数而非 iloc[0]，避免排序影响
                price_s = float(rowp["price"].median())
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
# 最佳主料清单模块的依赖导入
# ------------------------------
try:
    from tools.connectors_market import MarketPriceFetcher, FetcherConfig
except Exception:
    # 若用户未放置 connectors_market.py，也不影响主功能
    MarketPriceFetcher, FetcherConfig = None, None

@dataclass
class MaterialCandidate:
    name: str
    series: str
    tier: str
    fmin: float
    fmax: float
    f_allow: float
    strictness_stars: str  # 严格度星级标识 (★×1~5)
    local_price: Optional[float]
    lowfloat_price: Optional[float]
    score_physical: float  # 区间宽度 (fmax - fmin)
    score_economic: Optional[float]  # 单位区间宽度成本 = 价格 / (fmax - fmin)
    ev_double_fn: Optional[float]  # 双FN稳健假设：fn_avg
    ev_double_fn_max: Optional[float]  # 偏贵目标场景：fn_max
    value_per_item: Optional[float]  # 基于 fn_avg 的单件价值
    value_per_item_max: Optional[float]  # 基于 fn_max 的单件价值
    buy_cost: Optional[float]
    margin: Optional[float]  # 基于 fn_avg 的利润
    margin_max: Optional[float]  # 基于 fn_max 的利润
    profit_ratio: Optional[float]  # 基于 fn_avg 的ROI
    profit_ratio_max: Optional[float]  # 基于 fn_max 的ROI

def _normalize_name_key(s: str) -> str:
    return str(s).strip().casefold()

def find_optimal_combo(candidates: List[MaterialCandidate], K: int, 
                       optimize_for: str = "cost") -> Optional[List[MaterialCandidate]]:
    """找到K件材料的最优组合
    
    Args:
        candidates: 候选材料列表
        K: 需要的材料数量
        optimize_for: 优化目标
            - "cost": 最低成本（贪心）
            - "margin": 最高总利润
            - "roi": 最高平均ROI
            - "balanced": 平衡成本和利润
    
    Returns:
        最优组合的材料列表，如果无法组成则返回None
    """
    qualified = [c for c in candidates if c.lowfloat_price is not None]
    if len(qualified) < K:
        return None
    
    if optimize_for == "cost":
        # 简单贪心：按价格排序取前K个
        return sorted(qualified, key=lambda x: x.lowfloat_price)[:K]
    
    elif optimize_for == "margin":
        # 按总利润排序（优先正利润）
        return sorted(qualified, 
                     key=lambda x: (x.margin or -1e18), 
                     reverse=True)[:K]
    
    elif optimize_for == "roi":
        # 按ROI排序（优先正ROI）
        return sorted(qualified,
                     key=lambda x: (x.profit_ratio or -1e18),
                     reverse=True)[:K]
    
    elif optimize_for == "balanced":
        # 平衡策略：综合考虑成本和利润
        # score = margin / sqrt(cost)，鼓励"性价比高"的材料
        def balance_score(c):
            if c.margin is None or c.lowfloat_price is None or c.lowfloat_price <= 0:
                return -1e18
            # 使用平方根惩罚成本，避免过度偏向廉价但利润低的材料
            return c.margin / (c.lowfloat_price ** 0.5)
        
        return sorted(qualified, key=balance_score, reverse=True)[:K]
    
    else:
        # 默认返回成本最低
        return sorted(qualified, key=lambda x: x.lowfloat_price)[:K]

def _normalize_name_key(s: str) -> str:
    return str(s).strip().casefold()

def _smart_markup_for_f_allow(f_allow: float, prices_df: Optional[pd.DataFrame], name: str) -> float:
    """根据 f_allow 计算智能溢价倍数
    
    Args:
        f_allow: 允许的最大浮漂值
        prices_df: 价格表（可能包含多个外观价格用于分位数估计）
        name: 物品名称
    
    Returns:
        溢价倍数（1.0表示无溢价）
    
    逻辑：
        - f_allow ≤ 0.02: 使用FN桶价的P90（极低浮漂）
        - f_allow ≤ 0.03: 使用FN桶价的P75
        - 否则: 使用FN桶价的P60
        - 若无分位数数据，回退到7%固定溢价
    """
    # 尝试从价格表获取该物品的FN价格分位数
    if prices_df is not None:
        name_key = _normalize_name_key(name)
        item_prices = prices_df[
            (prices_df["name"].str.casefold() == name_key) & 
            (prices_df["exterior"] == "FN")
        ]["price"]
        
        if len(item_prices) >= 3:  # 至少需要3个数据点来估计分位数
            if f_allow <= 0.02:
                # 极低浮漂：P90溢价
                p90 = item_prices.quantile(0.90)
                median = item_prices.median()
                return (p90 / median) if median > 0 else 1.07
            elif f_allow <= 0.03:
                # 很低浮漂：P75溢价
                p75 = item_prices.quantile(0.75)
                median = item_prices.median()
                return (p75 / median) if median > 0 else 1.07
            else:
                # 低浮漂：P60溢价
                p60 = item_prices.quantile(0.60)
                median = item_prices.median()
                return (p60 / median) if median > 0 else 1.07
    
    # 回退到固定溢价
    if f_allow <= 0.02:
        return 1.15  # 极低浮漂：15%溢价
    elif f_allow <= 0.03:
        return 1.10  # 很低浮漂：10%溢价
    else:
        return 1.07  # 一般低浮漂：7%溢价

def _series_theta_star(series: str, next_tier: str, df_all: pd.DataFrame, meta_df: Optional[pd.DataFrame], target_exterior: str = "FN") -> Optional[float]:
    """计算该 series 的下一档所有候选目标的归一化阈值的最小值（双目标外观稳健条件）。"""
    # 获取目标外观的浮漂阈值
    target_threshold = EXTERIOR_UPPER_THRESHOLDS.get(target_exterior, 0.07)  # FN=0.07, MW=0.15, 等
    
    mask_target = (df_all["series"].str.casefold()==_normalize_name_key(series)) & (df_all["tier"]==next_tier)
    S = df_all[mask_target]
    if S.empty: 
        return None
    thetas = []
    for _, r in S.iterrows():
        s_name = r["name"]
        # 读目标的 min/max
        fmin_out=fmax_out=None
        if meta_df is not None:
            row = meta_df[meta_df["name"].str.casefold()==_normalize_name_key(s_name)]
            if not row.empty:
                fmin_out = float(row.iloc[0]["float_min"])
                fmax_out = float(row.iloc[0]["float_max"])
        if fmin_out is None or fmax_out is None or fmax_out<=fmin_out:
            fmin_out, fmax_out = 0.0, 1.0  # 回退
        # 目标外观归一化阈值
        theta = (target_threshold - fmin_out) / max(1e-9, (fmax_out - fmin_out))
        thetas.append(theta)
    if not thetas: 
        return None
    return max(0.0, min(1.0, min(thetas)))  # 双目标外观取最严阈值

def _expected_out_price_fn(series: str, next_tier: str, df_all: pd.DataFrame, prices_df: Optional[pd.DataFrame], target_exterior: str = "FN") -> Optional[float]:
    """期望的"输出为指定外观"时的稳健平均售价（使用中位数）。"""
    mask_target = (df_all["series"].str.casefold()==_normalize_name_key(series)) & (df_all["tier"]==next_tier)
    S = df_all[mask_target]
    if S.empty or prices_df is None:
        return None
    acc = []; 
    for _, r in S.iterrows():
        s_name = r["name"]
        rowp = prices_df[(prices_df["name"].str.casefold()==_normalize_name_key(s_name)) & (prices_df["exterior"]==target_exterior)]
        if not rowp.empty:
            # 使用中位数而非 iloc[0]，避免排序影响
            acc.append(float(rowp["price"].median()))
        elif "price" in S.columns and not math.isnan(r.get("price", float("nan"))):
            acc.append(float(r["price"]))  # 回退：用主表价格
    if not acc:
        return None
    return float(sum(acc)/len(acc))

def _expected_out_price_fn_max(series: str, next_tier: str, df_all: pd.DataFrame, prices_df: Optional[pd.DataFrame], target_exterior: str = "FN") -> Optional[float]:
    """期望的"输出为指定外观"时的最高售价（偏贵目标场景：瞄准贵隐秘）。"""
    mask_target = (df_all["series"].str.casefold()==_normalize_name_key(series)) & (df_all["tier"]==next_tier)
    S = df_all[mask_target]
    if S.empty or prices_df is None:
        return None
    acc = []; 
    for _, r in S.iterrows():
        s_name = r["name"]
        rowp = prices_df[(prices_df["name"].str.casefold()==_normalize_name_key(s_name)) & (prices_df["exterior"]==target_exterior)]
        if not rowp.empty:
            # 使用最大值而非 iloc[0]，用于偏贵目标场景
            acc.append(float(rowp["price"].max()))
        elif "price" in S.columns and not math.isnan(r.get("price", float("nan"))):
            acc.append(float(r["price"]))  # 回退：用主表价格
    if not acc:
        return None
    return float(max(acc))  # 返回最高价而非平均价

def best_materials_for_series(
    series: str,
    source_tier: str,
    df_all: pd.DataFrame,
    meta_df: Optional[pd.DataFrame],
    prices_df: Optional[pd.DataFrame],
    sell_fee: float,
    buy_fee: float,
    fetcher: Any,  # Optional[MarketPriceFetcher]，但为了避免导入问题使用 Any
    topn_external: int = 3,
    target_exterior: str = "FN"
) -> Tuple[List[MaterialCandidate], Optional[MaterialCandidate], Optional[MaterialCandidate]]:
    """
    返回：
      - candidates: 全部候选（含物理&经济指标）
      - physical_best: 物理榜第一
      - economic_best: 经济榜第一
    """
    if source_tier not in NEXT_TIER:
        return [], None, None
    next_tier, K = NEXT_TIER[source_tier]
    # 计算双目标外观的严格阈值
    theta_star = _series_theta_star(series, next_tier, df_all, meta_df, target_exterior)
    if theta_star is None:
        return [], None, None
    # 输出目标外观均价（稳健） 和 最高价（偏贵目标）
    fn_avg = _expected_out_price_fn(series, next_tier, df_all, prices_df, target_exterior)
    fn_max = _expected_out_price_fn_max(series, next_tier, df_all, prices_df, target_exterior)

    # 枚举该 series 的 source_tier 作为下级的所有材料
    M = df_all[(df_all["series"].str.casefold()==_normalize_name_key(series)) & (df_all["tier"]==source_tier)].copy()
    if M.empty:
        return [], None, None

    # 先按“物理可行性”粗排，筛出前 topn_external 个去外部抓“低漂价”
    M["fmin"] = 0.0; M["fmax"] = 1.0
    if meta_df is not None:
        m2 = meta_df.set_index(meta_df["name"].str.casefold())
        keys = M["name"].str.casefold()
        fmin = m2.reindex(keys)["float_min"].values
        fmax = m2.reindex(keys)["float_max"].values
        M.loc[:, "fmin"] = fmin
        M.loc[:, "fmax"] = fmax
        # 清洗极端/缺失
        M.loc[(M["fmax"]<=M["fmin"]) | M["fmin"].isna() | M["fmax"].isna(), ["fmin","fmax"]] = [0.0,1.0]

    # f_allow = fmin + theta_star * (fmax - fmin)，并clamp到合理范围
    M["f_allow"] = M["fmin"] + theta_star * (M["fmax"] - M["fmin"])
    M["f_allow"] = M[["f_allow", "fmin", "fmax"]].apply(
        lambda row: np.clip(row["f_allow"], row["fmin"], row["fmax"]), axis=1
    )
    
    # 计算严格度星级（f_allow接近fmin程度）：1-5星
    # 严格度 = (f_allow - fmin) / (fmax - fmin)，越小越严格
    M["strictness"] = ((M["f_allow"] - M["fmin"]) / (M["fmax"] - M["fmin"])).fillna(0.5)
    M["strictness_stars"] = M["strictness"].apply(lambda x: 
        "★★★★★" if x <= 0.1 else
        "★★★★" if x <= 0.2 else
        "★★★" if x <= 0.3 else
        "★★" if x <= 0.5 else
        "★"
    )

    # 物理榜分数：区间宽度（越大越容易凑双FN）
    M["score_physical"] = (M["fmax"] - M["fmin"]).astype(float)

    # 先按物理榜降序，挑 Top-N 做外部抓取
    M = M.sort_values("score_physical", ascending=False).reset_index(drop=True)
    # 局部副本，用于填充 lowfloat 价格
    M["lowfloat_price"] = None

    # 准备本地均价字典（兜底）
    local_price_map = {}
    if prices_df is not None:
        tmp = prices_df.groupby("name")["price"].median().reset_index()
        for _, r in tmp.iterrows():
            local_price_map[_normalize_name_key(r["name"])] = float(r["price"])
    else:
        # 主表 price 兜底
        tmp = M[["name","price"]].dropna()
        for _, r in tmp.iterrows():
            local_price_map[_normalize_name_key(r["name"])] = float(r["price"])

    # 外部抓取（Top-N）
    if fetcher is not None and topn_external>0:
        for i in range(min(topn_external, len(M))):
            nm = M.loc[i, "name"]
            f_allow = float(M.loc[i, "f_allow"])
            local_p = local_price_map.get(_normalize_name_key(nm))
            try:
                p = fetcher.get_lowfloat_price(nm, f_allow, local_p)
            except Exception:
                p = None
            M.loc[i, "lowfloat_price"] = p

    # 其余未抓到的，回退到"本地均价 + 智能溢价"（基于f_allow的分位数模型）
    for i in range(len(M)):
        if pd.isna(M.loc[i, "lowfloat_price"]) or M.loc[i, "lowfloat_price"] is None:
            nm = M.loc[i, "name"]
            f_allow = float(M.loc[i, "f_allow"])
            lp = local_price_map.get(_normalize_name_key(nm))
            if lp is not None:
                # 使用智能溢价而非固定7%
                markup_multiplier = _smart_markup_for_f_allow(f_allow, prices_df, nm)
                M.loc[i, "lowfloat_price"] = float(lp) * markup_multiplier
            else:
                M.loc[i, "lowfloat_price"] = None

    # 计算经济指标（单位区间宽度成本、EV、margin、ROI）
    cands: List[MaterialCandidate] = []
    for _, r in M.iterrows():
        lp = r["lowfloat_price"]
        if lp is None or (isinstance(lp, float) and math.isnan(lp)):
            # 没法定价，跳过经济指标，但仍可参与物理榜
            cand = MaterialCandidate(
                name=r["name"], series=r["series"], tier=r["tier"],
                fmin=float(r["fmin"]), fmax=float(r["fmax"]), f_allow=float(r["f_allow"]),
                strictness_stars=str(r.get("strictness_stars", "★")),
                local_price=local_price_map.get(_normalize_name_key(r["name"])),
                lowfloat_price=None,
                score_physical=float(r["score_physical"]),
                score_economic=None, ev_double_fn=None, ev_double_fn_max=None,
                value_per_item=None, value_per_item_max=None,
                buy_cost=None, margin=None, margin_max=None,
                profit_ratio=None, profit_ratio_max=None
            )
            cands.append(cand); 
            continue

        width = max(1e-9, float(r["fmax"] - r["fmin"]))
        score_econ = float(lp) / width

        # 输出端：双FN稳健假设 → 期望价约等于“目标FN均价”
        ev_out = fn_avg
        ev_out_max = fn_max
        
        # 基于 fn_avg 的计算（稳健）
        if ev_out is None:
            value_per_item = None
            margin = None
            roi = None
        else:
            # 单件投入价值（含卖出手续费）
            value_per_item = float(ev_out) * (1.0 - sell_fee) / float(K)
            buy_cost = float(lp) * (1.0 + buy_fee)
            margin = value_per_item - buy_cost
            roi = margin / buy_cost if buy_cost>0 else None
        
        # 基于 fn_max 的计算（偏贵目标）
        if ev_out_max is None:
            value_per_item_max = None
            margin_max = None
            roi_max = None
        else:
            value_per_item_max = float(ev_out_max) * (1.0 - sell_fee) / float(K)
            buy_cost_max = float(lp) * (1.0 + buy_fee)
            margin_max = value_per_item_max - buy_cost_max
            roi_max = margin_max / buy_cost_max if buy_cost_max>0 else None

        cand = MaterialCandidate(
            name=r["name"], series=r["series"], tier=r["tier"],
            fmin=float(r["fmin"]), fmax=float(r["fmax"]), f_allow=float(r["f_allow"]),
            strictness_stars=str(r.get("strictness_stars", "★")),
            local_price=local_price_map.get(_normalize_name_key(r["name"])),
            lowfloat_price=float(lp),
            score_physical=float(r["score_physical"]),
            score_economic=float(score_econ),
            ev_double_fn=float(ev_out) if ev_out is not None else None,
            ev_double_fn_max=float(ev_out_max) if ev_out_max is not None else None,
            value_per_item=float(value_per_item) if value_per_item is not None else None,
            value_per_item_max=float(value_per_item_max) if value_per_item_max is not None else None,
            buy_cost=float(lp) * (1.0 + buy_fee) if lp is not None else None,
            margin=float(margin) if margin is not None else None,
            margin_max=float(margin_max) if margin_max is not None else None,
            profit_ratio=float(roi) if roi is not None else None,
            profit_ratio_max=float(roi_max) if roi_max is not None else None
        )
        cands.append(cand)

    # 物理榜第一：score_physical 最大
    physical_best = None
    if cands:
        physical_best = sorted(cands, key=lambda x: x.score_physical, reverse=True)[0]

    # 经济榜第一：优先 margin>0 最高；其次 ROI；再其次 score_economic 最小
    economic_best = None
    econ_sorted = sorted(
        [c for c in cands if c.lowfloat_price is not None],
        key=lambda x: (
            -999 if (x.margin is not None and x.margin>0) else 0,   # 正利润优先
            -(x.margin if x.margin is not None else -1e18),
            -(x.profit_ratio if x.profit_ratio is not None else -1e18),
            (x.score_economic if x.score_economic is not None else 1e18),
        )
    )
    economic_best = econ_sorted[0] if econ_sorted else None

    return cands, physical_best, economic_best

def write_best_list(csv_path: str, md_path: str, results: List[Tuple[str, str, List[MaterialCandidate], Optional[MaterialCandidate], Optional[MaterialCandidate]]]):
    """把每个 series 的结果写到 CSV 与 Markdown。"""
    import csv
    # CSV
    rows = []
    for series, source_tier, cands, phys, econ in results:
        # 只取 Top3 以节省篇幅
        top3 = sorted(cands, key=lambda x: (-(x.margin or -1e18), x.score_economic or 1e18))[:3]
        def cand_to_row(tag: str, c: MaterialCandidate):
            return {
                "series": series,
                "source_tier": source_tier,
                "tag": tag,
                "name": c.name,
                "fmin": c.fmin, "fmax": c.fmax, "f_allow": c.f_allow,
                "strictness_stars": c.strictness_stars,
                "lowfloat_price": c.lowfloat_price,
                "score_physical": c.score_physical,
                "score_economic": c.score_economic,
                "ev_double_fn": c.ev_double_fn,
                "ev_double_fn_max": c.ev_double_fn_max,
                "value_per_item": c.value_per_item,
                "value_per_item_max": c.value_per_item_max,
                "buy_cost": c.buy_cost,
                "margin": c.margin,
                "margin_max": c.margin_max,
                "profit_ratio": c.profit_ratio,
                "profit_ratio_max": c.profit_ratio_max,
            }
        if phys: rows.append(cand_to_row("physical_best", phys))
        if econ: rows.append(cand_to_row("economic_best", econ))
        for idx, c in enumerate(top3, 1):
            rows.append(cand_to_row(f"candidate_top{idx}", c))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "series","source_tier","tag","name","fmin","fmax","f_allow","strictness_stars","lowfloat_price","score_physical","score_economic","ev_double_fn","ev_double_fn_max","value_per_item","value_per_item_max","buy_cost","margin","margin_max","profit_ratio","profit_ratio_max"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 每个系列最佳下级（主料）清单\n\n")
        for series, source_tier, cands, phys, econ in results:
            next_tier, K = NEXT_TIER[source_tier]
            f.write(f"## {series}（{source_tier} → {next_tier}）\n\n")
            if phys:
                f.write(f"- 物理最佳：**{phys.name}**（区间 {phys.fmin:.2f}–{phys.fmax:.2f}，f_allow≤{phys.f_allow:.4f} {phys.strictness_stars}）\n")
            if econ:
                f.write(f"- 经济最佳：**{econ.name}**（低漂价≈{econ.lowfloat_price:.2f}）\n")
                f.write(f"  - 稳健(fn_avg): 价值≈{(econ.value_per_item or 0):.2f}, 利润≈{(econ.margin or 0):.2f}, ROI≈{(econ.profit_ratio or 0):.2%}\n")
                f.write(f"  - 偏贵(fn_max): 价值≈{(econ.value_per_item_max or 0):.2f}, 利润≈{(econ.margin_max or 0):.2f}, ROI≈{(econ.profit_ratio_max or 0):.2%}\n")
            
            # === 新增：多种混合同组合策略 ===
            combo_strategies = {
                "最低成本": "cost",
                "最高利润": "margin", 
                "最高ROI": "roi",
                "平衡性价比": "balanced"
            }
            
            f.write("\n### 混合同组合推荐（满足双FN阈值）\n\n")
            for strategy_name, strategy_key in combo_strategies.items():
                combo = find_optimal_combo(cands, int(K), optimize_for=strategy_key)
                if combo:
                    combo_cost = sum(c.lowfloat_price for c in combo)
                    combo_total_margin = sum(c.margin or 0 for c in combo) if all(c.margin is not None for c in combo) else None
                    combo_avg_roi = (combo_total_margin / combo_cost) if (combo_total_margin is not None and combo_cost > 0) else None
                    
                    f.write(f"**{strategy_name}策略**（{K}件）：\n")
                    f.write(f"- 总成本: {combo_cost:.2f} | ")
                    if combo_total_margin is not None:
                        profit_status = "✅ 盈利" if combo_total_margin > 0 else "❌ 亏损"
                        f.write(f"总期望利润: {combo_total_margin:.2f} {profit_status} | ")
                    if combo_avg_roi is not None:
                        f.write(f"ROI: {combo_avg_roi:.2%}")
                    f.write("\n")
                    f.write(f"- 组合: {', '.join([f'{c.name}({c.lowfloat_price:.1f}₽)' for c in combo])}\n\n")
            
            # Top3 简表
            f.write("\n| 候选 | 名称 | f_min | f_max | f_allow | 难度 | 低漂价 | 物理分 | 经济分 | 价值(avg) | 价值(max) | 利润(avg) | 利润(max) | ROI(avg) | ROI(max) |\n")
            f.write("|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            top3 = sorted(cands, key=lambda x: (-(x.margin or -1e18), x.score_economic or 1e18))[:3]
            for i, c in enumerate(top3, 1):
                f.write(f"| Top{i} | {c.name} | {c.fmin:.2f} | {c.fmax:.2f} | {c.f_allow:.4f} | {c.strictness_stars} | {'' if c.lowfloat_price is None else f'{c.lowfloat_price:.2f}'} | {c.score_physical:.2f} | {'' if c.score_economic is None else f'{c.score_economic:.2f}'} | {'' if c.value_per_item is None else f'{c.value_per_item:.2f}'} | {'' if c.value_per_item_max is None else f'{c.value_per_item_max:.2f}'} | {'' if c.margin is None else f'{c.margin:.2f}'} | {'' if c.margin_max is None else f'{c.margin_max:.2f}'} | {'' if c.profit_ratio is None else f'{c.profit_ratio:.2%}'} | {'' if c.profit_ratio_max is None else f'{c.profit_ratio_max:.2%}'} |\n")
            f.write("\n")


def run_best_list_pipeline(
    df: pd.DataFrame,
    meta_df: Optional[pd.DataFrame],
    prices_df: Optional[pd.DataFrame],
    sell_fee: float,
    buy_fee: float,
    price_source: str,
    out_csv: str,
    out_md: str,
    target_source_tiers: Optional[List[str]] = None,
    topn_per_series: int = 3,
    target_exterior: str = "FN"
):
    """
    主入口：生成每个 series 的最佳下级清单（物理 + 经济）。
    """
    # 统一字段名
    cols = detect_columns(df)
    df = df.rename(columns={cols["name"]:"name", cols["series"]:"series", cols["tier"]:"tier"})
    df["tier"] = df["tier"].map(normalize_tier)
    df = df[["name","series","tier","price"] + ([cols["float"]] if cols.get("float") else [])].copy()
    # 标准化 name/series 用于 join
    df["series"] = df["series"].astype(str)
    df["name"] = df["name"].astype(str)

    # 价格表标准化：name, exterior, price（仅三列必需）
    if prices_df is not None:
        try:
            pmap = detect_price_columns(prices_df)  # ← 用专用函数
            prices_df = prices_df.rename(columns={
                pmap["name"]: "name",
                pmap["exterior"]: "exterior",
                pmap["price"]: "price",
            })
            prices_df["exterior"] = prices_df["exterior"].map(normalize_exterior)
            prices_df = prices_df.dropna(subset=["exterior","price"])
        except ValueError as e:
            print(f"警告：价格表列识别失败（{e}），已忽略价格表。")
            prices_df = None

    # 选择源层级
    tiers_all = ["Restricted","Classified","Covert"]
    target_source_tiers = target_source_tiers or tiers_all

    # 构造抓取器（离线也能跑，会回退）
    fetcher = None
    if MarketPriceFetcher is not None:
        fetcher = MarketPriceFetcher(FetcherConfig(source=price_source))

    results = []
    for series in sorted(df["series"].unique()):
        for src in target_source_tiers:
            cands, phys, econ = best_materials_for_series(
                series=series, source_tier=src, df_all=df, meta_df=meta_df, prices_df=prices_df,
                sell_fee=sell_fee, buy_fee=buy_fee, fetcher=fetcher, topn_external=topn_per_series,
                target_exterior=target_exterior
            )
            if cands:
                results.append((series, src, cands, phys, econ))

    write_best_list(out_csv, out_md, results)
    return results


# ------------------------------
# 主流程

# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="CS2 炼金期望（科学版，考虑磨损传导）")
    ap.add_argument("--input","-i", required=True, help="主CSV：包含 name/series/tier/price；可选 float/exterior/stattrak")
    ap.add_argument("--meta","-m", default=None, help="皮肤浮漂元数据 CSV：name,float_min,float_max")
    ap.add_argument("--prices","-p", default=None, help="外观价格 CSV：name,exterior(FN/MW/FT/WW/BS),price")
    ap.add_argument("--sell-fee", type=float, default=0.00, help="卖出手续费比例（默认 0.00）")
    ap.add_argument("--buy-fee", type=float, default=0.00, help="买入手续费比例（默认 0.00）")
    ap.add_argument("--filter-tier", default=None, nargs="*", help="仅计算这些输入稀有度（可多选），支持中英别名")
    ap.add_argument("--out-csv", default=None, help="导出结果 CSV 路径")
    ap.add_argument("--best-list-out", default=None, help="输出每个系列的最佳下级清单（CSV 路径）；将同时生成同名 .md")
    ap.add_argument("--price-source", default="hybrid", help="低漂价来源：local/csfloat/buff/youpin/hybrid（默认 hybrid）")
    ap.add_argument("--topn-per-series", type=int, default=3, help="每个系列仅对 Top-N 候选做外部抓取（默认 3）")
    ap.add_argument("--target-exterior", default="FN", choices=["FN","MW","FT","WW","BS"], 
                    help="目标输出外观（默认 FN=崭新出厂）。支持：FN/MW/FT/WW/BS")
    ap.add_argument("--split-st", action="store_true", 
                    help="分离 StatTrak™：生成 ST/非ST 两份独立榜单，避免混算（ST 溢价通常巨大）")
    args = ap.parse_args()

    # === 新增：若需要输出“最佳下级清单”，在主流程前先加载价格/元数据供管道使用 ===
    best_list_out = args.best_list_out
    price_source = (args.price_source or 'hybrid')
    topn_per_series = int(args.topn_per_series or 3)


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
            
            # 计算期望值
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
        
        df_work["avg_out_next"] = ev_list  # 保持原变量名：期望产出价（候选均值）
        df_work["best_exterior"] = best_exterior_list  # 新增：推荐外观

        # 价值与利润（保持原管线公式与变量名）
        sell_fee = float(args.sell_fee)
        buy_fee = float(args.buy_fee)
        df_work["value_per_item"] = (df_work["avg_out_next"] * (1.0 - sell_fee)) / df_work["K"]
        df_work["buy_cost"] = df_work["price"] * (1.0 + buy_fee)
        df_work["margin"] = df_work["value_per_item"] - df_work["buy_cost"]
        df_work["profit_ratio"] = df_work["margin"] / df_work["buy_cost"]

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
        
        # ===== 生成格式化 TXT 文件 =====
        def format_to_txt(df, title, max_rows=100):
            """将DataFrame格式化为易读的TXT文本"""
            lines = []
            lines.append("="*100)
            lines.append(title)
            lines.append("="*100)
            lines.append("")
            
            # 表头
            lines.append(f"{'排名':<6}{'物品名称':<40}{'外观':<8}{'箱子系列':<30}{'稀有度':<12}{'目标':<12}{'价格':<10}{'期望产出':<10}{'单件价值':<10}{'期望利润':<12}{'ROI':<10}")
            lines.append("-"*150)
            
            # 数据行
            for idx, row in df.head(max_rows).iterrows():
                name = str(row.get('name', ''))[:38]
                series = str(row.get('series', ''))[:28]
                tier = str(row.get('tier', ''))[:10]
                next_tier = str(row.get('next_tier', ''))[:10]
                price = row.get('price', 0)
                avg_out = row.get('avg_out_next', 0)
                value = row.get('value_per_item', 0)
                margin = row.get('margin', 0)
                roi = row.get('profit_ratio', 0)
                best_ext = row.get('best_exterior', '')
                
                lines.append(f"{idx+1:<6}{name:<40}{best_ext:<8}{series:<30}{tier:<12}{next_tier:<12}{price:<10.2f}{avg_out:<10.2f}{value:<10.2f}{margin:<12.2f}{roi*100:<9.2f}%")
            
            return "\n".join(lines)
        
        # 1. 按利润排序
        winners_by_margin = work_sorted[work_sorted["margin"] > 0].head(100)
        losers_by_margin = work_sorted[work_sorted["margin"] < 0].tail(100).iloc[::-1]
        
        # 2. 按 ROI 排序
        winners_by_roi = work_sorted[work_sorted["profit_ratio"] > 0].sort_values("profit_ratio", ascending=False).head(100)
        losers_by_roi = work_sorted[work_sorted["profit_ratio"] < 0].sort_values("profit_ratio", ascending=True).head(100)
        
        # 3. 新增：全部物品 - 按利润排序（包含所有物品，不排除Gold）
        all_items = work_sorted.copy()
        
        # 3. 生成 TXT 文件
        output_dir = Path(args.out_csv).parent if args.out_csv else Path(".")
        
        # 文件名后缀
        suffix = f"_{label}" if split_st and label != "All" else ""
        
        txt_files = {
            f"科学版_盈利TOP100_按利润{suffix}.txt": (winners_by_margin, f"{label} 科学版 Trade-Up 盈利 TOP 100 - 按期望利润排序"),
            f"科学版_亏损TOP100_按利润{suffix}.txt": (losers_by_margin, f"{label} 科学版 Trade-Up 亏损 TOP 100 - 按期望利润排序"),
            f"科学版_盈利TOP100_按ROI{suffix}.txt": (winners_by_roi, f"{label} 科学版 Trade-Up 盈利 TOP 100 - 按 ROI 排序"),
            f"科学版_亏损TOP100_按ROI{suffix}.txt": (losers_by_roi, f"{label} 科学版 Trade-Up 亏损 TOP 100 - 按 ROI 排序"),
            f"科学版_全部物品_按利润{suffix}.txt": (all_items, f"{label} 科学版 Trade-Up 全部物品（含炼金物品）- 按期望利润排序"),
        }
        
        for filename, (data, title) in txt_files.items():
            if len(data) > 0:
                txt_path = output_dir / filename
                with open(txt_path, 'w', encoding='utf-8') as f:
                    # 对于全部物品的文件，不限制行数；其他文件保持100行限制
                    max_rows = len(data) if "全部物品" in filename else 100
                    f.write(format_to_txt(data, title, max_rows=max_rows))
                print(f"已生成格式化文本：{txt_path}")
        
        # 保存本次结果
        all_results[label] = work_sorted
    
    print("\n所有文件生成完成！")

    # === 新增：输出每个系列的最佳下级清单 ===
    if best_list_out:
        try:
            out_csv = Path(best_list_out)
            out_md = out_csv.with_suffix(".md")
            # 价格表：优先 args.prices；否则尝试主表 + 外观价兜底
            prices_df_for_best = prices_df if "prices_df" in locals() else None
            # 元数据：meta_df（已在上文读取）
            run_best_list_pipeline(
                df=df.copy(), meta_df=meta_df, prices_df=prices_df_for_best,
                sell_fee=float(args.sell_fee), buy_fee=float(args.buy_fee),
                price_source=price_source, out_csv=str(out_csv), out_md=str(out_md),
                target_source_tiers=None, topn_per_series=topn_per_series,
                target_exterior=args.target_exterior
            )
            print(f"已生成\"最佳下级清单\"（目标外观={args.target_exterior}）：{out_csv} 和 {out_md}")
        except Exception as e:
            import traceback
            print(f"生成\"最佳下级清单\"失败：{e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()


