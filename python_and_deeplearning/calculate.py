# -*- coding: utf-8 -*-
"""
CS2 炼金期望扫描脚本（带中文注释）
核心思想：
  - 期望是线性的：一锅K件（普通K=10；红→金K=5）
  - 单件价值贡献 value_per_item = avg_out_next(series, next_tier) / K
  - margin = value_per_item * (1 - sell_fee) - price * (1 + buy_fee)
  - 按 margin 降序排序即可快速找到“最优一锅”（取前K件）

输入：CSV，需包含 列（名称/系列/品质/价格）——列名可自动识别或命令行指定
输出：
  1) 材料明细（按 margin 降序）
  2) 各输入档位“最优一锅”摘要（可选择 repeat 多锅，贪心不复用同一行）
"""

import argparse
import sys
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- 格式化输出工具函数 ----------
def get_display_width(s):
    """计算字符串的显示宽度（中文字符算2个宽度）"""
    width = 0
    for c in str(s):
        if ord(c) > 127:  # 非ASCII字符（包括中文）
            width += 2
        else:
            width += 1
    return width

def pad_string(s, target_width, align='left'):
    """填充字符串到指定显示宽度"""
    s = str(s)
    current_width = get_display_width(s)
    padding = target_width - current_width
    if padding <= 0:
        return s
    if align == 'left':
        return s + ' ' * padding
    elif align == 'right':
        return ' ' * padding + s
    else:  # center
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + s + ' ' * right_pad

def format_to_txt(df: pd.DataFrame, title: str, max_rows: int = 100) -> str:
    """
    将 DataFrame 格式化为对齐的文本表格
    
    Args:
        df: 数据框
        title: 标题
        max_rows: 最多显示行数
    
    Returns:
        格式化的文本字符串
    """
    df_display = df.head(max_rows)
    
    output = []
    output.append("=" * 160)
    output.append(title)
    output.append("=" * 160)
    output.append("")
    
    # 表头
    header_parts = [
        pad_string("排名", 6, 'left'),
        pad_string("物品名称", 48, 'left'),
        pad_string("外观", 8, 'left'),
        pad_string("箱子系列", 36, 'left'),
        pad_string("稀有度", 12, 'left'),
        pad_string("目标", 12, 'left'),
        pad_string("价格", 10, 'right'),
        pad_string("期望产出", 12, 'right'),
        pad_string("单件价值", 12, 'right'),
        pad_string("期望利润", 12, 'right'),
        pad_string("ROI", 10, 'right'),
    ]
    output.append(''.join(header_parts))
    output.append("-" * 168)
    
    # 数据行
    for idx, row in df_display.iterrows():
        rank = df_display.index.get_loc(idx) + 1
        name = str(row.get('name', ''))[:42]
        exterior = str(row.get('best_exterior', ''))[:6] if pd.notna(row.get('best_exterior')) else ""
        series = str(row.get('series', ''))[:32]
        tier = str(row.get('tier', ''))
        next_tier = str(row.get('next_tier', ''))
        price = f"{row.get('price', 0):.2f}" if pd.notna(row.get('price')) else "N/A"
        avg_out = f"{row.get('avg_out_next', 0):.2f}" if pd.notna(row.get('avg_out_next')) else "N/A"
        value = f"{row.get('value_per_item', 0):.2f}" if pd.notna(row.get('value_per_item')) else "N/A"
        margin = f"{row.get('margin', 0):.2f}" if pd.notna(row.get('margin')) else "N/A"
        roi = f"{row.get('profit_ratio', 0):.2%}" if pd.notna(row.get('profit_ratio')) else "N/A"
        
        line_parts = [
            pad_string(str(rank), 6, 'left'),
            pad_string(name, 48, 'left'),
            pad_string(exterior, 8, 'left'),
            pad_string(series, 36, 'left'),
            pad_string(tier, 12, 'left'),
            pad_string(next_tier, 12, 'left'),
            pad_string(price, 10, 'right'),
            pad_string(avg_out, 12, 'right'),
            pad_string(value, 12, 'right'),
            pad_string(margin, 12, 'right'),
            pad_string(roi, 10, 'right'),
        ]
        output.append(''.join(line_parts))
    
    output.append("=" * 168)
    output.append(f"\n总计 {len(df)} 个物品（显示前 {len(df_display)} 个）")
    output.append("\n说明：外观列显示推荐购买的磨损度（基于最低价格）")
    
    return '\n'.join(output)


# ---------- 常量与映射 ----------
# “输入档位 -> 下一档 + 需要数量K”
NEXT_TIER = {
    "Consumer": ("Industrial", 10),
    "Industrial": ("Mil-Spec", 10),
    "Mil-Spec": ("Restricted", 10),
    "Restricted": ("Classified", 10),
    "Classified": ("Covert", 10),
    "Covert": ("Gold", 5),   # 新版：5个红合金
}

# 品质归一化（中英文都识别）
def normalize_tier(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    mapping = {
        # 英文
        "consumer": "Consumer",
        "consumer grade": "Consumer",
        "industrial": "Industrial",
        "industrial grade": "Industrial",
        "mil-spec": "Mil-Spec",
        "mil-spec grade": "Mil-Spec",
        "milspec": "Mil-Spec",
        "restricted": "Restricted",
        "classified": "Classified",
        "covert": "Covert",
        "red": "Covert",
        "gold": "Gold",
        # 常见中文
        "消费级": "Consumer",
        "工业级": "Industrial",
        "军规级": "Mil-Spec",
        "受限级": "Restricted",
        "受限": "Restricted",
        "保密级": "Classified",
        "保密": "Classified",
        "隐秘级": "Covert",
        "隐秘": "Covert",
        "红": "Covert",
        "金": "Gold",
    }
    if s in mapping:
        return mapping[s]
    for k, v in mapping.items():
        if k in s:
            return v
    return x.strip().title()


# ---------- 工具函数 ----------
def robust_read_csv(path: str) -> pd.DataFrame:
    """带编码容错地读 CSV。"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "ansi", "ISO-8859-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    """在 DataFrame 里根据候选名猜测列名（先全等后包含）。"""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def detect_columns(df: pd.DataFrame,
                   name_arg: Optional[str],
                   series_arg: Optional[str],
                   tier_arg: Optional[str],
                   price_arg: Optional[str]) -> Tuple[str, str, str, str]:
    """自动/手动获取 名称/系列/品质/价格 列名。"""
    name_col   = name_arg   or find_col(df, ["name", "item_name", "market_hash_name", "皮肤名", "名称"])
    series_col = series_arg or find_col(df, ["series", "collection", "case", "箱子", "系列", "集合"])
    tier_col   = tier_arg   or find_col(df, ["tier", "rarity", "quality", "grade", "品质", "级别", "稀有度", "稀有"])
    price_col  = price_arg  or find_col(df, ["price", "lowest_price", "current_price", "sell_price", "价格", "售价"])
    missing = [label for col, label in [(name_col, "名称/name"),
                                        (series_col, "系列/collection"),
                                        (tier_col, "品质/tier"),
                                        (price_col, "价格/price")] if col is None]
    if missing:
        print("无法识别必要列：", ", ".join(missing))
        print("CSV 列名为：", list(df.columns))
        sys.exit(1)
    return name_col, series_col, tier_col, price_col


# ---------- 计算主逻辑 ----------
def compute_margins(df: pd.DataFrame,
                    sell_fee: float,
                    buy_fee: float,
                    filter_tier: Optional[List[str]] = None) -> pd.DataFrame:
    """
    给每一行“材料”计算：
      - 所属输入档位 tier（归一化）
      - 下一档 next_tier 与 K
      - 该系列在 next_tier 的平均出货价 avg_out_next（从同 CSV 的数据中按系列+品质取均值）
      - value_per_item = (avg_out_next * (1 - sell_fee)) / K
      - cost = price * (1 + buy_fee)
      - margin = value_per_item - cost
    返回：按 margin 降序排好的明细 DataFrame
    """
    # 归一化并筛选基础列
    work = df.copy()
    work["tier"] = work["tier_raw"].astype(str).map(normalize_tier)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work.dropna(subset=["series", "tier", "price"])

    # 可选：只看指定输入档位
    if filter_tier:
        filt_norm = {normalize_tier(t) for t in filter_tier}
        work = work[work["tier"].isin(filt_norm)]

    # 预计算：系列+品质 的平均价，用于估算下一档均价
    series_tier_avg = (
        work.groupby(["series", "tier"], as_index=False)["price"]
            .mean()
            .rename(columns={"price": "avg_price"})
    )
    avg_map: Dict = {(r["series"], r["tier"]): r["avg_price"] for _, r in series_tier_avg.iterrows()}

    # 下一档与 K
    work[["next_tier", "K"]] = work["tier"].apply(
        lambda t: pd.Series(NEXT_TIER.get(t, (None, None)))
    )
    work = work.dropna(subset=["next_tier", "K"])

    # 查下一档均价（同系列）
    work["avg_out_next"] = work.apply(lambda r: avg_map.get((r["series"], r["next_tier"]), np.nan), axis=1)
    work = work.dropna(subset=["avg_out_next"])

    # 期望与 margin（可选手续费）
    work["value_per_item"] = (work["avg_out_next"] * (1 - sell_fee)) / work["K"]
    work["buy_cost"] = work["price"] * (1 + buy_fee)
    work["margin"] = work["value_per_item"] - work["buy_cost"]

    # 新增：收益比率（期望利润 / 买入成本），避免除零
    work["profit_ratio"] = np.where(work["buy_cost"] > 0, work["margin"] / work["buy_cost"], np.nan)

    # 排序（同 margin 下，再按 value_per_item 降序）
    work = work.sort_values(["margin", "value_per_item"], ascending=[False, False])
    return work


def best_pots_greedy(work: pd.DataFrame, repeat: int = 1) -> pd.DataFrame:
    """
    在每个输入档位内，贪心地“连做 repeat 锅”：
      - 每锅取 top-K（K 由该档位的规则给定）
      - 取完后把这些行从可用材料里移除，继续下一锅
    返回：所有锅的摘要（按 profit_expectation 降序）
    """
    pots = []
    # 我们在每个档位内独立做
    for tier, group in work.groupby("tier", sort=False):
        g = group.copy()
        # 该档位固定的 K
        K = int(g["K"].iloc[0])
        times = 0
        while times < repeat and len(g) >= K:
            # 取该档位 margin 最大的前 K 件
            topk = g.nlargest(K, "margin")
            pot_profit = float(topk["margin"].sum())
            pot_ev = float(topk["value_per_item"].sum())
            pot_cost = float(topk["buy_cost"].sum())

            pots.append({
                "input_tier": tier,
                "K": K,
                "expected_output_sum": round(pot_ev, 4),
                "cost_sum": round(pot_cost, 4),
                "profit_expectation": round(pot_profit, 4),
                "items": "; ".join(topk["name"].astype(str).tolist()),
            })

            # 从可用材料里移除这些行，继续下一锅
            g = g.drop(index=topk.index)
            times += 1

    if pots:
        return pd.DataFrame(pots).sort_values(["profit_expectation", "expected_output_sum"], ascending=False)
    else:
        return pd.DataFrame(columns=["input_tier", "K", "expected_output_sum", "cost_sum", "profit_expectation", "items"])


# ---------- 主入口 ----------
def main():
    ap = argparse.ArgumentParser(description="CS2 炼金期望扫描（按单件margin降序 + 最优一锅）")
    ap.add_argument("--input", required=True, help="输入价格表 CSV 路径")
    ap.add_argument("--out-details", default="tradeup_margins.csv", help="明细输出 CSV（默认 tradeup_margins.csv）")
    ap.add_argument("--out-pots", default="best_pots.csv", help="最优一锅摘要输出 CSV（默认 best_pots.csv）")
    ap.add_argument("--sell-fee", type=float, default=0.0, help="卖出手续费占比，例如 0.15 表示 15%%（默认 0）")
    ap.add_argument("--buy-fee", type=float, default=0.0, help="买入手续费占比（默认 0）")
    ap.add_argument("--filter-tier", type=str, default="", help='只计算指定输入档位，逗号分隔，如 "Mil-Spec,Restricted"')
    ap.add_argument("--repeat", type=int, default=1, help="每个档位贪心连做几锅（默认 1）")

    # 手动列名（可选）
    ap.add_argument("--col-name", type=str, default=None, help="CSV 中：名称列名")
    ap.add_argument("--col-series", type=str, default=None, help="CSV 中：系列/箱子/集合 列名")
    ap.add_argument("--col-tier", type=str, default=None, help="CSV 中：品质/稀有度 列名")
    ap.add_argument("--col-price", type=str, default=None, help="CSV 中：价格列名")

    args = ap.parse_args()

    # 读取 CSV
    df = robust_read_csv(args.input)

    # 自动/手动定位列
    name_col, series_col, tier_col, price_col = detect_columns(
        df, args.col_name, args.col_series, args.col_tier, args.col_price
    )

    # 只保留所需列，并标准化命名
    data = df[[name_col, series_col, tier_col, price_col]].copy()
    data.columns = ["name", "series", "tier_raw", "price"]

    # 解析过滤档位
    filter_tier = [t.strip() for t in args.filter_tier.split(",") if t.strip()] if args.filter_tier else None

    # 计算明细（margin 降序）
    work_sorted = compute_margins(
        data,
        sell_fee=args.sell_fee,
        buy_fee=args.buy_fee,
        filter_tier=filter_tier
    )

    # 对数值列保留小数位数
    numeric_cols = ["price", "avg_out_next", "value_per_item", "buy_cost", "margin"]
    for col in numeric_cols:
        if col in work_sorted.columns:
            work_sorted[col] = work_sorted[col].round(2)
    # 收益比率保留4位小数
    if "profit_ratio" in work_sorted.columns:
        work_sorted["profit_ratio"] = work_sorted["profit_ratio"].round(4)

    # 导出明细（包含所有物品，按 margin 降序）
    work_sorted.to_csv(args.out_details, index=False, encoding="utf-8-sig")
    print(f"✅ 已输出明细（按单件期望利润/ margin 降序）：{args.out_details}")

    # 新增：多种排序榜单
    winners = work_sorted[work_sorted["margin"] > 0].copy()
    losers = work_sorted[work_sorted["margin"] < 0].copy()

    # 1. 按绝对利润排序
    winners_by_margin = winners.sort_values(["margin", "profit_ratio"], ascending=[False, False])
    losers_by_margin = losers.sort_values(["margin", "profit_ratio"], ascending=[True, True])

    winners_by_margin.to_csv("winners_by_margin.csv", index=False, encoding="utf-8-sig")
    losers_by_margin.to_csv("losers_by_margin.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 已输出正收益榜-按绝对利润（{len(winners)} 个物品）：winners_by_margin.csv")
    print(f"✅ 已输出负收益榜-按绝对亏损（{len(losers)} 个物品）：losers_by_margin.csv")
    
    # 生成格式化文本
    with open("winners_by_margin.txt", "w", encoding="utf-8") as f:
        f.write(format_to_txt(winners_by_margin, "正收益榜 - 按绝对利润降序", max_rows=100))
    with open("losers_by_margin.txt", "w", encoding="utf-8") as f:
        f.write(format_to_txt(losers_by_margin, "负收益榜 - 按绝对亏损升序（最亏在前）", max_rows=100))
    print(f"✅ 已生成格式化文本：winners_by_margin.txt, losers_by_margin.txt")

    # 2. 按收益比率排序（性价比榜单）
    winners_by_roi = winners.sort_values(["profit_ratio", "margin"], ascending=[False, False])
    losers_by_roi = losers.sort_values(["profit_ratio", "margin"], ascending=[True, True])

    winners_by_roi.to_csv("winners_by_roi.csv", index=False, encoding="utf-8-sig")
    losers_by_roi.to_csv("losers_by_roi.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 已输出正收益榜-按收益率ROI（{len(winners)} 个物品）：winners_by_roi.csv")
    print(f"✅ 已输出负收益榜-按亏损率ROI（{len(losers)} 个物品）：losers_by_roi.csv")
    
    # 生成格式化文本
    with open("winners_by_roi.txt", "w", encoding="utf-8") as f:
        f.write(format_to_txt(winners_by_roi, "正收益榜 - 按ROI降序（性价比之王）", max_rows=100))
    with open("losers_by_roi.txt", "w", encoding="utf-8") as f:
        f.write(format_to_txt(losers_by_roi, "负收益榜 - 按ROI升序（最不划算）", max_rows=100))
    print(f"✅ 已生成格式化文本：winners_by_roi.txt, losers_by_roi.txt")

    # 计算"最优一锅"（各档位各做 repeat 锅）
    pots_df = best_pots_greedy(work_sorted, repeat=args.repeat)
    
    # 对数值列保留2位小数
    numeric_cols_pots = ["expected_output_sum", "cost_sum", "profit_expectation"]
    for col in numeric_cols_pots:
        if col in pots_df.columns:
            pots_df[col] = pots_df[col].round(2)
    
    pots_df.to_csv(args.out_pots, index=False, encoding="utf-8-sig")
    print(f"✅ 已输出各档位‘最优一锅’摘要：{args.out_pots}")

    # 命令行简要预览
    print("\n=== 预览：明细 Top 10 ===")
    print(work_sorted[["name", "series", "tier", "next_tier", "K", "price", "avg_out_next",
                       "value_per_item", "buy_cost", "margin"]].head(10).to_string(index=False))

    if not pots_df.empty:
        print("\n=== 预览：最优一锅 Top 5 ===")
        print(pots_df.head(5).to_string(index=False))
    else:
        print("\n（没有找到可做的一锅，可能是下一档均价缺失或全部为负期望）")


if __name__ == "__main__":
    main()
