#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
Compute EV for simple case-pure trade-ups using a prices CSV produced by fetch_prices_with_steamdt.py

Two probability models:
  A) classic  : P(item in collection C) = (k_C / 10) * (1 / n_C)
  B) weighted : P(item in C) = k_C / Σ (k_D * n_D)   (see some community guides)

Usage examples:
  # Evaluate Restricted->Classified for ONE case
  python compute_tradeup_ev.py --items-csv cs2_case_items_full.csv --prices-csv prices_today.csv \
      --in-rarity Restricted --out-rarity Classified --case \"Gallery Case\" --platform steam \
      --fee 0.15 --prob-model A

  # Evaluate ALL cases for Restricted->Classified
  python compute_tradeup_ev.py --items-csv cs2_case_items_full.csv --prices-csv prices_today.csv \
      --in-rarity Restricted --out-rarity Classified --all-cases --platform steam --fee 0.15

Outputs:
  tradeup_ev.csv  - one row per (case, in_rarity, out_rarity) with EV and details
\"\"\"
import argparse, pandas as pd, numpy as np

def pick_prices(df_prices, platform):
    dfp = df_prices.copy()
    if \"platform\" in dfp.columns:
        dfp = dfp[dfp[\"platform\"].str.lower()==platform.lower()]
    return dfp

def avg_price_of_cheapest_inputs(df_case_items, df_prices, k=10):
    merged = df_case_items.merge(df_prices, on=[\"item_name_en\"], how=\"left\")
    merged = merged.dropna(subset=[\"price\"]).sort_values(\"price\")
    pool = merged.head(k)
    return float(pool[\"price\"].mean()), pool[[\"item_name_en\",\"price\"]].to_dict(orient=\"records\")

def outputs_with_prices(df_case_items_out, df_prices):
    m = df_case_items_out.merge(df_prices, on=\"item_name_en\", how=\"left\").dropna(subset=[\"price\"])
    return m[[\"item_name_en\",\"price\",\"case_name_en\"]]

def ev_for_case(df_items, df_prices, case_name, in_rarity, out_rarity, fee=0.15, prob_model=\"A\", k_inputs=10):
    inputs = df_items[(df_items.case_name_en==case_name) & (df_items.rarity_en==in_rarity)]
    if len(inputs)==0: return None
    outs = df_items[(df_items.case_name_en==case_name) & (df_items.rarity_en==out_rarity)]
    if len(outs)==0: return None

    in_avg_price, used_inputs = avg_price_of_cheapest_inputs(inputs, df_prices, k=k_inputs)
    out_prices = outputs_with_prices(outs, df_prices)
    if len(out_prices)==0: return None

    n = len(out_prices)
    p_each = 1.0/n  # case-pure
    ev_gross = float((out_prices[\"price\"] * p_each).sum())
    ev_net = ev_gross * (1.0 - fee) - k_inputs * in_avg_price
    return {
        \"case_name_en\": case_name,
        \"in_rarity\": in_rarity,
        \"out_rarity\": out_rarity,
        \"inputs_avg_price\": round(in_avg_price, 4),
        \"inputs_detail\": used_inputs,
        \"n_outputs\": int(n),
        \"prob_model\": prob_model,
        \"fee\": fee,
        \"ev_gross\": round(ev_gross, 4),
        \"ev_net\": round(ev_net, 4),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--items-csv\", required=True)
    ap.add_argument(\"--prices-csv\", required=True)
    ap.add_argument(\"--in-rarity\", required=True, choices=[\"Restricted\",\"Classified\",\"Covert\"])
    ap.add_argument(\"--out-rarity\", required=True, choices=[\"Classified\",\"Covert\"])
    ap.add_argument(\"--case\", default=\"\")
    ap.add_argument(\"--all-cases\", action=\"store_true\")
    ap.add_argument(\"--platform\", default=\"steam\")
    ap.add_argument(\"--fee\", type=float, default=0.15)
    ap.add_argument(\"--out\", default=\"tradeup_ev.csv\")
    args = ap.parse_args()

    df_items = pd.read_csv(args.items_csv, encoding=\"utf-8-sig\")
    df_prices = pd.read_csv(args.prices_csv, encoding=\"utf-8-sig\")
    if \"platform\" in df_prices.columns:
        df_prices = df_prices[df_prices[\"platform\"].str.lower()==args.platform.lower()]

    cases = [args.case] if args.case else sorted(df_items[\"case_name_en\"].dropna().unique().tolist())
    results = []
    for c in cases:
        r = ev_for_case(df_items, df_prices, c, args.in_rarity, args.out_rarity, fee=args.fee)
        if r: results.append(r)
    pd.DataFrame(results).to_csv(args.out, index=False, encoding=\"utf-8-sig\")
    print(f\"Saved {args.out} rows={len(results)}\")

if __name__ == \"__main__\":
    main()
