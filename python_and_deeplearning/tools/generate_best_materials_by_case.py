#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成按箱子分类的最佳主料清单
直接使用科学版计算结果，按箱子重新组织展示
"""

import pandas as pd
import sys

def main():
    print("正在读取科学版计算结果...")
    
    # 读取科学版计算结果
    try:
        df = pd.read_csv('科学版_盈利TOP100.csv')
    except FileNotFoundError:
        print("❌ 找不到 科学版_盈利TOP100.csv 文件")
        sys.exit(1)
    
    print(f"共加载 {len(df)} 条记录")
    
    # 保留所有K值有效的记录（包括盈利和亏损）
    df_profit = df[df['K'].notna()].copy()
    print(f"有效K值记录（全部）：{len(df_profit)} 条")
    print(f"  - 盈利记录：{len(df_profit[df_profit['margin'] > 0])} 条")
    print(f"  - 亏损记录：{len(df_profit[df_profit['margin'] <= 0])} 条")
    
    # 按箱子（series）和材料分组，找出每个材料的最佳利润
    results = []
    
    for _, row in df_profit.iterrows():
        results.append({
            '箱子名称': row['series'],
            '投入材料': row['name'],
            '材料稀有度': row['tier'],
            '材料外观': row['exterior'],
            '材料价格': row['price'],
            '产出稀有度': row['next_tier'],
            '需要数量': int(row['K']),
            '总投入': row['price'] * row['K'],
            '期望产出': row['avg_out_next'],
            '单件利润': row['margin'],
            '总利润': row['margin'] * row['K'],
            'ROI': f"{row['profit_ratio']:.2%}",
            '盈亏状态': '✅盈利' if row['margin'] > 0 else '❌亏损',
            '平台': row['platform']
        })
    
    # 转换为DataFrame并保存
    if results:
        result_df = pd.DataFrame(results)
        
        # 按箱子名称、材料稀有度和利润排序
        rarity_order = ['Consumer', 'Industrial', 'Mil-Spec', 'Restricted', 'Classified', 'Covert']
        result_df['材料稀有度_排序'] = result_df['材料稀有度'].apply(
            lambda x: rarity_order.index(x) if x in rarity_order else 99
        )
        result_df = result_df.sort_values(['箱子名称', '材料稀有度_排序', '总利润'], ascending=[True, True, False])
        result_df = result_df.drop(columns=['材料稀有度_排序'])
        
        # 保存为CSV
        output_file = '最佳主料清单_按箱子分类_完整版.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 已生成: {output_file}")
        print(f"   共 {len(result_df)} 条记录")
        
        # 统计每个箱子的记录数
        case_counts = result_df['箱子名称'].value_counts()
        print(f"\n各箱子记录数 (TOP 10):")
        for case, count in case_counts.head(10).items():
            print(f"  {case}: {count} 条")
        
        # 生成Markdown格式的报告
        generate_markdown_report(result_df)
        
    else:
        print("\n⚠️  没有找到匹配的数据")

def generate_markdown_report(df):
    """生成Markdown格式的报告（完美对齐）"""
    output_file = '最佳主料清单_按箱子分类_完整版.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# CS2 最佳Trade-Up材料清单 - 按箱子分类\n\n")
        f.write(f"**生成时间**: 2025-11-11  \n")
        f.write(f"**总记录数**: {len(df)}  \n\n")
        f.write("**说明**: 展示每个箱子中可盈利的Trade-Up投入材料及其期望收益\n\n")
        f.write("---\n\n")
        
        # 按箱子分组
        for case_name, group in df.groupby('箱子名称', sort=False):
            f.write(f"## 📦 {case_name}\n\n")
            
            # 计算每列的最大宽度
            col_widths = {
                '投入材料': max(len(str(x)) for x in group['投入材料']) + 2,
                '材料稀有度': 10,
                '材料外观': 6,
                '材料价格': 10,
                '产出稀有度': 10,
                '需要数量': 8,
                '总投入': 12,
                '总利润': 12,
                'ROI': 10
            }
            
            # 确保列标题能放得下
            col_widths['投入材料'] = max(col_widths['投入材料'], 12)
            
            # 生成表头
            f.write(f"| {'投入材料':<{col_widths['投入材料']}} | "
                   f"{'材料等级':<{col_widths['材料稀有度']}} | "
                   f"{'外观':<{col_widths['材料外观']}} | "
                   f"{'材料价格':<{col_widths['材料价格']}} | "
                   f"{'产出等级':<{col_widths['产出稀有度']}} | "
                   f"{'需要数量':<{col_widths['需要数量']}} | "
                   f"{'总投入':<{col_widths['总投入']}} | "
                   f"{'总利润':<{col_widths['总利润']}} | "
                   f"{'ROI':<{col_widths['ROI']}} |\n")
            
            # 生成分隔线
            f.write(f"|{'-' * (col_widths['投入材料'] + 2)}|"
                   f"{'-' * (col_widths['材料稀有度'] + 2)}|"
                   f"{'-' * (col_widths['材料外观'] + 2)}|"
                   f"{'-' * (col_widths['材料价格'] + 2)}|"
                   f"{'-' * (col_widths['产出稀有度'] + 2)}|"
                   f"{'-' * (col_widths['需要数量'] + 2)}|"
                   f"{'-' * (col_widths['总投入'] + 2)}|"
                   f"{'-' * (col_widths['总利润'] + 2)}|"
                   f"{'-' * (col_widths['ROI'] + 2)}|\n")
            
            # 生成数据行
            for _, row in group.iterrows():
                material = str(row['投入材料'])
                tier = str(row['材料稀有度'])
                exterior = str(row['材料外观'])
                price = f"¥{row['材料价格']:.2f}"
                next_tier = str(row['产出稀有度'])
                qty = str(row['需要数量'])
                total_cost = f"¥{row['总投入']:.2f}"
                profit = f"¥{row['总利润']:.2f}"
                roi = row['ROI']
                
                f.write(f"| {material:<{col_widths['投入材料']}} | "
                       f"{tier:<{col_widths['材料稀有度']}} | "
                       f"{exterior:<{col_widths['材料外观']}} | "
                       f"{price:>{col_widths['材料价格']}} | "
                       f"{next_tier:<{col_widths['产出稀有度']}} | "
                       f"{qty:>{col_widths['需要数量']}} | "
                       f"{total_cost:>{col_widths['总投入']}} | "
                       f"{profit:>{col_widths['总利润']}} | "
                       f"{roi:>{col_widths['ROI']}} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("**说明**:  \n")
        f.write("- **投入材料**: 用于 Trade-Up 的材料皮肤  \n")
        f.write("- **产出等级**: Trade-Up 后可获得的物品稀有度  \n")
        f.write("- **需要数量**: 完成一次 Trade-Up 需要的材料数量（普通 10 个，红→金 5 个）  \n")
        f.write("- **总投入**: 材料价格 × 需要数量  \n")
        f.write("- **总利润**: 单件利润 × 需要数量（期望值）  \n")
        f.write("- **ROI**: 投资回报率（总利润 / 总投入）  \n")
    
    print(f"✅ 已生成: {output_file}")

if __name__ == "__main__":
    main()
