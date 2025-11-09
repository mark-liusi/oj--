# CS2 Trade-Up 科学版计算 - 完整流程说明

## 📋 每日更新流程

### 方法1：使用批处理脚本（推荐）

直接双击运行：**`每日更新.bat`**

脚本会自动执行以下步骤：
1. 爬取最新价格
2. 提取外观信息
3. 准备计算输入
4. 运行科学版计算
5. 生成 CSV 和 4 个格式化 TXT 文件

### 方法2：手动执行命令

```bash
# 步骤 1: 爬取最新价格
python fetch_prices_with_steamdt.py --platform all --min-only

# 步骤 2: 提取外观信息
python extract_exterior_from_prices.py

# 步骤 3: 准备科学版输入
python prepare_scientific_inputs.py

# 步骤 4: 运行科学版计算
python calculate_scientific.py --input prices_with_exterior.csv --meta skins_meta_real.csv --prices skin_prices.csv --out-csv tradeup_scientific_latest.csv
```

---

## 📁 必需文件清单

### 1. 核心脚本（5个）
- ✅ `fetch_prices_with_steamdt.py` - 价格爬取
- ✅ `extract_exterior_from_prices.py` - 外观提取
- ✅ `prepare_scientific_inputs.py` - 数据准备
- ✅ `calculate_scientific.py` - 科学版计算
- ✅ `calculate.py` - 格式化函数（被 calculate_scientific.py 调用）

### 2. 基础数据（不需要更新）
- ✅ `cs2_case_items_full.csv` - 2379个箱子物品清单
- ✅ `skins_meta_real.csv` - 918个物品的真实浮漂区间

### 3. 每日生成/更新的文件
- `prices_all_min.csv` - 最新价格数据
- `prices_with_exterior.csv` - 包含外观的价格
- `skin_prices.csv` - 外观-价格映射
- `tradeup_scientific_latest.csv` - 计算结果（主文件）
- `科学版_盈利TOP100_按利润.txt` ⭐
- `科学版_亏损TOP100_按利润.txt`
- `科学版_盈利TOP100_按ROI.txt` ⭐
- `科学版_亏损TOP100_按ROI.txt`

### 4. 批处理脚本
- `每日更新.bat` - 自动化更新脚本
- `清理旧文件.bat` - 清理旧版本文件

---

## 🗑️ 可以删除的文件

### 旧版本输出（简单版，已被科学版替代）
- `tradeup_margins_new.csv`
- `tradeup_margins_formatted.txt`
- `winners_by_margin.csv` / `winners_by_margin.txt`
- `losers_by_margin.csv` / `losers_by_margin.txt`
- `winners_by_roi.csv` / `winners_by_roi.txt`
- `losers_by_roi.csv` / `losers_by_roi.txt`
- `tradeup_pots_new.txt`

### 旧版本科学版
- `tradeup_scientific.csv`
- `tradeup_scientific_v2.csv`
- `tradeup_scientific_real.csv`

### 中间/临时文件
- `skins_meta_generated.csv` (已被 skins_meta_real.csv 替代)
- `skins_meta.csv` (空文件)
- `cs2_cases.csv`

**删除方法：运行 `清理旧文件.bat`**

---

## 🔍 calculate_scientific.py 调用的文件

### 输入文件（3个）
1. **`prices_with_exterior.csv`** (必需，每日更新)
   - 来源：`extract_exterior_from_prices.py` 生成
   - 内容：包含外观信息的价格数据
   - 列：name, series, tier, price, exterior

2. **`skins_meta_real.csv`** (必需，不需要更新)
   - 来源：`fetch_real_float_ranges.py` 一次性生成
   - 内容：918个物品的真实浮漂区间
   - 列：name, float_min, float_max, source

3. **`skin_prices.csv`** (必需，每日更新)
   - 来源：`prepare_scientific_inputs.py` 生成
   - 内容：外观-价格映射表
   - 列：name, exterior, price

### 依赖模块
- **`calculate.py`**
  - 用途：导入 `format_to_txt()` 函数用于生成格式化文本

---

## 📊 输出文件说明

### CSV 文件
- **`tradeup_scientific_latest.csv`**
  - 完整计算结果（1899行）
  - 包含所有物品的期望利润和ROI

### TXT 文件（推荐查看）
1. **`科学版_盈利TOP100_按利润.txt`** ⭐
   - 盈利最多的 100 个 Trade-Up
   - TOP 3: SSG 08 Dragonfire (4780元), Five-SeveN Hyper Beast (3236元), M4A4 Buzz Kill (1954元)

2. **`科学版_盈利TOP100_按ROI.txt`** ⭐
   - ROI 最高的 100 个 Trade-Up
   - TOP 3: AWP Chromatic Aberration (264%), MP7 Bloodsport (216%), UMP-45 Grand Prix (213%)

3. **`科学版_亏损TOP100_按利润.txt`**
   - 避开这些亏损项目

4. **`科学版_亏损TOP100_按ROI.txt`**
   - 按 ROI 排序的亏损项目

---

## ⚙️ 高级配置

### 修改 API Key
编辑 `fetch_prices_with_steamdt.py`：
```python
API_KEY = "your_new_api_key_here"
```

### 修改手续费
编辑 `calculate_scientific.py`，添加参数：
```bash
--sell-fee 0.13  # 卖出手续费（默认13%）
--buy-fee 0.0    # 买入手续费（默认0%）
```

### 只计算特定稀有度
```bash
--filter-tier Covert Classified
```

---

## 🔄 工作流程图

```
cs2_case_items_full.csv (物品清单)
    ↓
[fetch_prices_with_steamdt.py] → prices_all_min.csv
    ↓
[extract_exterior_from_prices.py] → prices_with_exterior.csv
    ↓
[prepare_scientific_inputs.py] → skin_prices.csv
    ↓
[calculate_scientific.py] ← skins_meta_real.csv (浮漂区间)
    ↓
tradeup_scientific_latest.csv + 4个TXT文件
```

---

## ❓ 常见问题

### Q1: 多久更新一次？
**建议：每天更新一次**，因为价格会波动

### Q2: 为什么有些物品没有结果？
- 可能没有价格数据
- 可能不在 Trade-Up 链条中（如Gold档位无法继续升级）

### Q3: 如何验证数据正确性？
检查生成文件的时间戳，确保是最新的：
```bash
dir *.csv | sort-object LastWriteTime -Descending
```

### Q4: 计算需要多久？
- 价格爬取：约 5-10 分钟（取决于网络）
- 计算：约 10-30 秒

---

**最后更新：2025-11-09**
