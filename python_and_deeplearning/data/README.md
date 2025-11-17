# 固定数据文件夹

此文件夹包含 CS2 Trade-Up 科学版计算所需的固定元数据文件。

## 📁 文件说明

### 1. `cs2_case_items_full.csv`
- **用途**: CS2 所有箱子物品的完整清单
- **包含字段**: 物品名称、武器类型、涂装、系列、稀有度等
- **更新频率**: 游戏更新时手动维护
- **使用脚本**: `fetch_prices_with_steamdt.py`

### 2. `skins_meta_complete.csv`
- **用途**: CS2 所有皮肤的浮漂区间元数据
- **包含字段**: name, float_min, float_max
- **更新频率**: 游戏更新时自动抓取
- **生成脚本**: `fetch_float_from_github.py`
- **数据来源**: https://github.com/Nereziel/cs2-WeaponPaints

### 3. `name_mapping.csv`
- **用途**: 英文→中文物品名称映射表（用于中文平台搜索）
- **包含字段**: name（英文名）, market_hash_name（中文市场搜索名）
- **更新频率**: 手动维护或自动生成
- **生成脚本**: `generate_cn_name_mapping.py`
- **使用脚本**: `connectors_market.py`（在 BUFF、悠悠有品等中文平台搜索时使用）
- **作用说明**: 
  - 当从中文平台（如 BUFF）抓取价格时，需要用中文名称搜索
  - 例如: `AK-47 | Redline` → `AK-47 | 红线`
  - 包含基础名称和外观变体（如 `(崭新出厂)`、`(久经沙场)` 等）

### 4. `prices_with_exterior.csv`
- **用途**: 统一处理后的外观价格表（含所有平台最低价）
- **包含字段**: name, exterior, price, platform
- **更新频率**: 每日自动生成
- **生成脚本**: `unify_prices.py`（从 `prices_all.csv` 生成）

## 🔄 数据更新流程

```bash
# 1. 更新浮漂元数据（游戏更新时）
python fetch_float_from_github.py

# 2. 生成中文名称映射表（首次使用或游戏新增物品时）
python generate_cn_name_mapping.py --source steam    # 从 Steam 社区市场自动获取
# 或
python generate_cn_name_mapping.py --source manual   # 使用内置的手动映射表

# 3. 每日价格更新（自动化脚本）
./每日更新.bat   # Windows
./每日更新.sh    # Linux
```

## 📝 注意事项

- `cs2_case_items_full.csv` 需要手动维护（游戏新增箱子时更新）
- `skins_meta_complete.csv` 可自动更新但建议定期检查
- `name_mapping.csv` 可通过脚本自动生成，也可手动维护高频物品
- `prices_with_exterior.csv` 由每日更新脚本自动生成，不建议手动编辑

## 🔍 关于 name_mapping.csv 的说明

### 为什么需要这个文件？

当使用 `connectors_market.py` 从中文平台（BUFF、悠悠有品等）抓取价格时，需要使用中文名称进行搜索。例如：

- 英文: `AK-47 | Redline (Field-Tested)`
- 中文: `AK-47 | 红线 (久经沙场)`

### 如何生成完整映射表？

```bash
# 方式1: 从 Steam 社区市场自动获取（推荐）
python generate_cn_name_mapping.py --source steam

# 方式2: 使用内置的手动映射（快速但不完整）
python generate_cn_name_mapping.py --source manual

# 方式3: 从 BUFF 获取（需要登录 cookies）
python generate_cn_name_mapping.py --source buff --buff-cookies cookies.json
```

**建议**: 首次运行使用 `--source steam`，会自动从 Steam 社区市场获取所有物品的中文翻译（约需 30-60 分钟）。

### 当前映射表包含什么？

当前的 `name_mapping.csv` 只包含5个示例物品。运行上述脚本可以生成包含所有物品的完整映射表。
