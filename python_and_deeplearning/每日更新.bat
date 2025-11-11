@echo off
chcp 65001 > nul
echo ================================================================================
echo CS2 Trade-Up 科学版计算 - 每日更新脚本
echo ================================================================================
echo.

set PYTHON=E:\anaconda\envs\lsPython\python.exe
set STEAMDT_API_KEY=ef24f95ea93b45a3b79c828687b85c4e

echo [步骤 1/4] 爬取最新价格数据（全平台完整数据）...
echo   → 从 STEAM/BUFF/YOUPIN/C5 等9个平台获取所有价格
%PYTHON% fetch_prices_with_steamdt.py --items-csv cs2_case_items_full.csv --out prices_all.csv --platform all
if errorlevel 1 (
    echo ❌ 价格爬取失败！
    pause
    exit /b 1
)
echo ✅ 价格数据已更新
echo.

echo [步骤 2/4] 统一价格格式并选择最低价...
echo   → 处理外观信息、保留平台来源、自动选择每个物品的最低价
%PYTHON% unify_prices.py
if errorlevel 1 (
    echo ❌ 格式统一失败！
    pause
    exit /b 1
)
echo ✅ 格式统一完成
echo.

echo [步骤 3/4] 运行科学版计算...
echo   → 基于全平台最低价计算最优 Trade-Up 方案
%PYTHON% calculate_scientific.py --input prices_with_exterior.csv --meta skins_meta_complete.csv --out-csv 科学版_盈利TOP100.csv
if errorlevel 1 (
    echo ❌ 计算失败！
    pause
    exit /b 1
)
echo ✅ 科学版计算完成
echo.

echo [步骤 4/4] 生成最佳主料清单（按箱子分类）...
echo   → 从科学版结果提取每个箱子的最佳材料组合
%PYTHON% generate_best_materials_by_case.py
if errorlevel 1 (
    echo ❌ 清单生成失败！
    pause
    exit /b 1
)
echo ✅ 最佳主料清单生成完成
echo.

echo ================================================================================
echo ✅ 所有步骤完成！
echo ================================================================================
echo.
echo 生成的文件:
echo   - 科学版_盈利TOP100.csv           (完整数据)
echo   - 科学版_盈利TOP100_按利润.txt    (按利润排序)
echo   - 科学版_盈利TOP100_按ROI.txt     (按ROI排序)
echo   - prices_with_exterior.csv        (含平台来源的价格数据)
echo   - 最佳主料清单_按箱子分类_完整版.csv  (所有箱子的完整材料清单)
echo   - 最佳主料清单_按箱子分类_完整版.md   (Markdown格式清单)
echo.
echo 💰 提示: 所有价格均为全平台最低价（BUFF/YOUPIN/C5/STEAM等）
echo 📦 提示: 最佳主料清单包含39个箱子的456条Trade-Up方案
echo.
echo 按任意键打开盈利TOP文件...
pause > nul
notepad 科学版_盈利TOP100_按ROI.txt
