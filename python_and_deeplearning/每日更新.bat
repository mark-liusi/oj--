@echo off
chcp 65001 > nul
echo ================================================================================
echo CS2 Trade-Up 科学版计算 - 每日更新脚本
echo ================================================================================
echo.

set PYTHON=E:\anaconda\envs\lsPython\python.exe
set STEAMDT_API_KEY=ef24f95ea93b45a3b79c828687b85c4e

echo [步骤 1/4] 爬取最新价格数据...
%PYTHON% fetch_prices_with_steamdt.py --items-csv cs2_case_items_full.csv --out prices_all_min.csv --platform all --min-only
if errorlevel 1 (
    echo ❌ 价格爬取失败！
    pause
    exit /b 1
)
echo ✅ 价格数据已更新
echo.

echo [步骤 2/4] 提取外观信息...
%PYTHON% extract_exterior_from_prices.py
if errorlevel 1 (
    echo ❌ 外观提取失败！
    pause
    exit /b 1
)
echo ✅ 外观信息已提取
echo.

echo [步骤 3/4] 准备科学版输入数据...
%PYTHON% prepare_scientific_inputs.py --input prices_with_exterior.csv --meta-out skins_meta_real.csv --prices-out skin_prices.csv
if errorlevel 1 (
    echo ❌ 数据准备失败！
    pause
    exit /b 1
)
echo ✅ 输入数据已准备
echo.

echo [步骤 4/4] 运行科学版计算...
%PYTHON% calculate_scientific.py --input prices_with_exterior.csv --meta skins_meta_real.csv --prices skin_prices.csv --out-csv tradeup_scientific_latest.csv
if errorlevel 1 (
    echo ❌ 计算失败！
    pause
    exit /b 1
)
echo.

echo ================================================================================
echo ✅ 所有步骤完成！
echo ================================================================================
echo.
echo 生成的文件:
echo   - tradeup_scientific_latest.csv
echo   - 科学版_盈利TOP100_按利润.txt
echo   - 科学版_亏损TOP100_按利润.txt
echo   - 科学版_盈利TOP100_按ROI.txt
echo   - 科学版_亏损TOP100_按ROI.txt
echo.
echo 按任意键打开盈利TOP文件...
pause > nul
notepad 科学版_盈利TOP100_按利润.txt
