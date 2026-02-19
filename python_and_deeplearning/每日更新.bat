@echo off
chcp 65001 > nul
echo ================================================================================
echo CS2 Trade-Up 科学版计算 - 每日更新脚本
echo ================================================================================
echo.

set PYTHON=D:\acaconda\envs\dl\python.exe
set STEAMDT_API_KEY=ef24f95ea93b45a3b79c828687b85c4e

echo [步骤 1/4] 爬取最新价格数据（全平台完整数据）...
echo   -^> 从 STEAM/BUFF/YOUPIN/C5 等9个平台获取所有价格
%PYTHON% fetch_prices_with_steamdt.py --items-csv data\cs2_case_items_full.csv --out data\prices_all.csv --platform all
if errorlevel 1 (
    echo 价格爬取失败！
    pause
    exit /b 1
)
echo 价格数据已更新
echo.

echo [步骤 2/4] 统一价格格式并选择最低价...
echo   -^> 处理外观信息、保留平台来源、自动选择每个物品的最低价
%PYTHON% unify_prices.py
if errorlevel 1 (
    echo 格式统一失败！
    pause
    exit /b 1
)
echo 格式统一完成
echo.

echo [步骤 3/4] 运行科学版Plus计算（含缓存加速）...
echo   -^> 基于全平台最低价计算最优 Trade-Up 方案
echo   -^> 自动生成5份固定报告
%PYTHON% calculate_scientific_plus.py --input data\prices_with_exterior.csv --meta data\skins_meta_complete.csv --prices data\prices_with_exterior.csv --out-csv output\result.csv
if errorlevel 1 (
    echo 计算失败！
    pause
    exit /b 1
)
echo 科学版Plus计算完成
echo.

echo [步骤 4/4] 生成最佳主料清单（按箱子分类）...
echo   -^> 从科学版结果提取每个箱子的最佳材料组合
%PYTHON% tools\generate_best_materials_by_case.py
if errorlevel 1 (
    echo 清单生成失败！
    pause
    exit /b 1
)
echo 最佳主料清单生成完成
echo.

echo ================================================================================
echo 所有步骤完成！
echo ================================================================================
echo.
echo 生成的文件:
echo   [核心数据]
echo   - output\result.csv                  (完整数据)
echo.
echo   [固定报告 - 5份]
echo   - output\固定_每箱每物品_各外观_最优下级.txt
echo   - output\固定_炼金期望_正收益TOP100_按利润.txt
echo   - output\固定_炼金期望_负收益TOP100_按利润.txt
echo   - output\固定_炼金期望_正收益TOP100_按ROI.txt
echo   - output\固定_炼金期望_负收益TOP100_按ROI.txt
echo.
echo   [价格数据]
echo   - data\prices_all.csv              (全平台原始数据)
echo   - data\prices_all_min.csv          (每物品最低价)
echo   - data\prices_with_exterior.csv    (含外观处理)
echo.
echo   [最佳主料清单]
echo   - output\最佳主料清单_按箱子分类_完整版.csv
echo   - output\最佳主料清单_按箱子分类_完整版.md
echo.
echo 提示: 所有价格均为全平台最低价（BUFF/YOUPIN/C5/STEAM等）
echo 提示: 5份固定报告涵盖所有常用查询场景
echo 提示: 使用缓存加速，性能大幅提升
echo.
echo 按任意键打开盈利TOP报告...
pause > nul
notepad output\固定_炼金期望_正收益TOP100_按ROI.txt
