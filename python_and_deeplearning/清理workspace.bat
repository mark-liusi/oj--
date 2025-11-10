@echo off
chcp 65001 >nul
echo ======================================================
echo 清理非必要文件，只保留日常使用的核心文件
echo ======================================================
echo.

REM 需要保留的核心文件列表
echo 保留的核心文件:
echo   1. 数据文件 (3个)
echo      - cs2_case_items_full.csv (箱子物品清单)
echo      - skins_meta_complete.csv (浮漂区间数据)
echo      - prices_with_exterior.csv (价格数据)
echo.
echo   2. 核心程序 (3个)
echo      - fetch_prices_with_steamdt.py (抓取价格)
echo      - calculate_scientific.py (科学计算)
echo      - fetch_float_from_github.py (更新浮漂)
echo.
echo   3. 辅助工具 (2个)
echo      - update_manual_floats.py (更新手动浮漂)
echo      - unify_prices.py (统一价格格式)
echo.
echo   4. 配置文件 (2个)
echo      - 每日更新.bat (自动化脚本)
echo      - README_科学版流程.md (使用说明)
echo.
echo   5. 最新结果 (4个TXT)
echo      - 科学版_盈利TOP100_按利润.txt
echo      - 科学版_盈利TOP100_按ROI.txt
echo      - 科学版_亏损TOP100_按利润.txt
echo      - 科学版_亏损TOP100_按ROI.txt
echo.

echo ======================================================
echo 即将删除的临时文件类型:
echo ======================================================
echo   - 测试脚本 (check_*.py, verify_*.py等)
echo   - 备份文件 (*.backup, *_backup_*.csv)
echo   - 日志文件 (*_log.csv, failed_*.csv)
echo   - 中间数据 (new_prices_*.csv, estimated_*.csv等)
echo   - 旧版程序 (*_v2.py, *_v3.py等)
echo.

set /p confirm=确认删除? (输入 Y 继续): 
if /i not "%confirm%"=="Y" (
    echo 已取消
    pause
    exit /b
)

echo.
echo 开始清理...
echo.

REM 创建备份文件夹
if not exist "已清理备份" mkdir "已清理备份"

REM 删除测试和检查脚本
del /q check_*.py 2>nul
del /q verify_*.py 2>nul
del /q find_*.py 2>nul
del /q search_*.py 2>nul
del /q list_*.py 2>nul
del /q final_*.py 2>nul
del /q float_data_report.py 2>nul
echo [OK] 已删除测试脚本

REM 删除修正和导入的临时脚本
del /q fix_*.py 2>nul
del /q correct_*.py 2>nul
del /q import_manual_floats.py 2>nul
echo [OK] 已删除临时修正脚本

REM 删除旧版本脚本
del /q fetch_prices_remaining8*.py 2>nul
del /q fetch_corrected_prices.py 2>nul
del /q extract_exterior_from_prices.py 2>nul
del /q fetch_float_from_wiki.py 2>nul
del /q fetch_float_ranges.py 2>nul
del /q prepare_scientific_inputs.py 2>nul
echo [OK] 已删除旧版脚本

REM 删除备份文件
del /q *.backup 2>nul
del /q *_backup_*.csv 2>nul
echo [OK] 已删除备份文件

REM 删除日志文件
del /q *_log.csv 2>nul
del /q failed_*.csv 2>nul
del /q item_name_corrections.csv 2>nul
del /q name_corrections*.csv 2>nul
del /q manual_float_update_log.csv 2>nul
echo [OK] 已删除日志文件

REM 删除临时数据文件
del /q new_prices_*.csv 2>nul
del /q estimated_*.csv 2>nul
del /q prices_all_min_new.csv 2>nul
del /q prices_with_inspect.csv 2>nul
del /q skins_meta_real.csv 2>nul
del /q skin_prices.csv 2>nul
del /q manual_float_lookup.csv 2>nul
echo [OK] 已删除临时数据

REM 删除旧的结果文件
del /q tradeup_scientific_latest.csv 2>nul
echo [OK] 已删除旧结果文件

REM 删除旧的计算程序
del /q calculate.py 2>nul
del /q 2.py 2>nul
echo [OK] 已删除旧版计算程序

REM 删除旧的文档
del /q 使用说明.md 2>nul
del /q 未找到浮漂的物品清单.md 2>nul
del /q 清理旧文件.bat 2>nul
echo [OK] 已删除旧文档

echo.
echo ======================================================
echo 清理完成!
echo ======================================================
echo.
echo 保留的核心文件:
dir /b cs2_case_items_full.csv 2>nul
dir /b skins_meta_complete.csv 2>nul
dir /b prices_with_exterior.csv 2>nul
dir /b prices_all_min.csv 2>nul
dir /b manual_float_data.csv 2>nul
dir /b fetch_prices_with_steamdt.py 2>nul
dir /b calculate_scientific.py 2>nul
dir /b fetch_float_from_github.py 2>nul
dir /b update_manual_floats.py 2>nul
dir /b unify_prices.py 2>nul
dir /b 每日更新.bat 2>nul
dir /b README_科学版流程.md 2>nul
dir /b 科学版_*.txt 2>nul
dir /b 科学版_*.csv 2>nul

echo.
pause
