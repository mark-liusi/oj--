@echo off
chcp 65001 > nul
echo ================================================================================
echo 清理旧文件 - 保留必需文件
echo ================================================================================
echo.
echo 即将删除以下旧版本/中间文件:
echo.
echo 简单版输出:
echo   - tradeup_margins_*.csv/txt
echo   - winners_by_*.csv/txt
echo   - losers_by_*.csv/txt
echo   - tradeup_pots_new.txt
echo.
echo 旧版本科学版:
echo   - tradeup_scientific.csv
echo   - tradeup_scientific_v2.csv
echo   - tradeup_scientific_real.csv
echo.
echo 其他中间文件:
echo   - skins_meta_generated.csv
echo   - skins_meta.csv
echo   - cs2_cases.csv
echo.
echo 是否继续? (Y/N)
set /p choice=

if /i "%choice%" NEQ "Y" (
    echo 取消操作
    pause
    exit /b 0
)

echo.
echo 开始清理...

REM 删除简单版输出
del /q tradeup_margins_new.csv 2>nul
del /q tradeup_margins_formatted.txt 2>nul
del /q winners_by_margin.csv 2>nul
del /q winners_by_margin.txt 2>nul
del /q losers_by_margin.csv 2>nul
del /q losers_by_margin.txt 2>nul
del /q winners_by_roi.csv 2>nul
del /q winners_by_roi.txt 2>nul
del /q losers_by_roi.csv 2>nul
del /q losers_by_roi.txt 2>nul
del /q tradeup_pots_new.txt 2>nul

REM 删除旧版本
del /q tradeup_scientific.csv 2>nul
del /q tradeup_scientific_v2.csv 2>nul
del /q tradeup_scientific_real.csv 2>nul

REM 删除中间文件
del /q skins_meta_generated.csv 2>nul
del /q skins_meta.csv 2>nul
del /q cs2_cases.csv 2>nul

echo.
echo ✅ 清理完成！
echo.
echo 保留的必需文件:
echo   ✓ 核心脚本 (5个 .py)
echo   ✓ 基础数据 (cs2_case_items_full.csv, skins_meta_real.csv)
echo   ✓ 最新计算结果 (tradeup_scientific_latest.csv + 4个txt)
echo.
pause
