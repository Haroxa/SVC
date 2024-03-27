chcp 65001
@echo off

REM 定义requirements.txt文件路径
set REQUIREMENTS=requirements.txt

REM 检查requirements.txt文件是否存在
if not exist %REQUIREMENTS% (
    echo requirements.txt 文件不存在。
    exit /b 1
)

REM 读取requirements.txt文件中的所有库名称
for /f "tokens=1,* delims==" %%i in (%REQUIREMENTS%) do (
	REM 检查当前库是否已安装
    pip show %%i >nul 2>&1
    if errorlevel 1 (
        REM 如果未安装，则使用pip安装该库
        echo 安装 %%i...
        pip install %%i
    ) else (
        echo %%i 已安装。
    )
)

echo 所有库都已安装。
