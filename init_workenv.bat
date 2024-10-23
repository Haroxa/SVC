chcp 65001
@echo off

set start_time=%time%
echo 开始时间: %start_time%

echo 正在创建虚拟环境...
env\python -m venv workenv

echo 正在进入虚拟环境...
call workenv\Scripts\activate.bat
python.exe -m pip install --upgrade pip

echo 正在安装依赖...
pip install -r requirements.txt

set end_time=%time%
echo 结束时间: %end_time%

python -c "from datetime import datetime as dt; start_time = dt.strptime('%start_time%', '%%H:%%M:%%S.%%f'); end_time = dt.strptime('%end_time%', '%%H:%%M:%%S.%%f'); time_diff = end_time - start_time; print('环境配置完成，耗时:', time_diff)"
deactivate