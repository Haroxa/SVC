### 项目环境库配置
依据当前使用python的版本，创建对应版本的虚拟环境
```cmd
python -m venv workenv
```
在项目目录中激活虚拟环境
```cmd
workenv\Scripts\activate
```
下载依赖库
```cmd
check_requirements.bat
```

### 项目打包
安装相关库
```cmd
pip install pyinstaller
```
对入口程序进行打包
```cmd
pyinstaller setup.py
```
> 打包好后会出现 build 和 dist 目录，如果直接在dist目录中运行会出现路径错误，需要把对应文件移动到项目文件夹下即可正常运行 \
参考链接：https://zhuanlan.zhihu.com/p/398619997

### 模型压缩
原项目地址：https://github.com/svc-develop-team/so-vits-svc
```python
# 本项目不支持，请在原项目中使用
name = ["Adele","Justin Bieber","My","Taylor Swift","Trump"]
for n in name:
	print(n)
	# -c 配置文件路径 -i 模型路径 -o 输出路径
	os.system(f"python compress_model.py -c=\"configs/{n}.json\" -i=\"logs/44k/{n}.pth\" -o=\"logs/44k/{n}_release.pth\"  ")

```

### 模型转换
```python
# 使用原项目 app.py 中的 onnx_export_func 方法
# 在项目根目录下面创建相关目录，如下所示
# checkpoints
# |--My
# |--|--My.json
# |--|--My.pth
# ...
from app import onnx_export_func
onnx_export_func()
```

### 注意事项
由于 pretrain、logs、ffmpeg 文件夹过大，无法直接上传到 github，暂时采用网盘的方式存储，网盘链接：https://pan.baidu.com/s/1Yo_Z_4SHZtUlQOu3aOiy2Q?pwd=bw34