import json
import os
import time
from flask import Flask, render_template, request, jsonify
import shutil
from app import get_vc_fn, get_load_model_func
from app import vvoice, vc_transform

import logging
# log = logging.getLogger('werkzeug')
# log.disabled = True
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

output_file = None #具体输出文件
input_file = None # FileStorage类，传入的音源
voice = None # 音色选择的歌手名字（字符串）
voice_change = 0 # 变调数值
file_src = None # 输入音频的src

os.system("start http://127.0.0.1:5000")

@app.errorhandler(500)
def server_error(error):
    return "服务器内部发生错误",500

# 2024.02.24
# 这回大改了一下，每次传输数据并不重新加载页面
# 利用了ajax实现了前后端数据在不更新页面的情况下直接交互（可以理解为，这东西能直接把数据丢给后端）
# 要先下载jquery，如果直接下载我发的压缩包，里面应该是有的，就在static文件夹下
# 有一个jquery-3.5.1.min.js
# 也就是static中有两个文件夹用来存输入输出的文件，五个歌手的照片，一个前端音乐的符号图，一个背景图和一个jquery
# 如果没有的话，去网上找这个版本，下载之后塞到static文件夹下
# 在载入模型的时候加入了进度条，但是具体进度计算我还不会搞，所以这里只用了循环去测试进度条
# 推理那里还没加进度条，后续可以根据需要商量要不要加
# 注意flask要多导入一个jsonify


 # 2024.03.01更新内容总结如下：
 # 前端传输数据的方式统一为原生ajax，即不用jquery了，static内的jquery可以删掉了
 # 我之前傻乎乎地看错了，把进度条加在了模型载入的地方，应该加在推理的过程，现在改过来了
 # 代码里那个循环，sleep之类的，都是模拟情况的，拿来看效果的
 # 还有就是html页面的微调了，首先是页面调小到原本的90%左右
 # 模型载入之后会跳一个提示信息，提示信息会在1.5秒后自动消失，之后模型载入成功会弹出弹窗
 # 也就是我在html下面增加了一个函数showNotification（其实属于返场，我上上个版本的代码有用这个函数）
 # 我单纯感觉模型确定要有个成功的提示信息
 # 推理的时候会有进度条，至于进度计算就靠zhgg了，只要在底下show_progress方法里，给temp赋进度值就行，前端会自动去调取的
 # 前端成员音色名称改为Melody（member），但是在“当前模型”的文本后面仅显示Melody，而且后端传入的音色字符串为Melody
 # 还有就是把两个style写成了css，无伤大雅
 # 大体应该是这样，有什么问题可以直接问我————from cxx

 # 再次说一下接口，voice是音色字符串，voice_change是变调信息，input_file是输入的音源文件
 # 音源文件会存储在/static/input_voice下
 # 成果音源请存储在/statoc/output文件夹下（要确保就一个输出，前端会输出该文件夹下的文件）

@app.route('/')
def start():
    return render_template("startup.html")


@app.route('/startup.html')
def startup():
    return render_template("startup.html")

@app.route('/readme.html')
def readme():
    return render_template("readme.html")

@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/voice.html')
def voice():
    return render_template("voice.html")

@app.route('/video.html')
def video():
    return render_template("video.html")


# 载入文件（就是按下转换键之后执行的代码，同样前端会确保音色已选择，并且已经载入模型，有音源文件）
@app.route('/load_file', methods=['POST', 'GET'])
def load_file():
    # print(request.form)

    # 这段循环是拿来测试进度条的，临时看一下
    # 测试的代码很简单，就是用test_num的大小来代指已经运行了百分之多少了
    # 但是模型载入用什么来给出当前运行了多少，我还不知道

    # global test_num
    # test_num = 0
    # while test_num < 100:
    #     print(test_num)
    #     test_num += 1
    #     time.sleep(0.01)



    # 前端会传入变调信息，音源文件
    # 对应变量 voice_change，file

    global voice_change, input_file, file_src
    voice_change = request.form["voice_change"]
    input_file = request.files['input_files']  # 也是音源，保存输入罢了
    vc_transform.value = voice_change
    
    # 这里卡了我好几个小时，问就是我没想到这两个不在一起，一个是form，一个是files，悲
    # print(request.files)
    # print(voice_change)
    # print(input_file.filename)
    # 如果变调信息为空，默认是0
    if voice_change == "":
        voice_change = 0


    # 下面这段和之前的版本一样，同样为了测试，暂时利用了复制的方式，连接推理就麻烦zhgg了
    # 文件名不为空就存进路径
    if not input_file.filename == '':
        # 创建对应的路径，后端根据需要可以修改
        shutil.rmtree('./static/input_voice')
        # 清空路径内已有的东西
        os.mkdir('./static/input_voice')
        shutil.rmtree('./static/output')
        os.mkdir('./static/output')
        input_file.save("./static/input_voice/" + input_file.filename)  # save函数只能调用一次，所以后面是用的拷贝（论我不知道只能一次在这卡了半天）

        # 这里仅做测试，因此将输入的音频直接拷贝进了output文件夹，后期则是将转换后的文件传入文件夹
        # 即这里是进行推理，以及成品载入对应文件夹的地方！！！

        global input_file_flag
        input_file_flag = 1
        file_src = "./static/input_voice/" + input_file.filename  # 记录输入的音频的路径

        of = get_vc_fn(file_src)
        global target
        print(of)
        target = 100
        data = {
            "voice_change": voice_change,
            "output_file": of
        }
    # print(data["output_file"])
    # 要把输出转为json
    return json.dumps(data),201
    # 这里变了一点点，后面加上201，用于前端判断后端有没有执行完的


# 载入模型（按下确认模型后执行的代码，前端已确保有选择模型）
@app.route('/load_model', methods=['POST'])
def load_model():
    # print("loading")

    # 传入音色选择，前端已经确保不会空选
    # global test_num
    # test_num = 0
    # while test_num < 100:
    #     # print(test_num)
    #     test_num += 1
    #     time.sleep(0.1)
    #
    #

    # 模拟模型载入的时间
    # time.sleep(3)
    global voice
    voice = request.form["voice"]

    vvoice.value = voice
    # print(voice)


    # 载入模型后是否要将前一个模型的成果删除，以及输入的音源文件是否要保留
    # 这里进行模型载入！！！
    app.model_message, app.sid, app.cl_num, k_step = get_load_model_func()

    # 同样把音色返回回前端，用来更新“当前模型”那里的文字
    data ={
        "voice": voice
    }
    return jsonify(data),201

from inference.infer_tool_webui import InferJd
k = 1 ; target = 1 ; last_i = 0 ; jd = 0 ; temp = 0
# 这个是进度条定时访问的
# 进度条可以理解为，我设定他多久访问一次这个函数（我现在前端设定的是0.5秒访问一次这个函数）
# 这个函数会返回后端任务已经运行了百分之多少了（返回0-100的数字）
@app.route('/show_progress', methods=['POST', 'GET'])
def show_progress():

    # 返回当前进度给前端
    # 也就是说需要后端的友友，把当前进度的数据传给temp，数据会返回到前端形成进度条
    # 传的是0-100的数字，不是0-1
    global last_i, k, jd, temp, target
    if (last_i == 0):
        jd = 0

    if last_i == InferJd['i']:
        temp = ( (9*temp+jd)/10 )
    else:
        temp = jd
        jd = ( 100.0 * last_i / InferJd['len'])
        last_i = InferJd['i']

    if last_i == InferJd['len']:
        temp = 100
        last_i = 0
    data = {
        "time": temp
    }

    print(f"InferJd = {InferJd}, target={target},  \
          temp={temp}, last_i={last_i}, jd={jd}")
    # print(data)
    return jsonify(data), 201


if __name__ == '__main__':

    # 这里和之前一样，没变
    # 创建输入和输出文件所在文件夹
    if os.path.exists('./static/input_voice'):
        shutil.rmtree('./static/input_voice')
        os.mkdir('./static/input_voice')
    else:
        os.makedirs('./static/input_voice')
    if os.path.exists('./static/output'):
        shutil.rmtree('./static/output')
        os.mkdir('./static/output')
    else:
        os.makedirs("./static/output")

    app.run()


# http://127.0.0.1:5000