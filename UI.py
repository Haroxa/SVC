import os
import json
from flask import Flask, render_template, request, jsonify
import shutil
# from app import vvoice, sid, cl_num, k_step, load_model_func, cluster_choice, \
#     enhance, diff_choice, diff_config_choice, only_diffusion, using_device, diffusion_method, diffusion_speedup, \
#     use_spk_mix, vocoder_choice, get_choice_ckpt, get_config_choice, vc_transform, vc_input3, vc_fn, \
#     output_format, auto_f0, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cluster_ratio, slice_db, noise_scale, \
#     pad_seconds, cr_threshold, \
#     second_encoding, loudness_envelope_adjustment
from app import get_vc_fn, get_load_model_func
from app import vvoice, vc_transform

# from inference.infer_tool_webui import get_rate

app = Flask(__name__)
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

output_file = None  # 具体输出文件
file = None  # FileStorage类，传入的音源
voice = None  # 音色选择的歌手名字（字符串）
voice_change = 0  # 变调数值
file_src = None  # 输入音频的src

test_num = 0

os.system("start http://127.0.0.1:5000")
target = 99

@app.errorhandler(500)
def server_error(error):
    return "服务器内部发生错误",500

@app.route('/')
def index():
    '''output_file = os.listdir(".\static\output") # 输出的音频src（因为一开始不确定能不能多文件输入，所以这里是直接读取目录下所有文件）
    # print(output_file)
    # print(file)
    # print(voice)
    # print(voice_change)
    # print(file_src)
    return render_template("index.html",output_file=output_file, file_src=file_src, voice=voice, voice_change=voice_change, flag=flag, input_file_flag=input_file_flag)
    # 传入参数，用于初始化界面
    # 传入output_file用于前端展示输出音频
    # 传入file_src用于提交后的页面保留传入的音频播放，仅会保留音频播放器播放的音频，如果没有重新上传文件，再次提交表单实际上不会再次提交文件
    # 传入voice用于提交后页面保留音色的选择
    # 传入voice_change用于提交后页面保留变调数值
    # 传入flag用于判断是第一次进入该页面，还是点击过转换按钮，用来看是否要跳出提示信息
    # 传入input_file_flag用于在点击过转换按钮后，判断有没有上传音源文件
    '''

    return render_template("index.html")


@app.route('/load_file', methods=['POST'])
def load_file():
    # 前端会传入选择的音色，变调信息，音源文件
    # 对应变量 voice，voice_change，file

    global voice, voice_change, file, file_src
    # voice = request.form.get("voice")  # 获取前端传入表单的音色选择
    voice_change = request.form["voice_change"]  # 获取表单的变调信息
    input_file = request.files["input_files"]  # 获取传入的音源，file是FileStorage类
    vc_transform.value = voice_change

    # 如果变调信息为空，默认是0
    if voice_change == "":
        voice_change = 0

    # input_file = request.files['file']  # 也是音源，保存输入罢了

    # 每次提交表单都会重新刷新页面
    # 相当于初始化整个页面
    # 因此传入一些参数，比如音色选择，变调信息之类的，都是刷新后用来初始化页面的
    # 保证用户输入的信息保持不变
    # 前端加入了表单要素的判断，因此初次传入的信息必定包含选择的音色，变调信息，音源文件
    # 这里说初次是因为，若连续提交了两次表单，第二次没有重新选择音源文件，提交的表单实际上是没有音源文件的
    # 因此下面加了特判，如果有更新音源文件，才进行下面的操作
    # 音色选择和变调信息是必然会传入的
    # 因此上面直接读取了

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
        return json.dumps(data), 201


# 载入模型
@app.route('/load_model', methods=['POST'])
def load_model():
    # time.sleep(3) # 也是拿来测试模拟的罢了

    # 传入音色选择，前端已经确保不会空选
    global voice
    voice = request.form["voice"]

    vvoice.value = voice

    # 载入模型后是否要将前一个模型的成果删除，以及输入的音源文件是否要保留
    # 这里进行模型载入！！！
    app.model_message, app.sid, app.cl_num, k_step = get_load_model_func()

    data = {
        "voice": voice
    }

    return jsonify(data), 201


k = 0


@app.route('/show_progress', methods=['POST', 'GET'])
def show_progress():
    # 返回当前进度给前端
    # 也就是说需要后端的友友，把当前进度的数据传给temp，数据会返回到前端形成进度条
    # 传的是0-100的数字，不是0-1
    global test_num, k
    test_num = k
    k += 1
    if (k == 100):
        k = 99
    global target
    if (target == 100):
        test_num = 100
    temp = test_num
    #print(temp)
    data = {
        "time": temp
    }
    if(target==100):
        k=0
    target=99
    # print(data)
    return jsonify(data), 201


if __name__ == '__main__':

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
