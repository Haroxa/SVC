import ast
import glob
import json
import logging
import os
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

# from fap.cli.length import length
# from fap.utils.file import AUDIO_EXTENSIONS, list_files
# from fap.utils.slice_audio_v2 import slice_audio_file_v2
from inference.infer_tool_webui import Svc
# from onnx_export import main as onnx_export
# from utils import mix_model

os.environ["PATH"] += os.pathsep + os.path.join(os.getcwd(), "ffmpeg", "bin")

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Some directories
workdir = "logs/44k"
second_dir = "models"
diff_second_dir = "models/diffusion"
diff_workdir = "logs/44k/diffusion"
config_dir = "configs/"
dataset_dir = "dataset/44k"
raw_path = "dataset_raw"
raw_wavs_path = "raw"
models_backup_path = 'models_backup'
root_dir = "checkpoints"
default_settings_file = "settings.yaml"
current_mode = ""
# Some global variables
is_onnx = True
debug = False
precheck_ok = False
model = None
sovits_params = {}
diff_params = {}
# Some dicts for mapping
MODEL_TYPE = {
    "vec768l12": 768,
    "vec256l9": 256,
    "hubertsoft": 256,
    "whisper-ppg": 1024,
    "cnhubertlarge": 1024,
    "dphubert": 768,
    "wavlmbase+": 768,
    "whisper-ppg-large": 1280
}
ENCODER_PRETRAIN = {
    "vec256l9": "pretrain/checkpoint_best_legacy_500.pt",
    "vec768l12": "pretrain/checkpoint_best_legacy_500.pt",
    "hubertsoft": "pretrain/hubert-soft-0d54a1f4.pt",
    "whisper-ppg": "pretrain/medium.pt",
    "cnhubertlarge": "pretrain/chinese-hubert-large-fairseq-ckpt.pt",
    "dphubert": "pretrain/DPHuBERT-sp0.75.pth",
    "wavlmbase+": "pretrain/WavLM-Base+.pt",
    "whisper-ppg-large": "pretrain/large-v2.pt"
}
css = """
#warning {background-color: 	#800080}
.feedback textarea {font-size: 24px !important}
"""


class Config:
    def __init__(self, path, type):
        self.path = path
        self.type = type

    def read(self):
        if self.type == "json":
            with open(self.path, 'r') as f:
                return json.load(f)
        if self.type == "yaml":
            with open(self.path, 'r') as f:
                return yaml.safe_load(f)

    def save(self, content):
        if self.type == "json":
            with open(self.path, 'w') as f:
                json.dump(content, f, indent=4)
        if self.type == "yaml":
            with open(self.path, 'w') as f:
                yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)


class ReleasePacker:
    def __init__(self, speaker, model):
        self.speaker = speaker
        self.model = model
        self.output_path = os.path.join("release_packs", f"{speaker}_release.zip")
        self.file_list = []

    def remove_temp(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and not filename.endswith(".zip"):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)

    def add_file(self, file_paths):
        self.file_list.extend(file_paths)

    def spk_to_dict(self):
        spk_string = self.speaker.replace('，', ',')
        spk_string = spk_string.replace(' ', '')
        _spk = spk_string.split(',')
        return {_spk: index for index, _spk in enumerate(_spk)}

    def generate_config(self, diff_model, config_origin):
        _config_origin = Config(os.path.join(config_read_dir, config_origin), "json")
        _template = Config("release_packs/config_template.json", "json")
        _d_template = Config("release_packs/diffusion_template.yaml", "yaml")
        orig_config = _config_origin.read()
        config_template = _template.read()
        diff_config_template = _d_template.read()
        spk_dict = self.spk_to_dict()
        _net = torch.load(os.path.join(ckpt_read_dir, self.model), map_location='cpu')
        emb_dim, model_dim = _net['model'].get('emb_g.weight', torch.empty(0, 0)).size()
        vol_emb = _net['model'].get('emb_vol.weight')
        if vol_emb is not None:
            config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True
        # Keep the spk_dict length same as emb_dim
        if emb_dim > len(spk_dict):
            for i in range(emb_dim - len(spk_dict)):
                spk_dict[f"spk{i}"] = len(spk_dict)
        if emb_dim < len(spk_dict):
            for i in range(len(spk_dict) - emb_dim):
                spk_dict.popitem()
        self.speaker = ','.join(spk_dict.keys())
        config_template['model']['ssl_dim'] = config_template["model"]["filter_channels"] = config_template["model"][
            "gin_channels"] = model_dim
        config_template['model']['n_speakers'] = diff_config_template['model']['n_spk'] = emb_dim
        config_template['spk'] = diff_config_template['spk'] = spk_dict
        encoder = [k for k, v in MODEL_TYPE.items() if v == model_dim]
        if orig_config['model']['speech_encoder'] in encoder:
            config_template['model']['speech_encoder'] = orig_config['model']['speech_encoder']
        else:
            raise Exception("Config is not compatible with the model")

        if diff_model != "no_diff":
            _diff = torch.load(os.path.join(diff_read_dir, diff_model), map_location='cpu')
            _, diff_dim = _diff["model"].get("unit_embed.weight", torch.empty(0, 0)).size()
            if diff_dim == 256:
                diff_config_template['data']['encoder'] = 'hubertsoft'
                diff_config_template['data']['encoder_out_channels'] = 256
            elif diff_dim == 768:
                diff_config_template['data']['encoder'] = 'vec768l12'
                diff_config_template['data']['encoder_out_channels'] = 768
            elif diff_dim == 1024:
                diff_config_template['data']['encoder'] = 'whisper-ppg'
                diff_config_template['data']['encoder_out_channels'] = 1024

        with open("release_packs/install.txt", 'w') as f:
            f.write(str(self.file_list) + '#' + str(self.speaker))

        _template.save(config_template)
        _d_template.save(diff_config_template)

    def unpack(self, zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall("release_packs")

    def formatted_install(self, install_txt):
        with open(install_txt, 'r') as f:
            content = f.read()
        file_list, speaker = content.split('#')
        self.speaker = speaker
        file_list = ast.literal_eval(file_list)
        self.file_list = file_list
        for _, target_path in self.file_list:
            if target_path != "install.txt" and target_path != "":
                shutil.move(os.path.join("release_packs", target_path), target_path)
        self.remove_temp("release_packs")
        return self.speaker

    def pack(self):
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, target_path in self.file_list:
                if os.path.isfile(file_path):
                    zipf.write(file_path, arcname=target_path)

'''
def debug_change():
    global debug
    debug = debug_button.value
'''

def get_default_settings():
    global sovits_params, diff_params, second_dir_enable
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    sovits_params = default_settings['sovits_params']
    diff_params = default_settings['diff_params']
    webui_settings = default_settings['webui_settings']
    second_dir_enable = webui_settings['second_dir']
    sami_settings = default_settings['sami_settings']
    return sovits_params, diff_params, second_dir_enable, sami_settings


def webui_change(read_second_dir):
    global second_dir_enable
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    second_dir_enable = default_settings['webui_settings']['second_dir'] = read_second_dir
    config_file.save(default_settings)


def get_current_mode():
    global current_mode
    current_mode = "当前模式：独立目录模式，将从'./models/'读取模型文件" if second_dir_enable else "当前模式：工作目录模式，将从'./logs/44k'读取模型文件"
    return current_mode


def save_default_settings(log_interval, eval_interval, keep_ckpts, batch_size, learning_rate, amp_dtype, all_in_mem,
                          num_workers, cache_all_data, cache_device, diff_amp_dtype, diff_batch_size, diff_lr,
                          diff_interval_log, diff_interval_val, diff_force_save, diff_k_step_max):
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    default_settings['sovits_params']['log_interval'] = int(log_interval)
    default_settings['sovits_params']['eval_interval'] = int(eval_interval)
    default_settings['sovits_params']['keep_ckpts'] = int(keep_ckpts)
    default_settings['sovits_params']['batch_size'] = int(batch_size)
    default_settings['sovits_params']['learning_rate'] = float(learning_rate)
    default_settings['sovits_params']['amp_dtype'] = str(amp_dtype)
    default_settings['sovits_params']['all_in_mem'] = all_in_mem
    default_settings['diff_params']['num_workers'] = int(num_workers)
    default_settings['diff_params']['cache_all_data'] = cache_all_data
    default_settings['diff_params']['cache_device'] = str(cache_device)
    default_settings['diff_params']['amp_dtype'] = str(diff_amp_dtype)
    default_settings['diff_params']['diff_batch_size'] = int(diff_batch_size)
    default_settings['diff_params']['diff_lr'] = float(diff_lr)
    default_settings['diff_params']['diff_interval_log'] = int(diff_interval_log)
    default_settings['diff_params']['diff_interval_val'] = int(diff_interval_val)
    default_settings['diff_params']['diff_force_save'] = int(diff_force_save)
    default_settings['diff_params']['diff_k_step_max'] = diff_k_step_max
    config_file.save(default_settings)
    return "成功保存默认配置"


def get_model_info(choice_ckpt):
    pthfile = os.path.join(ckpt_read_dir, choice_ckpt)
    net = torch.load(pthfile, map_location=torch.device('cpu'))  # cpu load to avoid using gpu memory
    spk_emb = net["model"].get("emb_g.weight")
    if spk_emb is None:
        return "所选模型缺少emb_g.weight，你可能选择了一个底模"
    _layer = spk_emb.size(1)
    encoder = [k for k, v in MODEL_TYPE.items() if v == _layer]  # 通过维度对应编码器
    encoder.sort()
    if encoder == ["hubertsoft", "vec256l9"]:
        encoder = ["vec256l9 / hubertsoft"]
    if encoder == ["cnhubertlarge", "whisper-ppg"]:
        encoder = ["whisper-ppg / cnhubertlarge"]
    if encoder == ["dphubert", "vec768l12", "wavlmbase+"]:
        encoder = ["vec768l12 / dphubert / wavlmbase+"]
    return encoder[0]


def load_json_encoder(config_choice, choice_ckpt):
    if config_choice == "no_config":
        return "未启用自动加载，请手动选择配置文件"
    if choice_ckpt == "no_model":
        return "请先选择模型"
    config_file = Config(os.path.join(config_read_dir, config_choice), "json")
    config = config_file.read()
    try:
        # 比对配置文件中的模型维度与该encoder的实际维度是否对应，防止古神语
        config_encoder = config["model"].get("speech_encoder", "no_encoder")
        config_dim = config["model"]["ssl_dim"]
        # 旧版配置文件自动匹配
        if config_encoder == "no_encoder":
            config_encoder = config["model"]["speech_encoder"] = "vec256l9" if config_dim == 256 else "vec768l12"
            config_file.save(config)
        correct_dim = MODEL_TYPE.get(config_encoder, "unknown")
        if config_dim != correct_dim:
            return "配置文件中的编码器与模型维度不匹配"
        return config_encoder
    except Exception as e:
        return f"出错了: {e}"


def auto_load(choice_ckpt):
    global second_dir_enable
    model_output_msg = get_model_info(choice_ckpt)
    json_output_msg = config_choice = ""
    choice_ckpt_name, _ = os.path.splitext(choice_ckpt)
    if second_dir_enable:
        all_config = [json for json in os.listdir(second_dir) if json.endswith(".json")]
        for config in all_config:
            config_fname, _ = os.path.splitext(config)
            if config_fname == choice_ckpt_name:
                config_choice = config
                json_output_msg = load_json_encoder(config, choice_ckpt)
        if json_output_msg != "":
            return model_output_msg, config_choice, json_output_msg
        else:
            return model_output_msg, "no_config", ""
    else:
        return model_output_msg, "no_config", ""


def auto_load_diff(diff_model):
    global second_dir_enable
    if second_dir_enable is False:
        return "no_diff_config"
    all_diff_config = [yaml for yaml in os.listdir(second_dir) if yaml.endswith(".yaml")]
    for config in all_diff_config:
        config_fname, _ = os.path.splitext(config)
        diff_fname, _ = os.path.splitext(diff_model)
        if config_fname == diff_fname:
            return config
    return "no_diff_config"


def load_model_func(ckpt_name, cluster_name, config_name, enhance, diff_model_name, diff_config_name, only_diffusion,
                    use_spk_mix, using_device, method, speedup, cl_num, vocoder_name):
    global model
    ckpt_name = get_choice_ckpt()
    config_name = get_config_choice()
    using_device = get_using_device()
    cluster_name = get_cluster_choice()
    enhance = get_enhance()
    diff_model_name = get_diff_choice()             # no-diff
    diff_config_name =get_diff_config_choice()      
    only_diffusion = get_only_diffusion()           # False
    use_spk_mix = get_use_spk_mix()
    method = get_diffusion_method()
    speedup = get_diffusion_speedup()
    cl_num = get_cl_num()
    vocoder_name = get_vocoder_choice()
    config_path = os.path.join(config_read_dir, config_name) if not only_diffusion else "configs/config.json"
    diff_config_path = os.path.join(config_read_dir,
                                    diff_config_name.value) if diff_config_name != "no_diff_config" else "configs/diffusion.yaml"
    ckpt_path = os.path.join(ckpt_read_dir, ckpt_name)
    cluster_path = os.path.join(ckpt_read_dir, cluster_name)
    diff_model_path = os.path.join(diff_read_dir, diff_model_name)
    k_step_max = 1000

    if not only_diffusion:
        config = Config(config_path, "json").read()
    if diff_model_name != "no_diff":
        _diff = Config(diff_config_path, "yaml")
        _content = _diff.read()
        diff_spk = _content.get('spk', {})
        diff_spk_choice = spk_choice = next(iter(diff_spk), "未检测到音色")
        if not only_diffusion:
            if _content['data'].get('encoder_out_channels') != config["model"].get('ssl_dim'):
                # return "扩散模型维度与主模型不匹配，请确保两个模型使用的是同一个编码器", gr.Dropdown.update(choices=[],value=""), 0, None
                return "扩散模型维度与主模型不匹配，请确保两个模型使用的是同一个编码器", gr.Dropdown(choices=[],value="",interactive=True), 0, None
        _content["infer"]["speedup"] = int(speedup)
        _content["infer"]["method"] = str(method)
        _content["vocoder"]["ckpt"] = f"pretrain/{vocoder_name}/model"
        k_step_max = _content["model"].get('k_step_max', 0) if _content["model"].get('k_step_max', 0) != 0 else 1000
        _diff.save(_content)

    if not only_diffusion:
        net = torch.load(ckpt_path, map_location=torch.device('cpu'))
        # 读取模型各维度并比对，还有小可爱无视提示硬要加载底模的就返回个未初始张量

        emb_dim, model_dim = net["model"].get("emb_g.weight", torch.empty(0, 0)).size()
        '''
        if emb_dim > config["model"]["n_speakers"]:
            return "模型说话人数量与emb维度不匹配", gr.Dropdown.update(choices=[], value=""), 0, None
            '''

        if model_dim != config["model"]["ssl_dim"]:
            # return "配置文件与模型不匹配", gr.Dropdown.update(choices=[], value=""), 0, None
            return "配置文件与模型不匹配", gr.Dropdown(choices=[], value="",interactive=True), 0, None

        #encoder = config["model"]["speech_encoder"]
        spk_dict = config.get('spk', {})
        spk_choice = next(iter(spk_dict), "未检测到音色")
    else:

        spk_dict = diff_spk
        spk_choice = diff_spk_choice
    fr = cluster_name.endswith(".pkl")  # 如果是pkl后缀就启用特征检索
    shallow_diffusion = diff_model_name != "no_diff"  # 加载了扩散模型就启用浅扩散
    device = cuda[using_device] if "CUDA" in using_device else using_device

    model = Svc(ckpt_path,
                config_path,
                device=device if device != "Auto" else None,
                cluster_model_path=cluster_path,
                nsf_hifigan_enhance=enhance,
                diffusion_model_path=diff_model_path,
                diffusion_config_path=diff_config_path,
                shallow_diffusion=shallow_diffusion,        # False
                only_diffusion=only_diffusion,              # False
                spk_mix_enable=use_spk_mix,
                feature_retrieval=fr)
    spk_list = list(spk_dict.keys())
    if enhance:
        from modules.enhancer import Enhancer
        model.enhancer = Enhancer('nsf-hifigan', f'pretrain/{vocoder_name}/model', device=model.dev)
    if not only_diffusion:
        clip = 25 #if encoder == "whisper-ppg" or encoder == "whisper-ppg-large" else cl_num  # Whisper必须强制切片25秒
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        sovits_msg = f"模型被成功加载到了{device_name}上\n"
    else:
        clip = cl_num
        sovits_msg = "启用全扩散推理，未加载So-VITS模型\n"


    index_or_kmeans = "特征索引" if fr else "聚类模型"
    clu_load = "未加载" if cluster_name == "no_clu" else cluster_name
    diff_load = "未加载" if diff_model_name == "no_diff" else f"{diff_model_name} | 采样器: {method} | 加速倍数：{int(speedup)} | 最大浅扩散步数：{k_step_max} | 声码器： {vocoder_name}"
    output_msg = f"{sovits_msg}{index_or_kmeans}：{clu_load}\n扩散模型：{diff_load}"
    

    return (
        output_msg,
        # gr.Dropdown.update(choices=spk_list, value=spk_choice),
        gr.Dropdown(choices=spk_list, value=spk_choice,interactive=True),
        clip,
        # gr.Slider.update(value=100 if k_step_max > 100 else k_step_max, minimum=speedup, maximum=k_step_max)
        gr.Slider(value=100 if k_step_max > 100 else k_step_max, minimum=speedup, maximum=k_step_max, interactive=True)
    )


def get_load_model_func():
    choice_ckpt = get_choice_ckpt()
    config_choice = get_config_choice()
    global cluster_choice, enhance, diff_choice, diff_config_choice,            \
                only_diffusion, use_spk_mix, using_device, diffusion_method,    \
                diffusion_speedup, cl_num, vocoder_choice                       

    return load_model_func(choice_ckpt, cluster_choice, config_choice,
                            enhance, diff_choice, diff_config_choice,
                            only_diffusion, use_spk_mix, using_device,
                            diffusion_method, diffusion_speedup, cl_num,
                            vocoder_choice)


def model_empty_cache():
    global model
    if model is None:
        return sid.update(choices=[], value=""), "没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices=[], value=""), "模型卸载完毕!"


def get_file_options(directory, extension):
    return [file for file in os.listdir(directory) if file.endswith(extension)]


def load_options():
    ckpt_list = [file for file in get_file_options(ckpt_read_dir, ".pth") if
                 not file.startswith("D_") or file == "G_0.pth"]
    config_list = get_file_options(config_read_dir, ".json")
    cluster_list = ["no_clu"] + get_file_options(ckpt_read_dir, ".pt") + get_file_options(ckpt_read_dir,
                                                                                          ".pkl")  # 聚类和特征检索模型
    diff_list = ["no_diff"] + get_file_options(diff_read_dir, ".pt")
    diff_config_list = ["no_diff_config"] + get_file_options(config_read_dir, ".yaml")
    return ckpt_list, config_list, cluster_list, diff_list, diff_config_list


def vc_infer(output_format, sid, input_audio, sr, input_audio_path, vc_transform, auto_f0, cluster_ratio, slice_db,
             noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold,
             k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    print("下方即可查看您的推理进度：")
    if np.issubdtype(input_audio.dtype, np.integer):
        input_audio = (input_audio / np.iinfo(input_audio.dtype).max).astype(np.float32)
    if len(input_audio.shape) > 1:
        input_audio = librosa.to_mono(input_audio.transpose(1, 0))
    if sr != 44100:
        input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=44100)
    sf.write("temp.wav", input_audio, 44100, format="wav")
    _audio = model.slice_inference(
        "temp.wav",
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )
    model.clear_empty()
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits_"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff_"
    if model.only_diffusion:
        isdiffusion = "diff_"
    # Gradio上传的filepath因为未知原因会有一个无意义的固定后缀，这里去掉
    truncated_basename = Path(input_audio_path).stem[:-6] if Path(input_audio_path).stem[-6:] == "-0-100" else Path(
        input_audio_path).stem
    output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}{f0_predictor}.{output_format}'
    output_file_path = os.path.join("static\\output", output_file_name)
    if os.path.exists(output_file_path):
        count = 1
        while os.path.exists(output_file_path):
            output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}{f0_predictor}_{str(count)}.{output_format}'
            output_file_path = os.path.join("static\\output", output_file_name)
            count += 1

    sf.write(output_file_path, _audio, model.target_sample, format=output_format)
    return output_file_path


def vc_fn(output_format, sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
          cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
          second_encoding, loudness_envelope_adjustment, progress=gr.Progress(track_tqdm=True)):
    global model
    output_format = get_output_format()
    ckpt_name = get_choice_ckpt()
    config_name = get_config_choice()
    using_device = get_using_device()
    cluster_name = get_cluster_choice()
    enhance = get_enhance()
    diff_model_name = get_diff_choice()
    diff_config_name = get_diff_config_choice()
    only_diffusion = get_only_diffusion()
    use_spk_mix = get_use_spk_mix()
    method = get_diffusion_method()
    speedup = get_diffusion_speedup()
    cl_num = get_cl_num()
    vocoder_name = get_vocoder_choice()

    sid = get_sid()

    try:
        if input_audio is None:
            print("你还没有上传音频")
            return "你还没有上传音频", None
        if model is None:
            print("你还没有加载模型")
            return "你还没有加载模型", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                print("")
        audio, sr = sf.read(input_audio)

        output_file_path = vc_infer(output_format, sid, audio, sr, input_audio, vc_transform, auto_f0, cluster_ratio,
                                    slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor,
                                    enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding,
                                    loudness_envelope_adjustment)
        os.remove("temp.wav")
        return output_file_path
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def get_vc_fn(file_src):
    global vc_input3
    vc_input3.value = file_src
    global output_format, sid, vc_transform, auto_f0, cluster_ratio, \
                   slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, \
                   enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, \
                   second_encoding, loudness_envelope_adjustment
    return vc_fn(output_format, sid, file_src, vc_transform, auto_f0, cluster_ratio, 
                   slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, 
                   enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
                   second_encoding, loudness_envelope_adjustment, progress=gr.Progress(track_tqdm=True))

def get_choice_ckpt():
    global choice_ckpt

    if vvoice.value == "Adele":
        choice_ckpt = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Adele.pth", visible=False)
    elif vvoice.value == "Justin Bieber":
        choice_ckpt = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Justin Bieber.pth", visible=False)
    elif vvoice.value == "Taylor Swift":
        choice_ckpt = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Taylor Swift.pth", visible=False)
    elif vvoice.value == "Trump":
        choice_ckpt = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Trump.pth", visible=False)
    else:
        choice_ckpt = gr.Dropdown(label="音色选择", choices=ckpt_list, value="My.pth", visible=False)

    return choice_ckpt.value


def get_model_branch():
    if vvoice.value == "Justin Bieber":
        model_branch = gr.Textbox(label="模型编码器", placeholder="根据模型自动选择", interactive=False,
                                  value="vec256l9 / hubertsoft",
                                  visible=False)
    else:
        model_branch = gr.Textbox(label="模型编码器", placeholder="根据模型自动选择", interactive=False,
                                  value="vec768l12 / dphubert / wavlmbase+",
                                  visible=False)
    return model_branch.value


def get_config_choice():

    global config_choice
    if vvoice.value == "Adele":
        config_choice = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Adele.json", visible=False)
    elif vvoice.value == "Justin Bieber":
        config_choice = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Justin Bieber.json", visible=False)
    elif vvoice.value == "Taylor Swift":
        config_choice = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Taylor Swift.json", visible=False)
    elif vvoice.value == "Trump":
        config_choice = gr.Dropdown(label="音色选择", choices=ckpt_list, value="Trump.json", visible=False)
    else:
        config_choice = gr.Dropdown(label="音色选择", choices=ckpt_list, value="My.json", visible=False)

    return config_choice.value


def get_config_info():
    if vvoice.value != "Justin Bieber":
        config_info = gr.Textbox(label="配置文件编码器", placeholder="根据配置文件自动选择", value="vec768l12",
                                 visible=False)
    else:
        config_info = gr.Textbox(label="配置文件编码器", placeholder="根据配置文件自动选择", value="vec256l9",
                                 visible=False)
    return config_info.value


def get_diff_choice():
    diff_choice = gr.Dropdown(value="no_diff", interactive=True, visible=False)
    return diff_choice.value


def get_diff_config_choice():
    diff_config_choice = gr.Dropdown(value="no_diff_config", interactive=True, visible=False)
    return diff_config_choice.value


def get_cluster_choice():
    cluster_choice = gr.Dropdown(value="no_clu", visible=False)
    return cluster_choice.value


def get_vocoder_choice():
    vocoder_choice = gr.Dropdown(value="nsf_hifigan", visible=False)
    return vocoder_choice.value


def get_enhance():
    enhance = gr.Checkbox(
        label="是否使用NSF_HIFIGAN增强，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭",
        value=False, visible=False)
    return enhance.value


def get_only_diffusion():
    only_diffusion = gr.Checkbox(
        label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，不建议使用",
        value=False, visible=False)
    return only_diffusion.value


def get_diffusion_method():
    diffusion_method = gr.Dropdown(label="扩散模型采样器",
                                   choices=["dpm-solver++", "dpm-solver", "pndm", "ddim", "unipc"],
                                   value="dpm-solver++", visible=False)
    return diffusion_method.value


def get_diffusion_speedup():
    diffusion_speedup = gr.Number(label="扩散加速倍数，默认为10倍", value=10, visible=False)
    return diffusion_speedup.value


def get_using_device():
    using_device = gr.Dropdown(label="推理设备，可使用CPU/GPU进行推理，默认使用最优设备",
                               choices=[*cuda.keys(), "cpu"],
                               value="请选择你的推理设备", visible=False)
    return using_device.value


def get_noise_scale():
    noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4, visible=False)
    return noise_scale.value


def get_sid():
    global sid
    if vvoice.value == "Adele":
        sid = gr.Dropdown(label="说话人：请至少选择一位目标说话人", value="slicer_adele", visible=False)
    elif vvoice.value == "Justin Bieber":
        sid = gr.Dropdown(label="说话人：请至少选择一位目标说话人", value="JustinBieber", visible=False)
    elif vvoice.value == "Taylor Swift":
        sid = gr.Dropdown(label="说话人：请至少选择一位目标说话人", value="taylor", visible=False)
    elif vvoice.value == "Trump":
        sid = gr.Dropdown(label="说话人：请至少选择一位目标说话人", value="trump", visible=False)
    else:
        sid = gr.Dropdown(label="说话人：请至少选择一位目标说话人", value="slicer-my", visible=False)
    return sid.value


def get_use_microphone():
    use_microphone = gr.Checkbox(label="使用麦克风输入", visible=False)
    return use_microphone.value


def get_auto_f0():
    auto_f0 = gr.Checkbox(
        label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会跑调）",
        value=False, visible=False)
    return auto_f0.value


def get_f0_predictor():
    f0_predictor = gr.Radio(label="f0预测器选择（如遇哑音可以更换f0预测器解决，crepe为原F0使用均值滤波器）",
                            choices=f0_options, value="rmvpe", visible=False)
    return "rmvpe"


def get_cr_threshold():
    cr_threshold = gr.Number(
        label="F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音",
        value=0.05, visible=False)
    return 0.05


def get_vc_transform():
    vc_transform = gr.Number(
        label="变调（整数，可以正负。由于男女歌手音域的差异，需要通过适当变调来提升音色的相似性。12即代表一个8度）",
        value=0, visible=False)
    return vc_transform.value


def get_cluster_ratio():
    cluster_ratio = gr.Number(
        label="聚类模型/特征检索混合比例，0-1之间，默认为0不启用聚类或特征检索，能提升音色相似度，但会导致咬字下降",
        value=0, visible=False)
    return 0


def get_k_step():
    k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100,
                       minimum=1, maximum=1000, visible=False)
    return k_step.value


def get_output_format():
    output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value="wav", visible=False)
    return "wav"


def get_enhancer_adaptive_key():
    enhancer_adaptive_key = gr.Number(label="使NSF-HIFIGAN增强器适应更高的音域(单位为半音数)|默认为0",
                                      value=0, visible=False)
    return 0


def get_slice_db():
    slice_db = gr.Number(label="切片阈值", value=-50, visible=False)
    return -50


def get_cl_num():
    cl_num = gr.Number(label="音频自动切片，5为按默认方式切片，单位为秒/s，爆显存可以设置此处强制切片",
                       value=5, visible=False)
    return 5


def get_pad_seconds():
    pad_seconds = gr.Number(
        label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5,
        visible=False)
    return 0.5


def get_lg_num():
    lg_num = gr.Number(
        label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s",
        value=1, visible=False)
    return 1


def get_lgr_num():
    lgr_num = gr.Number(
        label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭",
        value=0.75, visible=False)
    return 0.75


def get_second_encoding():
    second_encoding = gr.Checkbox(
        label="二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False,
        visible=False)
    return second_encoding.value


def get_loudness_envelope_adjustment():
    loudness_envelope_adjustment = gr.Number(
        label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value=0, visible=False)
    return 0


def get_use_spk_mix():
    use_spk_mix = gr.Checkbox(label="动态声线融合，需要手动编辑角色混合轨道，没做完暂时不要开启", value=False,
                              interactive=False, visible=False)
    return use_spk_mix.value


def get_vc_submit():
    vc_submit = gr.Button("音频转换", variant="primary")
    return vc_submit.value


def get_vc_output1():
    vc_output1 = gr.Textbox(label="输出信息", visible=False)
    return vc_output1.value


def get_vc_output2():
    vc_output2 = gr.Audio(label="输出音频")
    return vc_output2.value


def clear_output():
    return gr.Textbox.update(value="Cleared!>_<")


def get_available_encoder():
    current_pretrain = os.listdir("pretrain")
    current_pretrain = [("pretrain/" + model) for model in current_pretrain]
    encoder_list = []
    for encoder, path in ENCODER_PRETRAIN.items():
        if path in current_pretrain:
            encoder_list.append(encoder)
    return encoder_list


# def mix_submit_click(js, mode):
#     try:
#         assert js.lstrip() != ""
#         modes = {"凸组合": 0, "线性组合": 1}
#         mode = modes[mode]
#         data = json.loads(js)
#         data = list(data.items())
#         model_path, mix_rate = zip(*data)
#         path = mix_model(model_path, mix_rate, mode)
#         return f"成功，文件被保存在了{path}"
#     except Exception as e:
#         if debug:
#             traceback.print_exc()
#         raise gr.Error(e)


# def onnx_export_func():
#     model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#     output_msg = ""
#     try:
#         for path in model_dirs:
#             pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
#             json_files = glob.glob(f"{root_dir}/{path}/*.json")
#             model_file = Path(pth_files[0]).name
#             json_file = Path(json_files[0]).name
#             try:
#                 onnx_export(path, json_file, model_file)
#                 output_msg += f"成功转换{path}\n"
#             except Exception as e:
#                 output_msg += f"转换{path}时出现错误: {e}\n"
#         return output_msg
#     except Exception as e:
#         if debug:
#             traceback.print_exc()
#         raise gr.Error(e)


# def load_raw_audio(audio_path):
#     audio_path = audio_path.replace("\"", "")
#     if not os.path.isdir(audio_path):
#         return "请输入正确的目录", None
#     audio_files = list_files(audio_path, extensions=AUDIO_EXTENSIONS, recursive=False)
#     if not audio_files:
#         return "未在目录中找到音频文件", None

#     return f"成功加载{len(audio_files)}条音频", str(os.path.join(audio_path, "output"))


# def auto_slice(input_dir, output_dir, max_sec, min_sec, min_silence_duration, max_silence_kept, progress=gr.Progress()):
#     if output_dir == "":
#         return "请先选择输出的文件夹"
#     if output_dir == input_dir:
#         return "输出目录不能和输入目录相同"
#     # 去除路径中的引号
#     output_dir = output_dir.replace("\"", "")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     audio_files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=False)
#     for file in progress.tqdm(audio_files, desc="Slicing"):
#         slice_audio_file_v2(
#             input_file=file,
#             output_dir=output_dir,
#             min_duration=min_sec,
#             max_duration=max_sec,
#             min_silence_duration=min_silence_duration,
#             top_db=-40,
#             hop_length=10,
#             max_silence_kept=max_silence_kept,
#         )

#     len_files, total_duration, avg_duration, min_duration, max_duration, short_files = length(output_dir,
#                                                                                               short_threshold=min_sec)

#     if short_files:
#         for i in short_files:
#             os.remove(str(os.path.join(output_dir, i[3])))
#             len_files -= 1
#         _, _, avg_duration, min_duration, _, _ = length(output_dir)

#     original_duration = 0
#     for file in audio_files:
#         original_duration += librosa.get_duration(filename=os.path.join(input_dir, file))

#     ratio = total_duration / original_duration

#     return (
#             f"成功将音频切分为{len_files}条片段，其中最长{max_duration:.2f}秒，最短{min_duration:.2f}秒，切片后的音频总时长{total_duration / 3600:.2f}小时，平均每条音频时长{avg_duration:.2f}秒\n" +
#             f"为原始音频时长的{ratio * 100:.2f}%")


def pack_autoload(model_to_pack):
    _, config_name, _ = auto_load(model_to_pack)
    if config_name == "no_config":
        return "未找到对应的配置文件，请手动选择", None
    else:
        _config = Config(os.path.join(config_read_dir, config_name), "json")
        _content = _config.read()
        spk_dict = _content["spk"]
        spk_list = ",".join(spk_dict.keys())
        return config_name, spk_list


# read default params
sovits_params, diff_params, second_dir_enable, sami_settings = get_default_settings()
ckpt_read_dir = second_dir if second_dir_enable else workdir
config_read_dir = second_dir if second_dir_enable else config_dir
diff_read_dir = diff_second_dir if second_dir_enable else diff_workdir
current_mode = get_current_mode()

# create dirs if they don't exist
dirs_to_check = [
    workdir,
    second_dir,
    diff_workdir,
    diff_second_dir,
    dataset_dir,
]
for dir in dirs_to_check:
    if not os.path.exists(dir):
        os.makedirs(dir)

# read ckpt list
ckpt_list, config_list, cluster_list, diff_list, diff_config_list = load_options()

# read available encoder list
encoder_list = get_available_encoder()

# read GPU info
ngpu = torch.cuda.device_count()
gpu_infos = []
if (torch.cuda.is_available() is False or ngpu == 0):
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if ("MX" in gpu_name):
            continue
        if (
                "RTX" in gpu_name.upper() or "10" in gpu_name or "16" in gpu_name or "20" in gpu_name or "30" in gpu_name or "40" in gpu_name or "A50" in gpu_name.upper() or "70" in gpu_name or "80" in gpu_name or "90" in gpu_name or "M4" in gpu_name or "P4" in gpu_name or "T4" in gpu_name or "TITAN" in gpu_name.upper()):  # A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
gpu_info = "\n".join(gpu_infos) if if_gpu_ok is True and len(gpu_infos) > 0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus = "-".join([i[0] for i in gpu_infos])

# read cuda info for inference
cuda = {}
min_vram = 0
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        current_vram = torch.cuda.get_device_properties(i).total_memory
        min_vram = current_vram if current_vram > min_vram else min_vram
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"
total_vram = round(min_vram * 9.31322575e-10) if min_vram != 0 else 0
auto_batch = total_vram - 2 if total_vram <= 12 and total_vram > 0 else total_vram
#print(f"Current vram: {total_vram} GiB, recommended batch size: {auto_batch}")

# Check BF16 support
amp_options = ["fp32", "fp16"]
if if_gpu_ok:
    if torch.cuda.is_bf16_supported():
        amp_options = ["fp32", "fp16", "bf16"]

    # Get F0 Options
f0_options = ["crepe", "pm", "dio", "harvest", "rmvpe", "fcpe"]

# Get Vocoder Options
vocoder_options = []
for dir in os.listdir("pretrain"):
    if os.path.isdir(os.path.join("pretrain", dir)):
        if os.path.isfile(os.path.join("pretrain", dir, "model")) and os.path.isfile(
                os.path.join("pretrain", dir, "config.json")):
            vocoder_options.append(dir)


def voice_change(evt: gr.SelectData):
    vvoice.value = evt.value
vvoice = gr.Radio(["Adele", "Justin Bieber", "Taylor Swift", "Trump", "xxxx"], type='value',elem_classes="radio")
loadckpt = gr.Button("确认选择好模型了吗", elem_classes="button")
unload = gr.Button("卸载模型", variant="primary", visible=False)
vc_transform = gr.Number(
                    label="变调（可以是正负整数。由于男女歌手音域的差异，需要通过适当变调来提升音色的相似性。12即代表一个8度。在音色为女声，待转音频为男声时，变调采用正整数，反之则采用负整数。）",
                    value=0)
vc_input3 = gr.Audio(label="上传音频", type="filepath", sources="upload",elem_classes="audio")
use_microphone = gr.Checkbox(label="使用麦克风输入", visible=False,elem_classes="button")

vc_submit = gr.Button("转换", variant="primary", elem_classes="button")
vc_output1 = gr.Textbox(label="输出信息", visible=False)
vc_output2 = gr.Audio(label="输出音频",elem_classes="audio")
interrupt_button = gr.Button("中止转换", variant="danger")
auto_f0 = gr.Checkbox(
                    label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会跑调）",
                    value=False, visible=False)
f0_predictor = gr.Radio(label="f0预测器选择（如遇哑音可以更换f0预测器解决，crepe为原F0使用均值滤波器）",
                                        choices=f0_options, value="rmvpe", visible=False)
cr_threshold = gr.Number(
                    label="F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音",
                    value=0.05, visible=False)
cluster_ratio = gr.Number(
                    label="聚类模型/特征检索混合比例，0-1之间，默认为0不启用聚类或特征检索，能提升音色相似度，但会导致咬字下降",
                    value=0, visible=False)
k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100,
                                   minimum=1, maximum=1000, visible=False)
output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value="wav",
                                         visible=False)
enhancer_adaptive_key = gr.Number(label="使NSF-HIFIGAN增强器适应更高的音域(单位为半音数)|默认为0",
                                                  value=0, visible=False)
slice_db = gr.Number(label="切片阈值", value=-50, visible=False)
cl_num = gr.Number(label="音频自动切片，5为按默认方式切片，单位为秒/s，爆显存可以设置此处强制切片",
                                   value=5, visible=False)
noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
pad_seconds = gr.Number(
                    label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5,
                    visible=False)
lg_num = gr.Number(
                    label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s",
                    value=1, visible=False)
lgr_num = gr.Number(
                    label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭",
                    value=0.75, visible=False)
second_encoding = gr.Checkbox(
                    label="二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False,
                    visible=False)
loudness_envelope_adjustment = gr.Number(
                    label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value=0, visible=False)
use_spk_mix = gr.Checkbox(label="动态声线融合，需要手动编辑角色混合轨道，没做完暂时不要开启", value=False,
                                          interactive=False, visible=False)
choice_ckpt = gr.Dropdown(label="音色选择", choices=ckpt_list, value="选择您想要转换的目标音色",
                                      visible=False)
model_branch = gr.Textbox(label="模型编码器", placeholder="根据模型自动选择", interactive=False,
                                      visible=False)
config_choice = gr.Dropdown(label="配置文件选择", choices=config_list,
                                        value="no_diff", visible=False)
config_info = gr.Textbox(label="配置文件编码器", placeholder="根据配置文件自动选择", visible=False)
diff_choice = gr.Dropdown(value="no_diff",
                                      interactive=True, visible=False)
diff_config_choice = gr.Dropdown(
                value="no_diff_config", interactive=True, visible=False)
cluster_choice = gr.Dropdown(
                value="no_clu", visible=False)
vocoder_choice = gr.Dropdown(value="nsf_hifigan", visible=False)
refresh = gr.Button("刷新按钮", visible=False)
enhance = gr.Checkbox(
                label="是否使用NSF_HIFIGAN增强，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭",
                value=False, visible=False)
only_diffusion = gr.Checkbox(
                label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，不建议使用",
                value=False, visible=False)
diffusion_method = gr.Dropdown(label="扩散模型采样器",
                                           choices=["dpm-solver++", "dpm-solver", "pndm", "ddim", "unipc"],
                                           value="dpm-solver++", visible=False)
diffusion_speedup = gr.Number(label="扩散加速倍数，默认为10倍", value=10, visible=False)
using_device = gr.Dropdown(label="推理设备，可使用CPU/GPU进行推理，默认使用最优设备",
                                   choices=[*cuda.keys(), "cpu"],
                                   value="请选择你的推理设备", visible=False)

model_message = gr.Textbox(label="输出信息", visible=False)
sid = gr.Dropdown(label="说话人：请至少选择一位目标说话人", value="默认说话人", visible=False)
