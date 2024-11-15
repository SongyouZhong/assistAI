import argparse
import json
import torch
from torch import Tensor
from pathlib import Path
from importlib import import_module
import soundfile as sf
import numpy as np
# 获取模型
def get_model(model_config: dict, device: torch.device):
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    return model


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


# 评估单个音频文件
def evaluate_audio(model, audio,device):
    X_pad = pad(audio, 64600)
    # input_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # 转为张量，并添加 batch 维度
    x_inp = Tensor(X_pad).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x_inp)  # 调用模型进行推理

 # 使用 softmax 或 sigmoid 对 logits 进行转换
    # probabilities = torch.softmax(output[1], dim=1)  # 对每个样本进行 softmax 转换
    # 或者：
    probabilities = torch.sigmoid(output[1])  # 如果是二分类，可以直接用 sigmoid
    print("probabilities")
    print(probabilities)
    # 判断类别为 'True'（真实音频）的概率是否大于 0.5
    is_true_audio = probabilities[:, 1] >0.5  # probabilities[1] 是真实音频的概率
    print("is_true_audio")
    print(is_true_audio)
    result = is_true_audio.item() 
    print(result)

    return result

# 使用 Soundfile 加载音频文件
def process_audio_with_soundfile(file_path):
    audio, sr = sf.read(file_path)
    print(f"file {sr}")
    return audio, sr

# 评估文件
def evaluate_audio_file(file_path: Path, model, device):
    audio, sr = process_audio_with_soundfile(file_path)
    results=[]
    for i in range(1,10):
        result = evaluate_audio(model, audio,device)
        results.append(result)
    
    print(f"Evaluating {file_path.name}...")
    print("Result:", results)
    

# 主函数
def test(model_config: dict, audio_file_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载预训练模型
    model = get_model(model_config, device)
    model.eval()
    model.to(device)

    # 确保音频文件存在
    audio_file_path = Path(audio_file_path)
    if not audio_file_path.exists():
        print(f"Audio file {audio_file_path} not found!")
        return

    # 进行评估并保存结果
    evaluate_audio_file(audio_file_path, model, device)

if __name__ == "__main__":
    model_config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [1, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    # 模型和音频文件路径
    model_path = "./models/weights/AASIST-L.pth"  # 替换为模型路径
    audio_file_path = "./test_audio/merged_mono.flac"  # 替换为你的 .flac 文件路径

    # 调用主函数进行评估
    test(model_config, audio_file_path)
