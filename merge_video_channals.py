import soundfile as sf
import numpy as np

def merge_channels_to_mono(file_path):
    # 读取音频文件
    data, samplerate = sf.read(file_path)
    
    # 确保是立体声（2个声道）
    if data.shape[1] == 2:
        # 对两个声道的数据取平均，生成单声道数据
        mono_data = np.mean(data, axis=1)
        return mono_data, samplerate
    else:
        raise ValueError("音频不是立体声，无法合成单声道")

# 示例
file_path = "./test_audio/testvideo-baseoutput.flac"
mono_data, samplerate = merge_channels_to_mono(file_path)

# 保存合成后的单声道音频
sf.write("./test_audio/testvideo-baseoutput-merged_mono.flac", mono_data, samplerate)
