import soundfile as sf

def extract_channel(file_path, channel_index=0):
    # 读取音频文件
    data, samplerate = sf.read(file_path)
    
    # 提取指定声道的数据
    channel_data = data[:, channel_index]  # 根据索引选择声道
    
    return channel_data, samplerate

# 示例
file_path = "./test_audio/testvideo-baseoutput.flac"
channel_index = 1  # 获取第一个声道
channel_data, samplerate = extract_channel(file_path, channel_index)

# 保存提取的声道为新文件
sf.write("./test_audio/testvideo-baseoutput-channel_1.flac", channel_data, samplerate)
