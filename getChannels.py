import soundfile as sf

def get_channels(file_path):
    with sf.SoundFile(file_path) as file:
        return file.channels

# 示例
file_path = "./test_audio/output.flac"
channels = get_channels(file_path)
print(f"该文件有 {channels} 个声道")
