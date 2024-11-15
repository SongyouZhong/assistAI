from moviepy.editor import VideoFileClip

def extract_audio(input_video_path, output_audio_path):
    # 加载视频
    video = VideoFileClip(input_video_path)
    # 提取音频
    audio = video.audio
    # 保存为 FLAC 格式
    audio.write_audiofile(output_audio_path, codec='flac')

# 示例：提取 MP4 视频中的音频为 FLAC 格式
input_video = "./testvideo.mp4"  # Windows 路径可以使用 /mnt/c/ 在 WSL 中
output_audio = "./test_audio/output.flac"
extract_audio(input_video, output_audio)
