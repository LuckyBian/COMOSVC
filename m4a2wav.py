import os
from pydub import AudioSegment

def convert_m4a_to_wav(folder_path):
    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith(".m4a"):
            m4a_path = os.path.join(folder_path, filename)
            wav_path = os.path.join(folder_path, filename[:-4] + ".wav")
            
            # 转换格式
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio.export(wav_path, format="wav")
            
            # 删除原始M4A文件
            os.remove(m4a_path)
            print(f"Converted and removed: {m4a_path}")

# 使用示例
convert_m4a_to_wav("/home/weizhenbian/vc/dataset_raw/newxue")
