import os
import shutil
import argparse
from glob import glob
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
import tqdm


def process(filename, wavformat, size):
    songname = os.path.splitext(os.path.basename(filename))[0]
    singer = filename.split('/')[-2]
    slice_name = f'./slice/{singer}/{songname}'

    if not os.path.exists(f'./slice/{singer}'):
        os.makedirs(f'./slice/{singer}', exist_ok=True)

    audio_segment = AudioSegment.from_file(filename)
    list_split_on_silence = split_on_silence(
        audio_segment, min_silence_len=600,
        silence_thresh=-40,
        keep_silence=400)

    combined = audio_segment[:1]
    for chunk in list_split_on_silence:
        combined += chunk

    chunks = make_chunks(combined, size)

    for i, chunk in enumerate(chunks):
        chunk_name = f"{slice_name}_{i}.wav"
        if not os.path.exists(chunk_name):
            chunk.export(chunk_name, format="wav")


def main():
    parser = argparse.ArgumentParser(description='comosvc inference')
    parser.add_argument('--input', type=str, default='test.wav', help='输入音频文件路径，如 ./all.wav')
    parser.add_argument('--name', type=str, default='test', help='子文件夹名称，例如 speaker01')
    parser.add_argument('--wavformat', type=str, default='wav', help='音频文件格式，例如 wav 或 mp3')
    parser.add_argument('--size', type=int, default=10000, help='音频切片大小（毫秒）')

    args = parser.parse_args()

    # 使用 name 创建 dataset_raw 子文件夹
    subfolder = f'./dataset_raw/{args.name}'
    os.makedirs(subfolder, exist_ok=True)

    # 拷贝输入音频文件到 dataset_raw/{name}/
    filename = args.input
    target_file = os.path.join(subfolder, os.path.basename(filename))
    if not os.path.exists(target_file):
        shutil.copyfile(filename, target_file)

    # 查找该文件夹内所有符合格式的音频文件，进行处理
    files = glob(f'{subfolder}/*.{args.wavformat}')
    for file in tqdm.tqdm(files):
        process(file, args.wavformat, args.size)


if __name__ == '__main__':
    main()
