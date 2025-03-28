import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm
import torchaudio

import utils
from Vocoder import Vocoder
from mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def process_one(filename, hmodel, device, hop_length, sampling_rate, filter_length, win_length, vocoder_type, vocoder_ckpt):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav).unsqueeze(0)

    soft_path = filename + ".new.soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        from Features import DioF0Predictor
        f0_predictor = DioF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))

    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        if sr != sampling_rate:
            raise ValueError(f"{sr} SR doesn't match target {sampling_rate} SR")
        spec = spectrogram_torch(
            audio_norm,
            filter_length,
            sampling_rate,
            hop_length,
            win_length,
            center=False,
        )
        torch.save(torch.squeeze(spec, 0), spec_path)

    volume_path = filename + ".vol.npy"
    if not os.path.exists(volume_path):
        volume_extractor = utils.Volume_Extractor(hop_length)
        volume = volume_extractor.extract(audio_norm)
        np.save(volume_path, volume.to('cpu').numpy())

    mel_path = filename + ".mel.npy"
    mel_extractor = Vocoder(vocoder_type, vocoder_ckpt, device=device)
    if not os.path.exists(mel_path) and mel_extractor is not None:
        mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
        mel = mel_t.squeeze().to('cpu').numpy()
        np.save(mel_path, mel)


def process_batch(file_chunk, hop_length, sampling_rate, filter_length, win_length, vocoder_type, vocoder_ckpt, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")

    from Features import ContentVec256L9
    hmodel = ContentVec256L9(device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")

    for filename in tqdm(file_chunk, position=rank):
        process_one(filename, hmodel, device, hop_length, sampling_rate, filter_length, win_length, vocoder_type, vocoder_ckpt)


def parallel_process(filenames, num_processes, hop_length, sampling_rate, filter_length, win_length, vocoder_type, vocoder_ckpt, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(
                process_batch, file_chunk, hop_length, sampling_rate, filter_length, win_length, vocoder_type, vocoder_ckpt, device
            ))
        for task in tqdm(tasks, position=0):
            task.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing stage 3: feature extraction')
    parser.add_argument('--num_processes', type=int, default=5, help='并行进程数，默认为 5')
    parser.add_argument('--name', type=str, default='test', help='子文件夹名称，例如 test')
    parser.add_argument('--config', type=str, default='./configs/diffusion.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    dconfig = utils.load_config(args.config)
    vocoder_type = dconfig.vocoder.type
    vocoder_ckpt = dconfig.vocoder.ckpt

    # 获取音频文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = glob(f'./dataset/{args.name}/*.wav')
    shuffle(filenames)

    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(
        filenames, num_processes,
        dconfig.data.hop_length,
        dconfig.data.sampling_rate,
        dconfig.data.filter_length,
        dconfig.data.win_length,
        vocoder_type,
        vocoder_ckpt,
        device
    )
