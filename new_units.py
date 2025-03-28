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

import utils
from Vocoder import Vocoder
from mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

dconfig = utils.load_config("/home/weizhenbian/vc/config2/config.yaml")


def process_one(filename, hmodel, device, hop_length, sampling_rate, filter_length, win_length):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = filename + ".new.soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k) # extract content from the pre-trained model
        torch.save(c.cpu(), soft_path)


def process_batch(file_chunk, hop_length, sampling_rate, filter_length, win_length, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    from Features import ContentVec768L12
    hmodel = ContentVec768L12(device = device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, device, hop_length, sampling_rate, filter_length, win_length)

def parallel_process(filenames, num_processes, hop_length, sampling_rate, filter_length, win_length, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, hop_length, sampling_rate, filter_length, win_length,device=device))
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--config", type=str, default='/home/weizhenbian/vc/config2/config.yaml', help="path to input dir")
    parser.add_argument(
        '-n','--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores')
    args = parser.parse_args()

    dconfig = utils.load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: " + str(device))
    print("Loading Mel Extractor...",dconfig.vocoder.type)
    # mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)

    filenames = glob("/aifs4su/data/weizhen/data/hua/cut2/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(filenames, num_processes, dconfig.data.hop_length, dconfig.data.sampling_rate, dconfig.data.filter_length, dconfig.data.win_length, device)