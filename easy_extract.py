from ecapa_tdnn import ECAPA_TDNN_SMALL
import torch
from torchaudio.transforms import Resample
import soundfile as sf
import numpy as np
from multiprocessing import Pool
from multiprocessing import Process
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch.multiprocessing as mp
import glob
import os
import argparse

def init_model(model_name, checkpoint=None):
    config_path = None
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def verification(model, filename, device):
    savename = filename + '.spknew.npy'
    if not os.path.exists(savename):
        wav, sr = sf.read(filename)
        wav = torch.from_numpy(wav).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr, new_freq=16000)
        wav = resample1(wav)
        wav = wav.to(device)
        with torch.no_grad():
            emb = model(wav)
            np.save(savename, emb.cpu().numpy())

def process_batch(file_chunk, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    model = init_model('wavlm_large', './wavlm/ckt/wavlm-large/wavlm_large_finetune.pth')
    model = model.to(device)
    model.eval()
    for filename in tqdm(file_chunk, position=rank):
        verification(model, filename, device)

def parallel_process(filenames, num_processes, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, device=device))
        for task in tqdm(tasks, position=0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker embedding extraction")
    parser.add_argument('--num_processes', type=int, default=2, help='并行进程数，默认 2')
    parser.add_argument('--name', type=str, default='test', help='数据集子文件夹名，例如 test')
    args = parser.parse_args()

    pattern = f'./dataset/{args.name}/*.wav'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = glob.glob(pattern)

    mp.set_start_method("spawn", force=True)
    parallel_process(filenames, args.num_processes, device)
