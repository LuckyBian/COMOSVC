import torchaudio
import math
import torch
import numpy as np
import librosa
from scipy.io import wavfile
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Pool
import multiprocessing
import os
from glob import glob
from tqdm import tqdm
import argparse


def calculate(filelist):
    # Calculate the mean of the speaker embeddings
    spk = []
    for filename in filelist:
        spk_emb = np.load(filename, allow_pickle=True)
        spk.append(spk_emb)
    final_spk = sum(spk) / len(spk)
    singer = filelist[0].split('/')[-2]
    savename = f'./dataset/{singer}/{singer}.spknew.npy'
    np.save(savename, final_spk)
    print(f"✅ Saved average embedding to {savename}")

def main():
    parser = argparse.ArgumentParser(description="Average speaker embedding generator")
    parser.add_argument('--name', type=str, required=True, help='说话人子文件夹名，例如 test')
    args = parser.parse_args()

    spkname = f'./dataset/{args.name}/*.spknew.npy'
    filelist = glob(spkname)

    if not filelist:
        print(f"❌ No .spknew.npy files found in ./dataset/{args.name}/")
        return

    calculate(filelist)

if __name__ == "__main__":
    main()
