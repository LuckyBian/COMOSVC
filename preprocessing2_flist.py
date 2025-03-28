import argparse
import os
import re
import wave
from random import shuffle
from loguru import logger
from tqdm import tqdm
from glob import glob
import utils

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            framerate = wav_file.getframerate()
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

def generate_file(name):
    filelist = glob('./dataset/' + name + '/*.wav')
    savename = './filelists/trainlist_' + name + '.txt'
    with open(savename, 'w') as f:
        for path in filelist:
            f.write(path + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate training file list and diffusion config')
    parser.add_argument('--speaker', type=str, required=True, help='说话人名称，例如 test')
    args = parser.parse_args()
    speaker = args.speaker

    generate_file(speaker)

    spk_dict = {speaker: 0}

    d_config_template = utils.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = 1
    d_config_template["env"]["expdir"] = './logs/' + speaker
    d_config_template["data"]["training_files"] = './filelists/trainlist_' + speaker + '.txt'
    d_config_template["spk"] = spk_dict

    logger.info("Writing to configs/diffusion.yaml")
    utils.save_config("configs/diffusion.yaml", d_config_template)

if __name__ == "__main__":
    main()
