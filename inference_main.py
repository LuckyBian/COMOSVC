import logging
import soundfile
import os
import time
import infer_tool
from infer_tool import Svc
import argparse
logging.getLogger('numba').setLevel(logging.WARNING)


def main():

    parser = argparse.ArgumentParser(description='comosvc inference')
    parser.add_argument('-t', '--teacher', action="store_false", help='if it is teacher model')
    parser.add_argument('--clip', type=float, default=0, help='Slicing the audios which are to be converted')

    parser.add_argument('-f', '--folder', type=str, default="/aifs4su/weizhenbian/code/vc/raw", help='Directory containing the WAV files')
    parser.add_argument('-k', '--keys', type=int, nargs='+', default=[0], help='To Adjust the Key')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['wujiawei'], help='The target singer')

    parser.add_argument('-cm', '--como_model_path', type=str, default="./logs/como/model_800000.pt", help='the path to checkpoint of CoMoSVC')
    parser.add_argument('-cc', '--como_config_path', type=str, default="./logs/como/config.yaml", help='the path to config file of CoMoSVC')

    parser.add_argument('-tm', '--teacher_model_path', type=str, default="/aifs4su/weizhenbian/code/vc/logs/wujiawei/model_700600.pt", help='the path to checkpoint of Teacher Model')
    parser.add_argument('-tc', '--teacher_config_path', type=str, default="/aifs4su/weizhenbian/code/vc/configs/diffusion.yaml", help='the path to config file of Teacher Model')

    args = parser.parse_args()

    # 新增代码，从指定文件夹读取所有.wav文件
    clean_names = [file for file in os.listdir(args.folder) if file.endswith('.wav')]

    keys = args.keys
    spk_list = args.spk_list
    slice_db = -40
    wav_format = 'wav'
    pad_seconds = 0.5
    clip = args.clip

    diffusion_model_path = args.teacher_model_path
    diffusion_config_path = args.teacher_config_path
    resultfolder = './results/' + args.spk_list[0]

    svc_model = Svc(diffusion_model_path, diffusion_config_path, args.teacher)

    infer_tool.mkdir(["raw", resultfolder])
    
    infer_tool.fill_a_to_b(keys, clean_names)
    for clean_name, tran in zip(clean_names, keys):
        raw_audio_path = os.path.join(args.folder, clean_name)  # 直接使用文件夹路径

        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path": raw_audio_path,
                "spk": spk,
                "tran": tran,
                "slice_db": slice_db,
                "pad_seconds": pad_seconds,
                "clip_seconds": clip
            }
            audio = svc_model.slice_inference(**kwarg)
            res_path = f'{resultfolder}/{clean_name}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time} 秒")
