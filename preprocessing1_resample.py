import librosa
import os,sys,tqdm
import multiprocessing as mp
import soundfile as sf
from glob import glob
import argparse

def resample_one(filename):
    songname=filename.split('/')[-1]
    singer=filename.split('/')[-2]
    
    output_path='./dataset/'+singer+'/'+songname

    if os.path.exists(output_path):
        return
    wav, sr = librosa.load(filename, sr=24000)
    # normalize the volume
    wav = wav / (0.00001+max(abs(wav)))*0.95
    # write to file using soundfile
    try:
        sf.write(output_path, wav, 24000)
    except:
        print("Error writing file",output_path)
        return

def mkdir_func(input_path):
    singer=input_path.split('/')[-2]
    out_dir = './dataset/'+singer
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def resample_parallel(name,bin_idx,total_bins,num_process):
    input_paths = glob('./slice/'+name+'/*.wav')    
    print("input_paths",len(input_paths))
    input_paths = input_paths[int(bin_idx)*len(input_paths)//int(total_bins):int(bin_idx+1)*len(input_paths)//int(total_bins)]
    # multiprocessing with progress bar
    pool = mp.Pool(num_process)
    for _ in tqdm.tqdm(pool.imap_unordered(resample_one, input_paths), total=len(input_paths)):
        pass

def path_parallel(name):
    input_paths = glob('./slice/'+name+'/*.wav')
    input_paths = list(set(input_paths))#sort
    print("input_paths",len(input_paths))

    # multiprocessing with progress bar
    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(mkdir_func, input_paths), total=len(input_paths)):
        pass

def main():
    parser = argparse.ArgumentParser(description="Parallel audio preprocessing and resampling")
    parser.add_argument('--bin_idx', type=int, default=0, help='当前处理的 bin 索引（-1 表示处理全部）')
    parser.add_argument('--total_bins', type=int, default=1, help='总共的 bin 数')
    parser.add_argument('--num_process', type=int, default=1, help='每个 bin 内的并发进程数')
    parser.add_argument('--name', type=str, default='test', help='数据集子文件夹名称')

    args = parser.parse_args()

    if args.bin_idx == -1:
        path_parallel(args.name)
    else:
        resample_parallel(args.name, args.bin_idx, args.total_bins, args.num_process)

if __name__ == "__main__":
    main()
