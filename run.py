import argparse
import os
import subprocess


def run_command(command: str):
    print(f"\n▶️ Running: {command}")
    subprocess.run(command, shell=True, check=True)


def run_pipeline(
    name,
    wavformat,
    raw_wav,
    stage,
    base_dir,
    slice_size,
    num_process,
    teacher_model_path,
    config_path,
    spk_list,
    inference_keys,
    inference_folder,
    mode
):
    py = lambda file: f"python {os.path.join(base_dir, file)}"

    if stage <= 0:
        run_command(
            f"{py('preparation_slice.py')} "
            f"--input {raw_wav} --wavformat {wavformat} --size {slice_size} --name {name}"
        )

    if stage <= 1:
        datasetname = f"./dataset/{name}"
        os.makedirs(datasetname, exist_ok=True)
        run_command(f"{py('preprocessing1_resample.py')} --bin_idx 0 --total_bins 1 --num_process 1 --name {name}")
        run_command(f"{py('preprocessing2_flist.py')} --speaker {name}")
        run_command(f"{py('preprocessing3_feature.py')} --num_processes {num_process} --name {name}")
        run_command(f"{py('easy_extract.py')} --num_processes {num_process} --name {name}")
        run_command(f"{py('average_spk.py')} --name {name}")

    if stage <= 2:
        log_target = os.path.join(base_dir, f"logs/{name}")
        model_exists = (
            os.path.exists(log_target)
            and any(f.endswith(".pt") for f in os.listdir(log_target))
        )

        if model_exists:
            print(f"✅ 模型已存在于 ./logs/{name}/，跳过复制。")
        else:
            print(f"📦 模型未找到，复制 ./logs/como/* 到 ./logs/{name}/")
            os.makedirs(log_target, exist_ok=True)
            run_command(f"cp -r {os.path.join(base_dir, 'logs/como/*')} {log_target}/")

        if mode == "teacher":
            run_command(
                f"{py('train.py')} --config {config_path} --mode teacher --teacher_model_path {teacher_model_path}"
            )
        else:
            run_command(
                f"{py('train.py')} --config {config_path} --mode student --teacher_model_path {teacher_model_path}"
            )

    if stage <= 3:
        os.makedirs(f"./results/{name}", exist_ok=True)
        spk_args = " ".join(spk_list)
        key_args = " ".join(str(k) for k in inference_keys)
        run_command(
            f"{py('inference_main.py')} "
            f"--folder {inference_folder} "
            f"--spk_list {spk_args} "
            f"--keys {key_args} "
            f"--teacher_model_path logs/{name}/model_700600.pt "
            f"--teacher_config_path {config_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="封装完整 VC 训练推理流程")
    parser.add_argument("--name", type=str, required=True, help="说话人名称")
    parser.add_argument("--wavformat", type=str, default="wav", help="原始音频格式")
    parser.add_argument("--raw_wav", type=str, required=True, help="原始 WAV 文件路径")
    parser.add_argument("--stage", type=int, default=0, help="从哪个阶段开始执行")
    parser.add_argument("--base_dir", type=str, default=".", help="Python 脚本所在根目录")
    parser.add_argument("--slice_size", type=int, default=10000, help="切片长度（ms）")
    parser.add_argument("--num_process", type=int, default=2, help="多进程数量")

    # 训练相关
    parser.add_argument("--teacher_model_path", type=str, default="logs/teacher/model_700600.pt", help="教师模型路径")
    parser.add_argument("--config_path", type=str, default="configs/diffusion.yaml", help="训练用配置文件路径")
    parser.add_argument("--mode", type=str, choices=["teacher", "student"], default="teacher", help="训练模式：teacher / student")

    # 推理相关
    parser.add_argument("--spk_list", nargs='+', default=["test"], help="推理目标 speaker 列表")
    parser.add_argument("--inference_keys", nargs='+', type=int, default=[0], help="转调音高")
    parser.add_argument("--inference_folder", type=str, default="./input", help="推理输入音频目录")

    args = parser.parse_args()

    run_pipeline(
        name=args.name,
        wavformat=args.wavformat,
        raw_wav=args.raw_wav,
        stage=args.stage,
        base_dir=args.base_dir,
        slice_size=args.slice_size,
        num_process=args.num_process,
        teacher_model_path=args.teacher_model_path,
        config_path=args.config_path,
        spk_list=args.spk_list,
        inference_keys=args.inference_keys,
        inference_folder=args.inference_folder,
        mode=args.mode
    )
