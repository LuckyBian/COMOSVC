import argparse
import torch
from loguru import logger
from torch.optim import lr_scheduler
import os

from data_loaders import get_data_loaders
import utils
from solver import train
from ComoSVC import ComoSVC
from Vocoder import Vocoder
from utils import load_teacher_model_with_pitch, traverse_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Train ComoSVC")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/diffusion.yaml",
        help="路径：配置文件 (.yml)"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["teacher", "student"],
        default="teacher",
        help="训练模式：teacher 表示微调教师模型，student 表示蒸馏学生模型"
    )

    parser.add_argument(
        "--teacher_model_path", "-p",
        type=str,
        default="logs/teacher/model_800000.pt",
        help="教师模型路径（用于 student 模式加载）"
    )

    return parser.parse_args()


if __name__ == "__main__":
    cmd = parse_args()

    # 加载配置
    args = utils.load_config(cmd.config)
    logger.info(f"✅ Loaded config: {cmd.config}")

    # 加载 vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)

    # 是否是 teacher 模式
    is_teacher = (cmd.mode == "teacher")

    # 初始化模型
    model = ComoSVC(
        args.data.encoder_out_channels,
        args.model.n_spk,
        args.model.use_pitch_aug,
        vocoder.dimension,
        args.model.n_layers,
        args.model.n_chans,
        args.model.n_hidden,
        teacher=is_teacher
    )

    if is_teacher:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
        initial_global_step, model, optimizer = utils.load_model(
            args.env.expdir, model, optimizer, device=args.device
        )
        logger.info("🚀 The Teacher Model is training now.")
    else:
        # 加载 teacher 模型参数（冻结/提取特征用）
        model = load_teacher_model_with_pitch(model, checkpoint_dir=cmd.teacher_model_path)
        optimizer = torch.optim.AdamW(params=model.decoder.denoise_fn.parameters())
        path_pt = traverse_dir(args.env.comodir, ['pt'], is_ext=False)
        if len(path_pt) > 0:
            initial_global_step, model, optimizer = utils.load_model(
                args.env.comodir, model, optimizer, device=args.device
            )
        else:
            initial_global_step = 0
        logger.info("🧠 The Student Model (CoMoSVC) is training now.")

    # 设置学习率、权重衰减
    for param_group in optimizer.param_groups:
        if is_teacher:
            param_group['initial_lr'] = args.train.lr
            param_group['lr'] = args.train.lr * (args.train.gamma ** max(((initial_global_step - 2) // args.train.decay_step), 0)) * 2
        else:
            param_group['initial_lr'] = args.train.comolr
            param_group['lr'] = args.train.comolr * (args.train.gamma ** max(((initial_global_step - 2) // args.train.decay_step), 0))
        param_group['weight_decay'] = args.train.weight_decay

    # 学习率调度器
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.train.decay_step,
        gamma=args.train.gamma,
        last_epoch=initial_global_step - 2
    )

    # 设置 GPU
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)

    # 把 optimizer 里的 tensor 移动到 GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)

    # 数据加载器
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)

    # 启动训练
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, teacher=is_teacher)
