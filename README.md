# 🎤 CoMoSVC 声音转换系统使用说明

本项目是对 [CoMoSVC 原始项目](https://github.com/Grace9994/CoMoSVC/tree/main) 的整理与封装，旨在实现从原始音频 → 数据预处理 → 教师模型训练 → 学生模型蒸馏 → 推理转换 的完整流程，支持多说话人语音转换与音高控制。

📄 论文链接： [CoMoSVC: A Diffusion-based Singing Voice Conversion System With Pitch and Timbre Modeling](https://arxiv.org/abs/2401.01792)


---

## 📦 环境配置


```bash
conda env create -f environment.yml
conda activate vc
```

> ⚠️ 如果使用 `environment.yml` 创建环境失败，可参考 `vc/env.txt` 手动安装依赖：[vc/env.txt](vc/env.txt)


---

## 📥 模型下载

请下载下表中列出的模型压缩包，并将其 **解压后放入 `./vc` 目录下**：

| 名称                  | 下载链接              |
|-----------------------|-----------------------|
| Content.zip           | 🔗 [下载 Content.zip](链接待补充)           |
| singer_hifigan.zip    | 🔗 [下载 singer_hifigan.zip](链接待补充)    |
| m4singer_pe.zip       | 🔗 [下载 m4singer_pe.zip](链接待补充)       |
| pretrained_models.zip | 🔗 [下载 pretrained_models.zip](链接待补充) |
| logs.zip              | 🔗 [下载 logs.zip](链接待补充)              |
| wavlm.zip             | 🔗 [下载 wavlm.zip](链接待补充)             |

> 📂 解压后，确保这些目录结构位于项目根目录 `./vc/` 下，例如：`./vc/logs/teacher/model_700600.pt`



## 📂 项目结构示例

```
vc/
├── configs/
├── configs_template/
├── Content/
├── dataset/
├── dataset_raw/
├── filelists/
├── input/
├── logs/
├── m4singer_hifigan/
├── m4singer_pe/
├── pretrained_models/
├── results/
├── slice/
├── vocoder/
├── wavlm/
├── env.txt               # 环境
├── environment.yml       # 环境
├── run.py                # 主控脚本
├── run.sh                # 一键用脚本
```
---

## 🗂️ 数据预处理流程

```bash
python run.py \
  --name test \
  --raw_wav /path/to/all.wav \
  --stage 1 \
  --base_dir /path/to/codebase
```

---

## 🎼 特征提取

系统将自动完成音频切分、重采样、特征提取、embedding、平均说话人向量提取等全部流程。

---

## 🧠 教师模型训练 / 微调

```bash
python run.py \
  --name test \
  --raw_wav /path/to/all.wav \
  --stage 2 \
  --mode teacher
```

---

## 🧑‍🎓 学生模型训练（蒸馏）

```bash
python run.py \
  --name test \
  --raw_wav /path/to/all.wav \
  --stage 2 \
  --mode student \
  --teacher_model_path logs/teacher/model_700600.pt
```

---

## 🔁 推理与音高转换

```bash
python run.py \
  --name test \
  --stage 3 \
  --inference_folder ./input \
  --inference_keys 0 \
  --spk_list test
```

---

## 🧪 分阶段执行

| Stage | 内容说明             |
|-------|----------------------|
| 0     | 原始音频切片         |
| 1     | 数据预处理 + 特征提取 |
| 2     | 教师训练或学生蒸馏   |
| 3     | 推理转换              |

---


