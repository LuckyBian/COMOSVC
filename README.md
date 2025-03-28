# 🎤 CoMoSVC 声音转换系统使用说明

本项目基于 [CoMoSVC](https://github.com/) 实现了一个从原始音频 → 训练 → 蒸馏 → 推理 的完整声音转换流程，支持教师模型训练、学生模型蒸馏以及音高转换推理。

---

## 📦 环境配置


```bash
conda env create -f environment.yml
conda activate vc
```

> ⚠️ 如果使用 `environment.yml` 创建环境失败，可参考 `vc/env.txt` 手动安装依赖：[vc/env.txt](vc/env.txt)


---

## 📥 模型下载

请准备以下模型文件：

| 模型类型       | 路径                                       | 说明                        |
|----------------|--------------------------------------------|-----------------------------|
| 教师模型权重   | `logs/teacher/model_700600.pt`             | 用于学生模型蒸馏指导        |
| PE 模型（可选）| `./m4singer_pe/model_ckpt_steps_280000.ckpt` | 用于 decoder.pe 的加载     |
| CoMoSVC 模板   | `logs/como/`                               | 初始模型权重与配置文件     |

---

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


