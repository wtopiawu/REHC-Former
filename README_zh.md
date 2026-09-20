# REHC-Former

**基于 Transformer 的单图像机器人眼在手上标定**

[English](README.md) · [数据格式](docs/DATA_FORMAT.md) · [分割模型](segmentation/README.md)

REHC-Former 从一张包含机器人末端执行器的 RGB 图像中，估计相机到末端执行器的刚体变换。模型通过双流 Transformer 融合 RGB 外观信息与前景掩码的几何信息，直接预测平移和旋转。

本项目为论文 **REHC-Former: Transformer-Based Robotic Eye-in-Hand Calibration from a Single Image** 的配套代码。作者：Xu Wu、Zhongtao Fu、Longhua Li、Bo Yang、Xuan Zhou、Zhenghua Huang 和 Xubing Chen。

## 方法概述

- **单图像推理**：推理时无需标定板或多组机器人位姿，即可估计手眼变换。
- **RGB–Mask 双流融合**：分别提取外观与前景特征，通过流内自注意力和顺序双向交叉注意力进行信息交互。
- **位姿回归**：独立回归头预测平移和 9D 旋转表示，旋转经 SVD 投影得到有效的旋转矩阵。
- **仿真到真实部署**：位姿网络使用仿真 RGB–Mask 数据训练；真实图像的前景掩码由独立训练的 SegFormer-B1 预测。

输出变换 `T_EC` 将相机坐标转换到末端执行器坐标：

```text
X_E = R_EC @ X_C + t_EC
```

## 环境安装

使用 Python 3.10 或更新版本。根据 CPU 或 CUDA 环境安装配套的 PyTorch 与 torchvision，然后在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
# 分割模型训练及自动掩码预测：
python -m pip install -r requirements-segmentation.txt
```

CPU 测试环境为 Python 3.12、PyTorch 2.6.0 和 torchvision 0.21.0。导出 ONNX 时还需安装 `onnx`。

## 数据准备

训练数据由对齐的 RGB 图像、二值前景掩码和相机到末端执行器的位姿标签组成。目录结构、标签约定和平移单位见[数据格式说明](docs/DATA_FORMAT.md)。

```bash
python check_dataset.py --data_dir data/train --label_mode tec
```

仿真数据采集见 [CoppeliaSim 采集指南](data_collection/README.md)；掩码标注、分割训练与推理见[分割模型指南](segmentation/README.md)。

数据集与预训练权重暂未提供下载。运行推理前，请准备数据并完成相应模型的训练。

## 模型训练

```bash
python train.py --data_dir data/train --output_dir runs/rehc_former --variant rehc_former --rotation_repr 9d --backbone_name lite_cnn --embed_dim 128 --depth 2 --num_heads 4 --patch_stride 2 --epochs 80 --batch_size 8
```

使用 CUDA 时可添加 `--amp` 开启混合精度。训练集与验证集按外参组划分，同一外参下的图像不会同时进入两者；平移标准化仅使用训练集统计量。权重、配置、数据划分和训练指标保存到 `--output_dir`，每次实验建议使用独立输出目录。

进行消融实验时，固定数据划分、随机种子和训练超参数，通过 `--variant` 选择模型：

| 参数值 | 模型 |
|---|---|
| `rehc_former` | 带交叉注意力的完整 RGB–Mask 模型 |
| `rgb_only` | 仅 RGB 输入的 Transformer |
| `mask_only` | 仅 Mask 输入的 Transformer |
| `early_fusion` | RGB 与 Mask 在输入端拼接 |
| `late_fusion` | 不使用交叉注意力的双流模型 |

## 单图像推理

### 使用已有掩码

掩码须与 RGB 图像对齐，前景值为 255，背景值为 0：

```bash
python predict.py --image data/demo/rgb/frame.png --mask data/demo/mask/frame.png --checkpoint runs/rehc_former/best_infer_only.pt --output outputs/prediction.json
```

### 自动预测掩码

提供训练好的 SegFormer 和位姿模型权重，即可从一张 RGB 图像完成推理：

```bash
python predict.py --image data/demo/rgb/frame.png --seg-model outputs/segformer_b1_gripper/hf_best --seg-config outputs/segformer_b1_gripper/config.yaml --checkpoint runs/rehc_former/best_infer_only.pt --output outputs/prediction.json --save-mask outputs/frame_mask.png
```

输出 JSON 包含 `T_EC`、逆变换 `T_CE`、平移单位和四元数顺序（`xyzw`）。命令中的相对路径均以当前工作目录为基准。

## 模型评估

使用带位姿标签的独立仿真测试集进行评估：

```bash
python evaluate.py --data-dir data/test --checkpoint runs/rehc_former/best_infer_only.pt --output-dir outputs/test
```

评估指标包括平移误差（mm）、旋转测地误差（°），以及 5 mm/2° 和 10 mm/5° 阈值下的联合成功率。如需计算平均点距离 `e_ad`，添加 `--points data/evaluation_points_C_mm.npy`，提供相机坐标系下、单位为 mm 的固定 N×3 点集。比较不同模型时应使用同一点集。

## 项目结构

| 路径 | 功能 |
|---|---|
| `rehc_former/` | 位姿模型、数据加载、损失函数和训练工具 |
| `train.py` | 位姿模型训练 |
| `predict.py` | 单图像推理 |
| `evaluate.py` | 仿真测试集评估 |
| `segmentation/` | SegFormer 训练、推理与 ONNX 导出 |
| `data_collection/` | CoppeliaSim 数据采集 |
| `tests/` | CPU 流程测试 |

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用合成数据检查模型和流程行为。实验配置与验证细节见[实现说明](docs/PAPER_ALIGNMENT.md)和[验证说明](docs/VALIDATION.md)。

## 引用

如果本项目对你的研究有帮助，请引用：

> Xu Wu, Zhongtao Fu, Longhua Li, Bo Yang, Xuan Zhou, Zhenghua Huang, and Xubing Chen. **REHC-Former: Transformer-Based Robotic Eye-in-Hand Calibration from a Single Image.**
