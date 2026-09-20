# REHC-Former

配套论文：**REHC-Former: Transformer-Based Robotic Eye-in-Hand Calibration from a Single Image**。

本目录是整理后的公开源码，原始实验文件保留在相邻的原始目录中。上传 GitHub 时只选择本目录。

## 保留的内容

- 完整 RGB–Mask 双流 Transformer：流内自注意力、顺序双向交叉注意力、四项池化和独立位姿回归头。
- 9D 旋转表示及 SVD 投影、训练集平移标准化、Smooth L1 损失。
- RGB-only、Mask-only、Early Fusion、Late Fusion 四类消融模型。
- 独立 SegFormer 分割训练及推理、单张 RGB 自动分割后预测手眼变换。
- 通用仿真测试指标、可配置仿真采集示例和分割数据处理工具。

## 从发布目录移除的内容

真实棋盘格评估、多图位姿融合、变换方向组合搜索、实验专用报表、默认读取真实实验目录的入口，以及缓存和来源未说明的示例照片。训练中的验证过程和通用有标签仿真测试仍保留。

## 使用

在本目录运行以下命令，Python 版本不低于 3.10：

```bash
python -m pip install -r requirements.txt
python train.py --variant rehc_former --data_dir data/train --output_dir runs/rehc_former --rotation_repr 9d
python predict.py --image data/demo/rgb/frame.png --mask data/demo/mask/frame.png --checkpoint runs/rehc_former/best_infer_only.pt --output outputs/prediction.json
```

若要仅提供 RGB 图片，让 SegFormer 自动生成掩码，安装 `requirements-segmentation.txt`，训练分割模型后使用英文 [README](README.md) 中的 `--seg-model` 命令。

`T_EC` 将相机坐标转换到末端坐标，输出 JSON 明确给出平移单位。数据格式见 [DATA_FORMAT.md](docs/DATA_FORMAT.md)。所有相对路径以运行命令时的当前目录为基准。训练与消融共用外参组划分，不把同一外参下的多张图片分到训练和验证两侧。

## 与原始实验的关系

整理中统一了主模型与消融的分组划分，修正了 Late Fusion 掩码分支额外下采样的问题。旧 Late Fusion 权重没有新标记时，加载器会明确提示并保留旧计算方式；新训练默认使用修正方式。这些变化可能影响重新训练的结果，详见 [论文对应说明](docs/PAPER_ALIGNMENT.md)。

当前未包含训练权重、数据集、仿真场景/CAD、论文 PDF、RGB Direct Regression 与 ResNet+MLP 两个独立对比基线。不能把现有 RGB-only 消融当作 Direct Regression。论文中的真实实验指标是固定物体重建的一致性，不是绝对标定误差。

源码可用于继续训练和部署准备，尚不能仅凭本目录复现论文所有结果。发布前还需作者确定许可证，并在准备好后补充论文公开链接及数据/权重获取方式。
