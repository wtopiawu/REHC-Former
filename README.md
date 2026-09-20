# REHC-Former

Code accompanying **REHC-Former: Transformer-Based Robotic Eye-in-Hand Calibration from a Single Image** by Xu Wu, Zhongtao Fu, Longhua Li, Bo Yang, Xuan Zhou, Zhenghua Huang, and Xubing Chen.

[中文说明](README_zh.md) · [Paper/code correspondence](docs/PAPER_ALIGNMENT.md) · [Data format](docs/DATA_FORMAT.md)

REHC-Former estimates the camera-to-end-effector transformation from one RGB observation. An independently trained SegFormer-B1 predicts the gripper foreground mask. RGB and mask streams use self-attention and **sequential bidirectional cross-attention**, followed by translation regression and 9D rotation regression with determinant-corrected SVD projection onto SO(3).

```text
RGB image -----------------> RGB features ----\
   |                                          self-attention
   +-> trained SegFormer --> mask features ---/    + cross-attention -> T_EC
```

The pose network is trained with simulator-rendered masks; segmentation is a separate preprocessing model at deployment. `T_EC` maps camera coordinates to end-effector coordinates: `X_E = R_EC @ X_C + t_EC`.

## Release contents

| Path | Purpose |
|---|---|
| `rehc_former/` | Full pose model, four ablations, shared data/loss/training code |
| `train.py` | Train the full model or an ablation |
| `predict.py` | Single-image inference with an existing or automatically predicted mask |
| `evaluate.py` | Generic labeled simulation evaluation |
| `segmentation/` | SegFormer training, mask inference, ONNX export, data tools |
| `data_collection/` | Optional scene-dependent CoppeliaSim collector |
| `tests/` | CPU regression checks using synthetic fixtures |

This source release does **not** include trained weights, experimental datasets, simulator scene/CAD assets, the manuscript PDF, or the separate RGB Direct Regression and ResNet+MLP comparison baselines. The RGB-only Transformer ablation is not the paper's Direct Regression baseline. This release has not been verified to reproduce the paper's numerical results; see the documented changes in [PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md).

## Installation

Use Python 3.10 or newer and run commands from this repository's root. Create an environment, install a matching PyTorch/torchvision pair for your machine, then:

```bash
python -m pip install -r requirements.txt
# For segmentation training and automatic RGB-only deployment:
python -m pip install -r requirements-segmentation.txt
```

CPU checks use Python 3.12, PyTorch 2.6.0 and torchvision 0.21.0. CUDA training requires a suitable PyTorch installation; it is not needed for the CPU checks. ONNX export additionally requires `onnx`.

## Train the pose model

Prepare the paired dataset described in [DATA_FORMAT.md](docs/DATA_FORMAT.md). Relative command-line paths are resolved from the current working directory.

```bash
python check_dataset.py --data_dir data/train --label_mode tec
python train.py --data_dir data/train --output_dir runs/rehc_former --variant rehc_former --rotation_repr 9d --backbone_name lite_cnn --embed_dim 128 --depth 2 --num_heads 4 --patch_stride 2 --epochs 80 --batch_size 8
```

Add `--amp` for CUDA mixed precision. Defaults are retained from the supplied training scripts and are not a recovered configuration of the published experiments. Training/validation splitting is by extrinsic group. Translation statistics use only the training split. Checkpoints, `config.json`, `split_manifest.csv`, metrics and curves are written to the output directory. Use a fresh output directory for each run. If there is only one extrinsic group, no independent validation set or `best_*` checkpoint can be created; the `last_*` checkpoints are still written.

For controlled ablations, use the same data, seed and hyperparameters, changing only `--variant` and `--output_dir`:

| Variant | Paper term |
|---|---|
| `rgb_only` | RGB-only |
| `mask_only` | Mask-only |
| `early_fusion` | RGB–Mask Concatenation |
| `late_fusion` | Without Cross-Attention |

## Predict from one image

Supply trained weights; none are bundled. An existing RGB-aligned mask must encode foreground as 255 and background as 0:

```bash
python predict.py --image data/demo/rgb/frame.png --mask data/demo/mask/frame.png --checkpoint runs/rehc_former/best_infer_only.pt --output outputs/prediction.json
```

For a single RGB image with automatic mask prediction, first train SegFormer as described in [segmentation/README.md](segmentation/README.md), then:

```bash
python predict.py --image data/demo/rgb/frame.png --seg-model outputs/segformer_b1_gripper/hf_best --seg-config outputs/segformer_b1_gripper/config.yaml --checkpoint runs/rehc_former/best_infer_only.pt --output outputs/prediction.json --save-mask outputs/frame_mask.png
```

The JSON contains `T_EC`, its inverse `T_CE`, translation units and quaternion order (`xyzw`). The command needs no ground-truth pose, robot observations or calibration target. The original pose model parameter names are preserved for checkpoint compatibility. Checkpoints use PyTorch's restricted `weights_only=True` loader.

## Evaluate a held-out simulated dataset

```bash
python evaluate.py --data-dir data/test --checkpoint runs/rehc_former/best_infer_only.pt --output-dir outputs/test
```

The same evaluator supports all five pose variants. It exports per-image predictions, group means, mean/median/P95 translation and geodesic rotation error, and joint success rates at 5 mm/2° and 10 mm/5°. All translations are converted to mm. Pass `--points data/evaluation_points_C_mm.npy` to compute `e_ad` on a fixed N×3 point set in camera frame C, measured in mm. Without those points, `e_ad` is explicitly unavailable. Use the same point set for all models. No real-experiment reconstruction or multi-image-fusion evaluation is included.

## Checks

```bash
python -m unittest discover -s tests -v
```

Checks use artificial inputs and do not measure calibration accuracy. See [validation scope](docs/VALIDATION.md).

## Citation and release status

Please cite the manuscript title and authors above. Publication venue, year, DOI and a public paper link are not set in this source snapshot. No software license has been selected; the authors should add the intended license before publishing the repository.
