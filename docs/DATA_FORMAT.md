# Dataset contract

```text
data/train/
  rgb/000001.png
  mask/000001.png
  anno/000001.json
```

An annotation contains:

```json
{
  "image": "rgb/000001.png",
  "mask": "mask/000001.png",
  "extrinsic_group_id": "mount_001",
  "T_EC_unit": "mm",
  "T_EC": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
}
```

This is a schema illustration, not measured calibration data. Paths are relative to the dataset root. Provide valid 4×4 rigid transforms and one consistent unit (`mm`, `cm`, or `m`) and direction throughout a training dataset. The default training label is `T_EC`; the optional `tce` mode requires `T_CE` annotations and is not the paper default.

RGB and mask must be registered pixel-for-pixel. Pose masks are single-channel 0/255 images (white = visible gripper); use `--mask_invert` in training for inverse encoding. The transform resizes masks with nearest-neighbor interpolation. The segmentation dataset additionally supports 0/1 labels, but convert those to 0/255 before feeding the pose dataset. Pose training uses only photometric augmentation so geometry remains aligned.

Assign a stable `extrinsic_group_id` to every fixed camera mounting and use the same ID for all its images. If absent, the loader derives an ID from the pose rounded to five decimal places; explicit IDs are preferred. Group identifiers must be unique across different mounting configurations. Keep external test mounting configurations separate from the training directory. Automatically splitting by group does not independently check an external test dataset for overlap.

Training writes the split manifest and computes per-axis translation mean/std only from training samples. Use identical manifests and masks for full-model/ablation comparisons. A single group cannot provide an independent validation group. Generated cache files contain data paths and are ignored by Git.

The paper describes 12,000 simulation training observations and 100 unseen test configurations with 15 observations each. The collector is a scene-specific template and does not bundle those exact splits or the complete appearance-randomization pipeline.
