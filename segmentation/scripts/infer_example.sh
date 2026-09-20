#!/usr/bin/env bash
set -e
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
python -m segmentation.infer \
  --config outputs/segformer_b1_gripper/config.yaml \
  --hf-model-dir outputs/segformer_b1_gripper/hf_best \
  --input "${1:?Usage: bash segmentation/scripts/infer_example.sh IMAGE_OR_DIR}" \
  --output outputs/segmentation \
  --device auto
