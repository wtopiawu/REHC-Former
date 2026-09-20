#!/usr/bin/env bash
set -e
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
python -m segmentation.train --config segmentation/configs/segformer_b1.yaml --device auto
