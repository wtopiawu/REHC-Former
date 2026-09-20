# Gripper segmentation

Run these commands from the repository root after installing `requirements-segmentation.txt`.

Use matching relative filename stems under `data/gripper/images/{train,val}` and `data/gripper/masks/{train,val}`. Binary masks have background=0 and foreground>0. Edit `segmentation/configs/segformer_b1.yaml` for dataset paths. The initial backbone is `nvidia/segformer-b1-finetuned-ade-512-512`; first-time training downloads it unless `local_files_only` is enabled with a populated cache.

```bash
python -m segmentation.train --config segmentation/configs/segformer_b1.yaml
python -m segmentation.infer --config outputs/segformer_b1_gripper/config.yaml --hf-model-dir outputs/segformer_b1_gripper/hf_best --input data/demo/rgb --output outputs/masks
```

Outputs include `masks/<stem>_mask.png` (0/255) and overlays. The root `predict.py` calls the same preprocessing and postprocessing with a trained `hf_best` directory. Train segmentation separately from pose regression.

Optional tools:

```bash
python segmentation/tools/check_dataset.py --root data/gripper
python segmentation/tools/split_dataset.py --images data/raw/images --masks data/raw/masks --out data/gripper --mode copy
python segmentation/tools/augment_pairs.py --input-root data/segmentation_train --output-root data/segmentation_augmented --seed 42
python -m segmentation.export_onnx --config outputs/segformer_b1_gripper/config.yaml --checkpoint outputs/segformer_b1_gripper/checkpoints/best.pth --output outputs/segformer.onnx
```

`split_dataset.py` randomly partitions images: only use it when inputs are independent. For correlated captures, split by acquisition/scene before augmentation. Keep all augmented versions of a source image in its original split. The geometric augmenter operates on segmentation image/mask pairs; it does not update camera geometry or pose annotations and must not be used as a pose-data generator. It adds background noise; despite the original filename, it does not composite arbitrary background images.

ONNX export needs the optional `onnx` dependency. Its output is class logits; export has not been validated in the source-cleanup checks.
