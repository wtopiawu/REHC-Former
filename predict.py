"""Single RGB observation -> foreground mask -> camera-to-end-effector pose."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from rehc_former.factory import load_model, mask_options
from rehc_former.geometry import pose_to_T_EC_transform, invert_transform_np, matrix_to_quat_xyzw_np
from rehc_former.transforms import BasicImageTransform, MaskTransform


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mask", type=Path, help="Aligned foreground mask, encoded as 0/255")
    source.add_argument("--seg-model", type=Path, help="Trained SegFormer hf_best directory")
    parser.add_argument("--seg-config", type=Path, help="Segmentation training config.yaml; required with --seg-model")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.json"))
    parser.add_argument("--save-mask", type=Path, help="Optional path for an automatically predicted mask")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--mask-invert", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--soft-mask", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.seg_model and not args.seg_config:
        parser.error("--seg-config is required with --seg-model to preserve preprocessing")
    return args


def generate_mask(image, model_dir, config_path, device):
    import cv2
    from transformers import SegformerForSemanticSegmentation
    from segmentation.infer import predict_one
    from segmentation.src.utils import load_yaml

    cfg = load_yaml(config_path)
    model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    if model.config.num_labels != 2:
        raise ValueError("The segmentation model must have two classes: background=0, gripper=1")
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    threshold = float(cfg.get("infer", {}).get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("Segmentation threshold must be in [0, 1]")
    mask, _ = predict_one(model, image_bgr, cfg, device, threshold)
    return Image.fromarray(mask * 255)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    with Image.open(args.image) as handle:
        rgb = handle.convert("RGB")
    if args.mask:
        with Image.open(args.mask) as handle:
            mask = handle.convert("L")
    else:
        mask = generate_mask(rgb, args.seg_model, args.seg_config, device)
        if args.save_mask:
            args.save_mask.parent.mkdir(parents=True, exist_ok=True)
            mask.save(args.save_mask)
    if mask.size != rgb.size:
        raise ValueError("RGB and mask must have the same original size and pixel alignment")
    options = mask_options(checkpoint, args.mask_threshold, args.mask_invert, args.soft_mask)
    # Generated masks already have foreground=255, regardless of source training-mask encoding.
    if args.seg_model:
        options["invert"] = False
    size = tuple(checkpoint.get("image_size", checkpoint["model_config"]["image_size"]))
    image_tensor = BasicImageTransform(size, train=False)(rgb).unsqueeze(0).to(device)
    mask_tensor = MaskTransform(size, **options)(mask).unsqueeze(0).to(device)
    if mask_tensor.max() == 0:
        raise ValueError("The foreground mask is empty; check segmentation or mask encoding")
    output = model(image_tensor, mask_tensor)
    mean = torch.tensor(checkpoint["translation_mean"], device=device)
    std = torch.tensor(checkpoint["translation_std"], device=device)
    raw = np.eye(4, dtype=np.float32)
    raw[:3, :3] = output["pred_rotation_matrix"][0].cpu().numpy()
    raw[:3, 3] = (output["pred_translation_norm"][0] * std + mean).cpu().numpy()
    tec = pose_to_T_EC_transform(raw, checkpoint.get("label_mode", "tec"), checkpoint.get("pose_name", ""))
    payload = {
        "convention": "X_E = R_EC @ X_C + t_EC (camera to end effector)",
        "translation_unit": checkpoint["translation_unit"],
        "mask_source": "provided" if args.mask else "segmentation",
        "T_EC": tec.tolist(),
        "T_CE": invert_transform_np(tec).tolist(),
        "translation": tec[:3, 3].tolist(),
        "rotation_matrix": tec[:3, :3].tolist(),
        "quaternion_xyzw": matrix_to_quat_xyzw_np(tec[:3, :3]).tolist(),
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
