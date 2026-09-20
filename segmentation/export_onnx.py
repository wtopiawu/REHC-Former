from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from segmentation.src.model import build_segformer
from segmentation.src.utils import get_device, load_checkpoint, load_yaml


class SegFormerONNXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, out_size: tuple[int, int]):
        super().__init__()
        self.model = model
        self.out_size = out_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=pixel_values).logits
        logits = F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained SegFormer-B1 to ONNX.")
    parser.add_argument("--config", type=str, default="outputs/segformer_b1_gripper/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/segformer_b1_gripper/checkpoints/best.pth")
    parser.add_argument("--output", type=str, default="outputs/segformer_b1_gripper/segformer_b1_gripper.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic batch only. Height/width stay fixed for stability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device(args.device)

    model = build_segformer(cfg["model"])
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    h, w = [int(x) for x in cfg["input"]["image_size"]]
    wrapper = SegFormerONNXWrapper(model, out_size=(h, w)).eval().to(device)
    dummy = torch.randn(1, 3, h, w, device=device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"pixel_values": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    print(f"ONNX exported to: {output_path}")
    print("Output is raw logits [B, 2, H, W]. Apply softmax and take class 1 for foreground probability.")


if __name__ == "__main__":
    main()
