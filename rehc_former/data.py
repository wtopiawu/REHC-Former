import json
import hashlib
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import BasicImageTransform, MaskTransform
from .geometry import extract_target_from_json, infer_target_length_unit, quat_xyzw_to_matrix_np


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
MASK_META_KEYS = [
    "mask",
    "mask_path",
    "mask_image",
    "mask_filename",
    "segmentation",
    "segmentation_mask",
    "foreground_mask",
]
MASK_DIR_NAMES = ["mask", "masks", "seg", "segs", "segmentation", "labels", "label"]
RGB_DIR_NAMES = ["rgb", "image", "images", "imgs"]


@dataclass
class SampleRecord:
    image_path: Path
    json_path: Path
    mask_path: Optional[Path]
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    rotation_matrix: np.ndarray
    pose_name: str
    translation_unit: str
    phase: str = "unknown"
    global_index: int = -1
    extrinsic_group_id: str = ""


class EyeInHandPoseMaskDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        translation_mean: np.ndarray,
        translation_std: np.ndarray,
        image_size: Tuple[int, int] = (480, 640),
        train: bool = False,
        require_mask: bool = True,
        mask_threshold: float = 0.5,
        mask_invert: bool = False,
        mask_binary: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.translation_mean = translation_mean.astype(np.float32)
        self.translation_std = translation_std.astype(np.float32)
        self.image_height, self.image_width = image_size
        self.train = train
        self.require_mask = bool(require_mask)
        self.image_transform = BasicImageTransform(image_size=(self.image_height, self.image_width), train=train)
        self.mask_transform = MaskTransform(
            image_size=(self.image_height, self.image_width),
            threshold=mask_threshold,
            invert=mask_invert,
            binary=mask_binary,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, sample: SampleRecord) -> torch.Tensor:
        if sample.mask_path is None:
            if self.require_mask:
                raise FileNotFoundError(f"样本缺少 mask: image={sample.image_path}, json={sample.json_path}")
            return torch.ones((1, self.image_height, self.image_width), dtype=torch.float32)
        with Image.open(sample.mask_path) as mask_file:
            return self.mask_transform(mask_file)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
        image = self.image_transform(image)
        mask = self._load_mask(sample)

        t = sample.translation.astype(np.float32)
        q = sample.quaternion_xyzw.astype(np.float32)
        t_norm = ((t - self.translation_mean) / self.translation_std).astype(np.float32)

        return {
            "image": image,
            "mask": mask,
            "translation": torch.from_numpy(t),
            "translation_norm": torch.from_numpy(t_norm),
            "quaternion_xyzw": torch.from_numpy(q),
            "rotation_matrix": torch.from_numpy(sample.rotation_matrix.astype(np.float32)),
            "rotation_9d": torch.from_numpy(sample.rotation_matrix.astype(np.float32).reshape(9)),
            "image_path": str(sample.image_path),
            "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
            "json_path": str(sample.json_path),
            "phase": sample.phase,
            "global_index": int(sample.global_index),
            "extrinsic_group_id": sample.extrinsic_group_id,
        }


# Compatibility alias.
EyeInHandPoseDataset = EyeInHandPoseMaskDataset
EyeToHandPoseDataset = EyeInHandPoseMaskDataset


def _safe_read_json(json_path: Path) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def _candidate_image_paths_from_meta(meta: Dict, json_path: Path, data_root: Path) -> List[Path]:
    candidates: List[Path] = []
    image_value = meta.get("image")
    if isinstance(image_value, str) and image_value.strip():
        image_rel = Path(image_value)
        if image_rel.is_absolute():
            candidates.append(image_rel)
        else:
            candidates.append(data_root / image_rel)
            candidates.append(json_path.parent / image_rel)

    stem = json_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidates.append(json_path.parent / f"{stem}{ext}")
        candidates.append(json_path.parent / f"{stem}{ext.upper()}")
        candidates.append(data_root / f"{stem}{ext}")
        candidates.append(data_root / f"{stem}{ext.upper()}")
        candidates.append(data_root / "rgb" / f"{stem}{ext}")
        candidates.append(data_root / "rgb" / f"{stem}{ext.upper()}")
    return _dedupe_paths(candidates)


def _find_matching_image(json_path: Path, meta: Dict, data_root: Path, image_index: Dict[str, Path]) -> Optional[Path]:
    for candidate in _candidate_image_paths_from_meta(meta, json_path, data_root):
        if _is_image_file(candidate):
            return candidate.resolve()
    return image_index.get(json_path.stem.lower())


def _build_image_index(data_root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for image_path in data_root.rglob(f"*{ext}"):
            stem = image_path.stem.lower()
            image_index.setdefault(stem, image_path.resolve())
        for image_path in data_root.rglob(f"*{ext.upper()}"):
            stem = image_path.stem.lower()
            image_index.setdefault(stem, image_path.resolve())
    return image_index


def _build_mask_index(data_root: Path) -> Dict[str, Path]:
    mask_index: Dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for mask_path in list(data_root.rglob(f"*{ext}")) + list(data_root.rglob(f"*{ext.upper()}")):
            parts = {part.lower() for part in mask_path.parts}
            name = mask_path.name.lower()
            if parts.intersection(MASK_DIR_NAMES) or "mask" in name or "seg" in name:
                mask_index.setdefault(mask_path.stem.lower().replace("_mask", ""), mask_path.resolve())
    return mask_index


def _resolve_relative_path(value: str, base_dirs: Sequence[Path]) -> List[Path]:
    value_path = Path(value)
    if value_path.is_absolute():
        return [value_path]
    return [base / value_path for base in base_dirs]


def _candidate_mask_paths_from_meta(meta: Dict, json_path: Path, image_path: Path, data_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for key in MASK_META_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            candidates.extend(_resolve_relative_path(value, [data_root, json_path.parent, image_path.parent]))

    image_rel_to_root: Optional[Path]
    try:
        image_rel_to_root = image_path.resolve().relative_to(data_root.resolve())
    except ValueError:
        image_rel_to_root = None

    suffixes = []
    for ext in IMAGE_EXTENSIONS:
        suffixes.append(ext)
        suffixes.append(ext.upper())

    stem = image_path.stem
    image_suffix_candidates = [image_path.suffix] + suffixes

    # Case A: data_root/rgb/xxx.png -> data_root/mask/xxx.png.
    if image_rel_to_root is not None and len(image_rel_to_root.parts) >= 2:
        first = image_rel_to_root.parts[0].lower()
        tail = Path(*image_rel_to_root.parts[1:])
        if first in RGB_DIR_NAMES:
            for mask_dir in MASK_DIR_NAMES:
                for ext in image_suffix_candidates:
                    candidates.append(data_root / mask_dir / tail.with_suffix(ext))

    # Case B: sibling mask folder beside the image folder.
    parent = image_path.parent
    grandparent = parent.parent
    for mask_dir in MASK_DIR_NAMES:
        for ext in image_suffix_candidates:
            candidates.append(grandparent / mask_dir / f"{stem}{ext}")
            candidates.append(data_root / mask_dir / f"{stem}{ext}")

    # Case C: same folder with conventional suffix/prefix.
    for ext in image_suffix_candidates:
        candidates.append(parent / f"{stem}_mask{ext}")
        candidates.append(parent / f"{stem}_seg{ext}")
        candidates.append(parent / f"mask_{stem}{ext}")
        candidates.append(parent / f"seg_{stem}{ext}")

    return _dedupe_paths(candidates)


def _find_matching_mask(
    json_path: Path,
    meta: Dict,
    image_path: Path,
    data_root: Path,
    mask_index: Dict[str, Path],
) -> Optional[Path]:
    for candidate in _candidate_mask_paths_from_meta(meta, json_path, image_path, data_root):
        if _is_image_file(candidate):
            return candidate.resolve()
    stem = image_path.stem.lower()
    return mask_index.get(stem)


def _safe_relpath(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stat_signature(paths: Sequence[Path]) -> Dict[str, float]:
    latest_mtime = 0.0
    for path in paths:
        try:
            latest_mtime = max(latest_mtime, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return {"count": len(paths), "latest_mtime": latest_mtime}


def _load_cached_samples(
    cache_path: Path,
    data_root: Path,
    label_mode: str,
    json_files: Sequence[Path],
    require_mask: bool,
) -> Optional[List[SampleRecord]]:
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    expected_signature = _stat_signature(json_files)
    meta = payload.get("meta", {})
    if int(meta.get("version", 0)) < 5:
        return None
    if meta.get("label_mode") != label_mode:
        return None
    if bool(meta.get("require_mask", True)) != bool(require_mask):
        return None
    if meta.get("json_count") != expected_signature["count"]:
        return None
    if abs(float(meta.get("latest_json_mtime", -1.0)) - expected_signature["latest_mtime"]) > 1e-6:
        return None

    samples: List[SampleRecord] = []
    for item in payload.get("samples", []):
        image_path = Path(item["image_relpath"])
        json_path = Path(item["json_relpath"])
        mask_rel = item.get("mask_relpath", "")
        mask_path = Path(mask_rel) if mask_rel else None
        if not image_path.is_absolute():
            image_path = data_root / image_path
        if not json_path.is_absolute():
            json_path = data_root / json_path
        if mask_path is not None and not mask_path.is_absolute():
            mask_path = data_root / mask_path
        if require_mask and (mask_path is None or not mask_path.is_file()):
            return None
        samples.append(
            SampleRecord(
                image_path=image_path.resolve(),
                json_path=json_path.resolve(),
                mask_path=None if mask_path is None else mask_path.resolve(),
                translation=np.asarray(item["translation"], dtype=np.float32),
                quaternion_xyzw=np.asarray(item["quaternion_xyzw"], dtype=np.float32),
                rotation_matrix=np.asarray(item["rotation_matrix"], dtype=np.float32),
                pose_name=item["pose_name"],
                translation_unit=item["translation_unit"],
                phase=item.get("phase", "unknown"),
                global_index=int(item.get("global_index", -1)),
                extrinsic_group_id=str(item.get("extrinsic_group_id", "")),
            )
        )
    return samples


def _save_cached_samples(
    cache_path: Path,
    data_root: Path,
    label_mode: str,
    json_files: Sequence[Path],
    samples: Sequence[SampleRecord],
    require_mask: bool,
) -> None:
    signature = _stat_signature(json_files)
    payload = {
        "meta": {
            "version": 5,
            "project": "eye_in_hand_T_EC_rgb_mask",
            "label_mode": label_mode,
            "require_mask": bool(require_mask),
            "json_count": signature["count"],
            "latest_json_mtime": signature["latest_mtime"],
            "generated_at": time.time(),
        },
        "samples": [
            {
                "image_relpath": _safe_relpath(sample.image_path, data_root),
                "mask_relpath": _safe_relpath(sample.mask_path, data_root),
                "json_relpath": _safe_relpath(sample.json_path, data_root),
                "translation": sample.translation.tolist(),
                "quaternion_xyzw": sample.quaternion_xyzw.tolist(),
                "rotation_matrix": sample.rotation_matrix.tolist(),
                "pose_name": sample.pose_name,
                "translation_unit": sample.translation_unit,
                "phase": sample.phase,
                "global_index": sample.global_index,
                "extrinsic_group_id": sample.extrinsic_group_id,
            }
            for sample in samples
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, cache_path)


def discover_samples(
    data_dir: str,
    label_mode: str = "tec",
    cache_path: Optional[str] = None,
    require_mask: bool = True,
) -> List[SampleRecord]:
    data_root = Path(data_dir).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    json_files = sorted(
        path for path in data_root.rglob("*.json")
        if not path.name.startswith(".sample_cache_")
        and path.name not in {"dataset_summary.json", "summary.json"}
        and not (path.parent.name == "groups" and path.name.startswith("group_"))
    )
    cache_file = Path(cache_path).resolve() if cache_path else (data_root / f".sample_cache_{label_mode}_rgb_mask.json")
    cached_samples = _load_cached_samples(cache_file, data_root, label_mode, json_files, require_mask=require_mask)
    if cached_samples is not None:
        print(f"[INFO] 已加载样本缓存: {cache_file}")
        return cached_samples

    image_index = _build_image_index(data_root)
    mask_index = _build_mask_index(data_root)
    samples: List[SampleRecord] = []
    missing_images: List[str] = []
    missing_masks: List[str] = []
    skipped_json: List[str] = []

    for json_path in json_files:
        meta = _safe_read_json(json_path)
        if meta is None:
            skipped_json.append(str(json_path))
            continue

        try:
            translation, quaternion_xyzw, pose_name = extract_target_from_json(meta, label_mode=label_mode)
            translation_unit = infer_target_length_unit(meta, label_mode=label_mode)
        except (KeyError, ValueError, TypeError) as exc:
            skipped_json.append(f"{json_path}: {exc}")
            continue

        image_path = _find_matching_image(json_path, meta, data_root, image_index)
        if image_path is None:
            missing_images.append(str(json_path))
            continue

        mask_path = _find_matching_mask(json_path, meta, image_path, data_root, mask_index)
        if require_mask and mask_path is None:
            missing_masks.append(f"json={json_path}, image={image_path}")
            continue

        rotation_matrix = quat_xyzw_to_matrix_np(quaternion_xyzw).astype(np.float32)
        explicit_group = next(
            (meta.get(key) for key in ("extrinsic_group_id", "camera_extrinsic_id", "pose_group_id", "group_id") if meta.get(key) not in (None, "")),
            None,
        )
        if explicit_group is None:
            # Images captured under the same fixed T_EC receive the same stable ID.
            pose_values = np.concatenate([translation.reshape(-1), rotation_matrix.reshape(-1)])
            pose_key = ",".join(f"{float(value):.5f}" for value in pose_values)
            explicit_group = "pose_" + hashlib.sha1(pose_key.encode("utf-8")).hexdigest()[:12]
        samples.append(
            SampleRecord(
                image_path=image_path,
                json_path=json_path.resolve(),
                mask_path=None if mask_path is None else mask_path.resolve(),
                translation=translation.astype(np.float32),
                quaternion_xyzw=quaternion_xyzw.astype(np.float32),
                rotation_matrix=rotation_matrix,
                pose_name=pose_name,
                translation_unit=translation_unit,
                phase=str(meta.get("phase", "unknown")),
                global_index=int(meta.get("global_index", -1)),
                extrinsic_group_id=str(explicit_group),
            )
        )

    if not samples:
        detail = "\n".join((skipped_json + missing_images + missing_masks)[:15])
        raise RuntimeError(
            "没有找到可用的 RGB-mask-JSON 配对样本。请确认 JSON 中含有 T_EC/T_CE，"
            "图片路径有效，并且 mask 位于 rgb 同级 mask/ 或 JSON 的 mask 字段中。"
            + (f"\n前几个跳过原因:\n{detail}" if detail else "")
        )

    if missing_images:
        print(f"[WARN] 有 {len(missing_images)} 个 JSON 没有找到 RGB 图片，已自动跳过。")
    if missing_masks:
        print(f"[WARN] 有 {len(missing_masks)} 个样本没有找到 mask，已自动跳过。")
        print("[WARN] 示例缺失 mask:", missing_masks[0])
    if skipped_json:
        print(f"[WARN] 有 {len(skipped_json)} 个 JSON 不是有效样本或缺少 T_EC/T_CE，已自动跳过。")

    _save_cached_samples(cache_file, data_root, label_mode, json_files, samples, require_mask=require_mask)
    print(f"[INFO] 已生成样本缓存: {cache_file}")
    return samples


def split_samples(
    samples: Sequence[SampleRecord],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    """Split by extrinsic group so frames sharing one T_EC never leak across splits."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    samples = list(samples)
    if val_ratio == 0:
        return samples, []
    if len(samples) < 2:
        return samples, []
    groups: Dict[str, List[SampleRecord]] = {}
    for index, sample in enumerate(samples):
        group_id = sample.extrinsic_group_id or f"ungrouped_{index:08d}"
        groups.setdefault(group_id, []).append(sample)
    if len(groups) < 2:
        print("[WARN] 只检测到一个外参组，无法生成独立验证组；所有样本用于训练。")
        return samples, []
    rng = random.Random(seed)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    val_group_count = max(1, int(round(len(group_ids) * val_ratio)))
    val_group_count = min(val_group_count, len(group_ids) - 1)
    val_group_ids = set(group_ids[:val_group_count])
    train_samples = [sample for group_id in group_ids if group_id not in val_group_ids for sample in groups[group_id]]
    val_samples = [sample for group_id in group_ids if group_id in val_group_ids for sample in groups[group_id]]
    return train_samples, val_samples


def compute_translation_stats(samples: Sequence[SampleRecord]) -> Tuple[np.ndarray, np.ndarray]:
    translations = np.stack([s.translation for s in samples], axis=0).astype(np.float32)
    mean = translations.mean(axis=0)
    std = translations.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def infer_image_size_from_samples(
    samples: Sequence[SampleRecord],
    fallback: Tuple[int, int] = (480, 640),
) -> Tuple[int, int]:
    size_counter: Counter = Counter()
    seen_paths = set()
    for sample in samples:
        image_path = sample.image_path.resolve()
        if image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        try:
            with Image.open(image_path) as image_file:
                width, height = image_file.size
        except (OSError, FileNotFoundError):
            continue
        size_counter[(height, width)] += 1
    if not size_counter:
        return fallback
    return max(size_counter.items(), key=lambda item: (item[1], item[0][0] * item[0][1]))[0]


def build_datasets(
    data_dir: str,
    label_mode: str = "tec",
    image_size: Optional[Tuple[int, int]] = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    cache_path: Optional[str] = None,
    require_mask: bool = True,
    mask_threshold: float = 0.5,
    mask_invert: bool = False,
    mask_binary: bool = True,
) -> Tuple[EyeInHandPoseMaskDataset, EyeInHandPoseMaskDataset, Dict]:
    all_samples = discover_samples(
        data_dir=data_dir,
        label_mode=label_mode,
        cache_path=cache_path,
        require_mask=require_mask,
    )
    if len({s.translation_unit for s in all_samples}) != 1:
        raise ValueError("All annotations must use the same translation unit")
    if len({s.pose_name for s in all_samples}) != 1:
        raise ValueError("All annotations must use the same pose direction")
    if image_size is None:
        image_size = infer_image_size_from_samples(all_samples)
    train_samples, val_samples = split_samples(all_samples, val_ratio=val_ratio, seed=seed)
    translation_mean, translation_std = compute_translation_stats(train_samples)

    train_dataset = EyeInHandPoseMaskDataset(
        samples=train_samples,
        translation_mean=translation_mean,
        translation_std=translation_std,
        image_size=image_size,
        train=True,
        require_mask=require_mask,
        mask_threshold=mask_threshold,
        mask_invert=mask_invert,
        mask_binary=mask_binary,
    )
    val_dataset = EyeInHandPoseMaskDataset(
        samples=val_samples,
        translation_mean=translation_mean,
        translation_std=translation_std,
        image_size=image_size,
        train=False,
        require_mask=require_mask,
        mask_threshold=mask_threshold,
        mask_invert=mask_invert,
        mask_binary=mask_binary,
    )

    pose_name = train_samples[0].pose_name if train_samples else (val_samples[0].pose_name if val_samples else "unknown")
    translation_unit = train_samples[0].translation_unit if train_samples else (val_samples[0].translation_unit if val_samples else "unknown")
    phase_counts: Dict[str, int] = {}
    for sample in all_samples:
        phase_counts[sample.phase] = phase_counts.get(sample.phase, 0) + 1

    num_with_mask = sum(1 for sample in all_samples if sample.mask_path is not None)
    train_group_ids = sorted({sample.extrinsic_group_id for sample in train_samples})
    val_group_ids = sorted({sample.extrinsic_group_id for sample in val_samples})
    metadata = {
        "project": "eye_in_hand_T_EC_rgb_mask",
        "num_total": len(all_samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_with_mask": num_with_mask,
        "require_mask": require_mask,
        "mask_threshold": mask_threshold,
        "mask_invert": mask_invert,
        "mask_binary": mask_binary,
        "translation_mean": translation_mean.tolist(),
        "translation_std": translation_std.tolist(),
        "translation_unit": translation_unit,
        "label_mode": label_mode,
        "pose_name": pose_name,
        "image_size": list(image_size),
        "phase_counts": phase_counts,
        "split_strategy": "extrinsic_group_id",
        "num_train_groups": len(train_group_ids),
        "num_val_groups": len(val_group_ids),
        "train_group_ids": train_group_ids,
        "val_group_ids": val_group_ids,
    }
    return train_dataset, val_dataset, metadata
