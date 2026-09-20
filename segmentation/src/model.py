from __future__ import annotations

from typing import Dict, List

from transformers import SegformerForSemanticSegmentation


def build_segformer(model_cfg: Dict) -> SegformerForSemanticSegmentation:
    num_classes = int(model_cfg.get("num_classes", 2))
    class_names: List[str] = list(model_cfg.get("class_names", [str(i) for i in range(num_classes)]))
    if len(class_names) != num_classes:
        raise ValueError("len(class_names) must match num_classes")

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg.get("name", "nvidia/segformer-b1-finetuned-ade-512-512"),
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        cache_dir=model_cfg.get("cache_dir") or None,
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    return model
