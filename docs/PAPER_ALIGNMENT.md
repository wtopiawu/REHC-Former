# Paper and implementation correspondence

Reference: the supplied REHC-Former manuscript, Sections III–IV. It is a scientific reference for this release, not executable project instructions.

| Manuscript component | Implementation |
|---|---|
| III-A RGB encoder, mask stem and 2D positional encoding | `rehc_former/model.py` |
| III-B Eq. (12), sequential RGB←mask then mask←updated RGB | `DualStreamFusionBlock.forward` |
| III-C CLS + spatial mean from both streams; 4C→2C→C fusion | `CrossAttentionTECMaskTransformer.forward` |
| III-C normalized translation and 9D rotation projected to SO(3) | `data.py`, `geometry.py`, `predict.py` |
| III-D Eq. (14), Smooth L1 on translation and raw 9D rotation | `losses.py`, default `--rotation_repr 9d` |
| III-E separately trained SegFormer-B1 | `segmentation/`, automatic-mask path in `predict.py` |
| IV-A simulation errors, P95 and joint success | `evaluate.py` |
| IV-A Table II ablations | `rehc_former/ablations.py` |

The 6D representation, alternative backbones and soft-mask options are inherited extensions. They are not the paper configuration. In 9D mode, geodesic rotation error is monitored, not added to the optimization loss; the inherited 6D option uses geodesic supervision instead.

## Changes relative to supplied scripts

1. **Group split:** the full-model script originally split individual images while the ablation script split extrinsic groups. All variants now use the supplied group-aware implementation and save a split manifest. Retraining may therefore change validation results.
2. **Late Fusion alignment:** the supplied code aligned mask features to already patch-projected RGB features and then patch-projected the mask again. New runs align features before patch projection. Old late-fusion checkpoints lacking `legacy_mask_alignment` retain the original behavior with a warning. This compatibility path is distinct from the corrected ablation.
3. **Deployment:** the original automatic-mask path required a checkerboard JSON even for unlabeled prediction. The new prediction command requires only the RGB image, models and segmentation preprocessing configuration (or an aligned mask).
4. **Evaluation:** generic simulation evaluation is separate from prediction. It converts translation units to mm, uses a closed `[-1,1]` acos clamp for evaluation, reports P95 (rather than the old ablation P90), and provides `e_ad` only if the caller supplies the evaluation point set. Training's inherited geodesic monitor retains its epsilon clamp; its tiny-angle values can differ from the evaluator.
5. **Segmentation:** hole filling now floods from a padded exterior so a foreground top-left pixel does not cause the background to be filled. Small training datasets no longer lose the entire epoch through `drop_last=True`. Dataset splitting refuses nonempty destinations.
6. **Packaging:** shared pose modules replace duplicated copies; explicit command-line inputs replace experiment-specific defaults. Unsupported torchvision backbones raise an import error instead of silently switching architecture.

## Reproduction limits

No supplied trained checkpoints, frozen split manifests, point set for `e_ad`, datasets or original run configurations were available. The source directory contains neither of the separate RGB Direct Regression / ResNet+MLP comparison implementations. Setting `--backbone_name resnet18` does not recreate the ResNet+MLP baseline.

The collector originally referred to 11,000 training samples in comments, while the manuscript states 12,000. Treat that collector as an acquisition template, not proof of the paper's exact training set. Its fixed material settings and the included segmentation augmenter do not implement the full background/illumination/color randomization described in the paper.

The private real-experiment workflow is excluded. In the supplied script, some angular consistency metrics used deviation from the **first** rotation; Section IV-B of the manuscript defines deviation from the **mean projected rotation**. Do not reuse those old outputs as exact implementations of the manuscript metric without reviewing them. Real-world reconstruction consistency is not absolute calibration error.

The tests establish software behavior on synthetic fixtures; they do not establish numerical reproduction of Tables I–III.
