# Validation scope

Validated locally on Windows with Python 3.12, CPU PyTorch 2.6.0, torchvision 0.21.0, NumPy 2.3.5, Pillow 12.3.0, Transformers 4.46.3, Albumentations 1.4.3 and OpenCV 4.10.0.

The following 7 tests pass with `python -m unittest discover -s tests -v`:

- Finite backward gradients, SO(3) outputs and checkpoint round trips for all five pose variants; RGB-only/Mask-only invariance to the unused modality.
- Determinant correction for a reflected 9D rotation matrix.
- Equal RGB/mask patch-grid sizes in corrected Late Fusion and preservation of the legacy grid behavior.
- Disjoint extrinsic-group splits, training-only normalization and validation-ratio handling.
- One-epoch training on artificial samples, standalone prediction and simulation evaluation with a supplied point set.
- Segmentation hole filling when the image corner is foreground.
- Offline automatic mask generation and pose prediction with a tiny artificial SegFormer checkpoint.

All fixtures are generated in temporary directories. The last test verifies API and preprocessing integration using a tiny SegFormer architecture; it is not an accuracy or performance test of a trained SegFormer-B1. Optional segmentation tests are skipped if their dependencies are not installed.

The manuscript's full training runs, trained-checkpoint accuracy, CUDA/AMP behavior, live CoppeliaSim collection and ONNX export have not been validated. No experimental reproduction claim is made.
