"""Regression tests use artificial images/labels, never experimental data."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from rehc_former.data import build_datasets, split_samples
from rehc_former.factory import VARIANTS, build_model, load_model
from rehc_former.geometry import rotation_9d_to_matrix
from rehc_former.losses import PoseLoss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def create_data(root):
    for directory in ("rgb", "mask", "anno"):
        (root/directory).mkdir(parents=True)
    for group in range(4):
        for frame in range(2):
            name = f"g{group}_f{frame}"
            rgb = np.full((64, 96, 3), 60 + group * 30, dtype=np.uint8)
            mask = np.zeros((64, 96), dtype=np.uint8)
            mask[24:56, 32:64] = 255
            Image.fromarray(rgb).save(root/'rgb'/f'{name}.png')
            Image.fromarray(mask).save(root/'mask'/f'{name}.png')
            transform = np.eye(4)
            transform[:3, 3] = [group * 10, group * 5, 100 + group]
            annotation = {"image": f"rgb/{name}.png", "mask": f"mask/{name}.png",
                          "T_EC": transform.tolist(), "T_EC_unit": "mm", "extrinsic_group_id": str(group)}
            (root/'anno'/f'{name}.json').write_text(json.dumps(annotation), encoding='utf-8')


def run_cli(*args):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


class PoseTests(unittest.TestCase):
    def test_all_models_backward_and_roundtrip(self):
        image, mask = torch.randn(2, 3, 64, 96), torch.rand(2, 1, 64, 96)
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                model = build_model(variant, image_size=(64, 96), embed_dim=32,
                                    stem_width=8, depth=1, dropout=0.0)
                out = model(image, mask)
                loss = PoseLoss()(out['pred_translation_norm'], torch.ones(2, 3),
                                  out['pred_rotation_matrix'], torch.eye(3).repeat(2, 1, 1),
                                  out['pred_rotation_9d'], torch.eye(3).reshape(1, 9).repeat(2, 1))
                loss['loss'].backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
                rotations = out['pred_rotation_matrix'].detach()
                torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), torch.eye(3).expand_as(rotations), atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2), atol=1e-5, rtol=1e-5)
                checkpoint = {"model_state": model.state_dict(), "model_config": model.get_init_config(),
                              "model_variant": variant, "translation_mean": [10., 20., 30.],
                              "translation_std": [1., 2., 3.], "translation_unit": "mm"}
                path = Path(directory)/'model.pt'
                torch.save(checkpoint, path)
                restored, _ = load_model(path)
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored(image, mask)['pred_translation_norm'], model(image, mask)['pred_translation_norm'])
                    if variant == 'rgb_only':
                        torch.testing.assert_close(model(image, mask)['pred_translation_norm'], model(image, 1-mask)['pred_translation_norm'])
                    if variant == 'mask_only':
                        torch.testing.assert_close(model(image, mask)['pred_translation_norm'], model(image+1, mask)['pred_translation_norm'])

    def test_rotation_reflection_correction(self):
        raw = torch.diag(torch.tensor([1., 1., -1.])).reshape(1, 9)
        projected = rotation_9d_to_matrix(raw)
        torch.testing.assert_close(torch.linalg.det(projected), torch.ones(1))

    def test_late_fusion_token_alignment(self):
        for legacy in (False, True):
            model = build_model('late_fusion', embed_dim=32, stem_width=8, depth=1,
                                legacy_mask_alignment=legacy).eval()
            sizes = {}
            hooks = [module.register_forward_hook(lambda m, inp, out, key=key: sizes.update({key: out.shape[-2:]}))
                     for key, module in [('rgb', model.rgb_patch_proj), ('mask', model.mask_patch_proj)]]
            with torch.no_grad():
                model(torch.randn(2, 3, 64, 96), torch.ones(2, 1, 64, 96))
            for hook in hooks:
                hook.remove()
            self.assertEqual(sizes['rgb'] == sizes['mask'], not legacy)

    def test_group_split_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_data(root)
            train, val, meta = build_datasets(str(root), val_ratio=0.25)
            self.assertFalse(set(meta['train_group_ids']) & set(meta['val_group_ids']))
            self.assertEqual(len(train)+len(val), 8)
            expected = np.stack([s.translation for s in train.samples]).mean(axis=0)
            np.testing.assert_allclose(meta['translation_mean'], expected)
            no_val, empty = split_samples(train.samples, val_ratio=0)
            self.assertEqual(len(no_val), len(train))
            self.assertEqual(empty, [])
            with self.assertRaises(ValueError):
                split_samples(train.samples, val_ratio=1)

    def test_train_predict_evaluate_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_data(root/'data')
            run_cli('train.py', '--data_dir', root/'data', '--output_dir', root/'run',
                    '--epochs', '1', '--batch_size', '2', '--num_workers', '0',
                    '--embed_dim', '32', '--stem_width', '8', '--depth', '1', '--dropout', '0')
            checkpoint = root/'run/best_infer_only.pt'
            self.assertTrue(checkpoint.exists())
            self.assertTrue((root/'run/split_manifest.csv').exists())
            run_cli('predict.py', '--image', root/'data/rgb/g0_f0.png', '--mask', root/'data/mask/g0_f0.png',
                    '--checkpoint', checkpoint, '--output', root/'prediction.json')
            prediction = json.loads((root/'prediction.json').read_text())
            np.testing.assert_allclose(np.asarray(prediction['T_EC']) @ np.asarray(prediction['T_CE']), np.eye(4), atol=1e-4)
            np.save(root/'points.npy', np.array([[0., 0., 0.], [10., 0., 0.]]))
            run_cli('evaluate.py', '--data-dir', root/'data', '--checkpoint', checkpoint,
                    '--output-dir', root/'evaluation', '--points', root/'points.npy')
            summary = json.loads((root/'evaluation/summary.json').read_text())
            self.assertEqual(summary['num_images'], 8)
            self.assertEqual(summary['num_extrinsic_groups'], 4)
            self.assertTrue(np.isfinite(summary['ead_mm_mean']))
            self.assertIn('p95', summary['translation_mm'])


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('transformers', 'cv2', 'albumentations', 'yaml')), 'optional segmentation dependencies unavailable')
class SegmentationTests(unittest.TestCase):
    def test_hole_fill_with_foreground_corner(self):
        from segmentation.infer import fill_holes
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 1
        mask[3:6, 3:6] = 1
        mask[4, 4] = 0
        result = fill_holes(mask)
        self.assertEqual(result[7, 7], 0)
        self.assertEqual(result[4, 4], 1)
        self.assertEqual(result.sum(), 10)

    def test_automatic_mask_pipeline_offline(self):
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_data(root/'data')
            cfg = SegformerConfig(num_labels=2, depths=[1,1,1,1], hidden_sizes=[8,16,32,32],
                                  num_attention_heads=[1,2,4,4], decoder_hidden_size=16)
            seg = SegformerForSemanticSegmentation(cfg).eval()
            with torch.no_grad():
                seg.decode_head.classifier.weight.zero_()
                seg.decode_head.classifier.bias.copy_(torch.tensor([0., 5.]))
            seg.save_pretrained(root/'seg')
            preprocessing = {'input': {'image_size': [64,96], 'normalize_mean': [0.485,0.456,0.406],
                                      'normalize_std': [0.229,0.224,0.225]},
                             'infer': {'threshold': 0.5, 'fill_holes': True, 'min_area': 0}}
            (root/'seg.yaml').write_text(yaml.safe_dump(preprocessing))
            pose = build_model(embed_dim=32, stem_width=8, depth=1, image_size=(64,96))
            torch.save({'model_state': pose.state_dict(), 'model_config': pose.get_init_config(),
                        'translation_mean': [0.,0.,0.], 'translation_std': [1.,1.,1.],
                        'translation_unit': 'mm', 'label_mode': 'tec'}, root/'pose.pt')
            run_cli('predict.py', '--image', root/'data/rgb/g0_f0.png', '--seg-model', root/'seg',
                    '--seg-config', root/'seg.yaml', '--checkpoint', root/'pose.pt',
                    '--output', root/'pred.json', '--save-mask', root/'mask.png')
            result = json.loads((root/'pred.json').read_text())
            self.assertEqual(result['mask_source'], 'segmentation')
            self.assertTrue((root/'mask.png').exists())


if __name__ == '__main__':
    unittest.main()
