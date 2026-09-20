# Optional CoppeliaSim acquisition

This is an adapted copy of the supplied grouped simulation collector. It requires a running CoppeliaSim instance, its ZeroMQ Remote API, and a scene containing the configured UR5, camera, mask camera and gripper tree. Scene and CAD files are not included. It only controls simulator objects.

```bash
python -m pip install -r data_collection/requirements.txt
python data_collection/collect_simulation.py --scene-config data_collection/scene.example.json --output data/simulation_test --groups 100 --images-per-group 15 --seed 42
```

Edit scene object paths and image-flip conventions in your own configuration. The example preserves the source collector's horizontal flip; verify the convention against your rendering and training images. Both cameras must have matching projection/intrinsic settings. The script writes aligned `rgb/`, `mask/`, and `anno/` samples plus group metadata, and refuses a nonempty output directory. Group IDs are local to the acquisition: namespace them before combining different sessions. Use separate seeds/configurations for training and test acquisition and check transform overlap explicitly.

Default perturbations are ±15 mm and ±5° around the current scene setup. The script changes simulated poses, materials and sensor settings. Its final pose reset does not restore all scene properties; run it on a saved working copy of your scene.

This collector has not been exercised against a live simulator during repository cleanup. It does not reproduce the full appearance-randomization or dataset-generation history of the paper.
