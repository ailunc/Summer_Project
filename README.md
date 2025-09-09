# AbdomenCT-1K Baseline (MONAI + PyTorch)

A minimal, reproducible baseline to evaluate open‑source medical image segmentation models on **AbdomenCT‑1K** using a standard 3D UNet in MONAI.

> This follows the proposal: evaluate five open-source models/frameworks on a benchmark **not used in their original papers**. Here we implement a clean, extensible codebase for **AbdomenCT‑1K** multi‑organ segmentation.

## Features
- 3D UNet (MONAI) with Dice + Cross‑Entropy loss
- Reproducible splits (train/val/test) and deterministic seed
- Per‑class Dice + mean Dice metrics
- Config‑driven (YAML) pipeline
- Mixed precision (AMP) + gradient clipping
- Saves best model (by validation mean Dice), metrics CSV, tensorboard logs

## Dataset (AbdomenCT‑1K)
- Please obtain AbdomenCT‑1K following the dataset license.
- Expected structure after preparation (NIfTI):
```
DATA_ROOT/
  imagesTr/*.nii.gz
  labelsTr/*.nii.gz      # integer mask with class IDs
```
- Then run the provided script to create split files (`splits/train.txt`, `splits/val.txt`, `splits/test.txt`).

## Quickstart
### 1) Create environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare splits
```bash
python scripts/prepare_abdomenct1k.py   --data_root /path/to/abdomenct1k   --val_ratio 0.1 --test_ratio 0.1   --seed 42
```

### 3) Train
```bash
python src/train.py --config configs/abdomenct1k.yaml   data_root=/path/to/abdomenct1k   work_dir=/path/to/workdir
```

### 4) Validate / Inference (single volume example)
```bash
python src/infer.py   --checkpoint /path/to/workdir/checkpoints/best_metric_model.pt   --image /path/to/abdomenct1k/imagesTr/case_0001.nii.gz   --out_pred /path/to/workdir/preds/case_0001_pred.nii.gz
```

## Config overrides
Most keys in the YAML can be overridden from CLI, e.g.:
```bash
python src/train.py --config configs/abdomenct1k.yaml   trainer.max_epochs=200 optim.lr=1e-4 data.patch_size="[128,128,64]"
```

## Notes
- By default, we assume **background=0** and foreground classes starting from 1 consecutively.
- Update `num_classes` in config if your label schema differs.
- This baseline intentionally uses a single, solid UNet to provide a fair point of comparison.
- You can plug in other backbones (e.g., KM‑UNet variants) by replacing `build_model` in `src/train.py`.

## License
This baseline code is MIT licensed. The dataset is governed by its original license.
