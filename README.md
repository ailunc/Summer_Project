# MedSegBench 2D Baseline (MONAI + PyTorch)

A lightweight, reproducible baseline for training/evaluating **2D medical image segmentation** on **MedSegBench** (35 datasets across ultrasound, X-ray, endoscopy, dermoscopy, MRI, CT, etc.).
This repo avoids heavyweight 3D volumes and works on a per-image basis to keep storage & compute modest.

## Why MedSegBench?
- Smaller per-dataset footprint than large 3D CT volumes (e.g., AbdomenCT-1K >100GB).
- Diverse modalities & anatomies to stress-test generalization.

## Folder Layout (Unified Format)
Prepare one MedSegBench subset at a time into the following layout (you can symlink/copy from the official release):
```
DATA_ROOT/
  images/                # *.png/*.jpg/*.tif ...
  masks/                 # same filenames as images, integer index masks
  splits/
    train.txt            # each line is a basename without extension (e.g., case_0001)
    val.txt
    test.txt             # optional
```
> Example: `images/case_0001.png` + `masks/case_0001.png`

If you don't have official splits, generate them:
```bash
python scripts/prepare_medsegbench.py --data_root Dataset/
```

## Quickstart
export GOOGLE_CLOUD_PROJECT="summerproject-471608"
### 1) Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train
```bash
python src/train2d.py --config configs/medsegbench.yaml   data_root=Dataset/   work_dir=work_dir   data.num_classes=2        # set to actual number of classes incl. background
```

### 3) Inference (single image)
```bash
python src/infer2d.py   --checkpoint work_dir/checkpoints/best_metric_model.pt   --image Dataset/images/case_0001.png   --out_mask work_dir/preds/case_0001_pred.png   --num_classes 2
```

## Notes
- Masks must be **integer label maps** with class IDs in `[0..C-1]`. Background=0.
- Update `data.num_classes` to match your subset.
- This baseline uses 2D UNet; swap in other backbones easily in `build_model()`.
- Metrics: mean Dice (exclude background), per-class Dice, mean IoU (Jaccard).

## Roadmap (optional)
- Add multi-dataset training (domain generalization)
- Add TTA & ensembling
- Add more metrics (Hausdorff95, boundary F-score)
