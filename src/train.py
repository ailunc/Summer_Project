import os, argparse, time, yaml, random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import UNet
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandGaussianNoised, EnsureTyped, EnsureType,
    AsDiscrete, Invertd, SaveImaged
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from monai.data.utils import pad_list_data_collate
import nibabel as nib

def parse_kv_overrides(kv_list):
    cfg_over = {}
    for kv in kv_list or []:
        if "=" not in kv: 
            continue
        k, v = kv.split("=", 1)
        # try to parse simple types
        try:
            cfg_over[k] = yaml.safe_load(v)
        except Exception:
            cfg_over[k] = v
    return cfg_over

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    ov = parse_kv_overrides(args.overrides)
    deep_update(cfg, ov)

    seed = cfg.get("seed", 42)
    if cfg.get("deterministic", True):
        set_determinism(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["root"]
    work_dir = cfg["logging"]["work_dir"]
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)

    writer = SummaryWriter(cfg["logging"]["log_dir"])

    # Build file lists from split txt inside data_root/splits
    def read_ids(path):
        if not os.path.isabs(path):
            path = os.path.join(data_root, "splits", os.path.basename(path))
        with open(path, "r") as f:
            return [l.strip() for l in f if l.strip()]

    train_ids = read_ids(cfg["data"]["train_list"])
    val_ids   = read_ids(cfg["data"]["val_list"])

    def to_items(ids):
        items = []
        for sid in ids:
            items.append({
                "image": os.path.join(data_root, "imagesTr", f"{sid}.nii.gz"),
                "label": os.path.join(data_root, "labelsTr", f"{sid}.nii.gz"),
                "id": sid
            })
        return items

    train_files = to_items(train_ids)
    val_files = to_items(val_ids)

    ps = cfg["data"]["patch_size"]
    tr_transforms = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes=cfg["data"]["orientation"]),
        Spacingd(keys=["image","label"], pixdim=cfg["data"]["pixdim"], mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["data"]["a_min"], a_max=cfg["data"]["a_max"], b_min=cfg["data"]["b_min"], b_max=cfg["data"]["b_max"], clip=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image","label"], label_key="label", spatial_size=ps, pos=1, neg=1, num_samples=2),
        RandFlipd(keys=["image","label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image","label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image","label"], prob=0.2, spatial_axis=2),
        RandRotate90d(keys=["image","label"], prob=0.2, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=["image","label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes=cfg["data"]["orientation"]),
        Spacingd(keys=["image","label"], pixdim=cfg["data"]["pixdim"], mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=cfg["data"]["a_min"], a_max=cfg["data"]["a_max"], b_min=cfg["data"]["b_min"], b_max=cfg["data"]["b_max"], clip=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        EnsureTyped(keys=["image","label"]),
    ])

    train_ds = CacheDataset(train_files, transform=tr_transforms, cache_rate=cfg["data"]["cache_rate"], num_workers=cfg["data"]["num_workers"])
    val_ds   = CacheDataset(val_files, transform=val_transforms, cache_rate=0.0, num_workers=cfg["data"]["num_workers"])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=pad_list_data_collate)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=pad_list_data_collate)

    model = UNet(
        spatial_dims=3,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=cfg["model"]["channels"],
        strides=cfg["model"]["strides"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"], betas=tuple(cfg["optim"]["betas"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["optim"]["scheduler"]["t_max"], eta_min=cfg["optim"]["scheduler"]["eta_min"])

    dice_metric = DiceMetric(include_background=False, reduction="none")

    best_mean_dice = -1
    best_ckpt = os.path.join(cfg["logging"]["ckpt_dir"], "best_metric_model.pt")
    results = []

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["trainer"]["amp"])

    for epoch in range(1, cfg["trainer"]["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["trainer"]["amp"]):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= max(1, len(train_loader))
        writer.add_scalar("train/loss", epoch_loss, epoch)

        if epoch % cfg["trainer"]["val_every"] == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    with torch.cuda.amp.autocast(enabled=cfg["trainer"]["amp"]):
                        logits = model(images)
                        probs = torch.softmax(logits, dim=1)
                        pred = torch.argmax(probs, dim=1, keepdim=True)  # [B,1,D,H,W]
                        # one-hot for metric
                        pred_oh = torch.nn.functional.one_hot(pred.long().squeeze(1), num_classes=cfg["data"]["num_classes"]).permute(0,4,1,2,3).float()
                        labels_oh = torch.nn.functional.one_hot(labels.long().squeeze(1), num_classes=cfg["data"]["num_classes"]).permute(0,4,1,2,3).float()
                        # exclude background for metric
                        if pred_oh.shape[1] > 1:
                            dice_metric(y_pred=pred_oh[:,1:], y=labels_oh[:,1:])

            class_dice = dice_metric.aggregate().cpu().numpy() if dice_metric not in (None, ) else np.array([])
            if class_dice.size > 0:
                mean_dice = float(class_dice.mean())
            else:
                mean_dice = float("nan")

            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            for ci, dc in enumerate(class_dice, start=1):
                writer.add_scalar(f"val/dice_class_{ci}", float(dc), epoch)

            results.append({"epoch": epoch, "train_loss": epoch_loss, "mean_dice": mean_dice})
            pd.DataFrame(results).to_csv(cfg["logging"]["results_csv"], index=False)

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                torch.save(model.state_dict(), best_ckpt)

            print(f"Epoch {epoch:03d} | loss {epoch_loss:.4f} | meanDice {mean_dice:.4f} | best {best_mean_dice:.4f}")

    print("Training done. Best model @", best_ckpt, "best mean dice:", best_mean_dice)

if __name__ == "__main__":
    main()
