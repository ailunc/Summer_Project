import os, argparse, yaml, time, numpy as np, pandas as pd
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.utils import set_determinism
from torchvision import transforms as T
from PIL import Image

def parse_kv_overrides(kv_list):
    cfg_over = {}
    for kv in kv_list or []:
        if "=" not in kv: 
            continue
        k, v = kv.split("=", 1)
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

def load_ids(list_path):
    with open(list_path, "r") as f:
        return [l.strip() for l in f if l.strip()]

class Seg2DDataset(Dataset):
    def __init__(self, root, ids, split='train', size=(512,512), rgb=False, mean=[0.5], std=[0.5], aug=False):
        self.root = root
        self.ids = ids
        self.split = split
        self.size = tuple(size)
        self.rgb = rgb
        self.mean = mean
        self.std = std
        self.aug = aug

        self.images = []
        self.masks = []
        
        img_key = f"{self.split}_images"
        lbl_key = f"{self.split}_label"

        for npz_id in self.ids:
            npz_path = os.path.join(self.root, npz_id)
            with np.load(npz_path) as npz:
                if img_key in npz and lbl_key in npz:
                    self.images.extend(npz[img_key])
                    self.masks.extend(npz[lbl_key])

        assert len(self.images) > 0, f"No images found for split '{self.split}' with key '{img_key}'"
        assert len(self.images) == len(self.masks), "Mismatch between number of images and masks"
        
        self.tf_resize = T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR)
        self.tf_resize_mask = T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST)
        self.flip = T.RandomHorizontalFlip(p=0.5) if aug else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, msk = self.images[idx], self.masks[idx]

        # from numpy to PIL
        img = Image.fromarray(img).convert("RGB" if self.rgb else "L")
        msk = Image.fromarray(msk)

        # resize
        img = self.tf_resize(img)
        msk = self.tf_resize_mask(msk)

        if self.aug:
            seed = torch.seed()
            torch.manual_seed(seed); img = self.flip(img)
            torch.manual_seed(seed); msk = self.flip(msk)

        img = T.ToTensor()(img)
        img = T.Normalize(mean=self.mean, std=self.std)(img)
        msk = torch.from_numpy(np.array(msk, dtype=np.int64))[None, ...] # Add channel dim
        return img, msk

def build_model(in_ch, out_ch, channels, strides, dropout):
    return UNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=channels,
        strides=strides,
        dropout=dropout,
    )

import re

def resolve_config_vars(config):
    """Recursively resolves placeholders like ${key.subkey} in a config dict."""
    def get_value(key_path):
        parts = key_path.split('.')
        val = config
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return None
        return val # Return value with its original type

    def resolve_item(item):
        if isinstance(item, str):
            # If the whole string is a placeholder, replace it with the typed value
            match = re.fullmatch(r'\$\{(.*?)\}', item)
            if match:
                key_path = match.group(1)
                value = get_value(key_path)
                return value if value is not None else item

            # Otherwise, do string substitution for paths etc.
            for _ in range(5): # Max 5 levels of nesting
                match = re.search(r'\$\{(.*?)\}', item)
                if not match:
                    break
                key_path = match.group(1)
                value = get_value(key_path)
                if value is not None:
                    item = item.replace(match.group(0), str(value))
                else:
                    break # Stop if a key is not found
            return item
        return item

    def resolve_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                resolve_dict(v)
            elif isinstance(v, list):
                d[k] = [resolve_item(i) for i in v]
            else:
                d[k] = resolve_item(v)

    # Multiple passes to resolve dependencies
    for _ in range(5):
        resolve_dict(config)
    return config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    ov = parse_kv_overrides(args.overrides)
    deep_update(cfg, ov)
    cfg = resolve_config_vars(cfg)

    if cfg.get("deterministic", True):
        set_determinism(seed=cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["root"]
    os.makedirs(cfg["logging"]["work_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["logging"]["log_dir"])

    train_ids = load_ids(cfg["data"]["train_list"])
    val_ids   = load_ids(cfg["data"]["val_list"])

    in_ch = 3 if cfg["data"]["rgb"] else 1
    tr_ds = Seg2DDataset(data_root, train_ids, split='train', size=cfg["data"]["size"], rgb=cfg["data"]["rgb"], mean=cfg["data"]["mean"], std=cfg["data"]["std"], aug=True)
    va_ds = Seg2DDataset(data_root, val_ids,   split='val',   size=cfg["data"]["size"], rgb=cfg["data"]["rgb"], mean=cfg["data"]["mean"], std=cfg["data"]["std"], aug=False)

    tr_loader = DataLoader(tr_ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(in_ch, cfg["model"]["out_channels"], cfg["model"]["channels"], cfg["model"]["strides"], cfg["model"]["dropout"]).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"], betas=tuple(cfg["optim"]["betas"]))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["optim"]["scheduler"]["t_max"], eta_min=cfg["optim"]["scheduler"]["eta_min"])

    dice_metric = DiceMetric(include_background=False, reduction="none")
    iou_metric  = MeanIoU(include_background=False)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["trainer"]["amp"])
    best_mean_dice = -1.0
    best_ckpt = os.path.join(cfg["logging"]["ckpt_dir"], "best_metric_model.pt")
    import pandas as pd
    results = []

    for epoch in range(1, cfg["trainer"]["max_epochs"]+1):
        model.train()
        epoch_loss = 0.0
        for imgs, msks in tr_loader:
            imgs = imgs.to(device); msks = msks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["trainer"]["amp"]):
                logits = model(imgs)
                loss = loss_fn(logits, msks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"]["grad_clip"])
            scaler.step(opt); scaler.update()
            epoch_loss += float(loss.item())
        epoch_loss /= max(1, len(tr_loader))
        writer.add_scalar("train/loss", epoch_loss, epoch)
        sch.step()

        # validation
        if epoch % cfg["trainer"]["val_every"] == 0:
            model.eval()
            dice_metric.reset(); iou_metric.reset()
            with torch.no_grad():
                for imgs, msks in va_loader:
                    imgs = imgs.to(device); msks = msks.to(device)
                    with torch.cuda.amp.autocast(enabled=cfg["trainer"]["amp"]):
                        logits = model(imgs)
                        probs = torch.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        nc = cfg["data"]["num_classes"]
                        pred_oh = torch.nn.functional.one_hot(preds, num_classes=nc).permute(0,3,1,2).float()
                        gt_oh   = torch.nn.functional.one_hot(msks,  num_classes=nc).permute(0,3,1,2).float()

                        if nc > 1:
                            dice_metric(y_pred=pred_oh[:,1:], y=gt_oh[:,1:])
                            iou_metric(y_pred=pred_oh[:,1:], y=gt_oh[:,1:])

            class_dice = dice_metric.aggregate().cpu().numpy() if dice_metric not in (None, ) else np.array([])
            mean_dice = float(class_dice.mean()) if class_dice.size>0 else float("nan")
            mean_iou  = float(iou_metric.aggregate().item()) if hasattr(iou_metric.aggregate(), "item") else float("nan")

            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            writer.add_scalar("val/mean_iou",  mean_iou,  epoch)
            for ci, dc in enumerate(class_dice, start=1):
                writer.add_scalar(f"val/dice_class_{ci}", float(dc), epoch)

            results.append({"epoch": epoch, "train_loss": epoch_loss, "mean_dice": mean_dice, "mean_iou": mean_iou})
            pd.DataFrame(results).to_csv(cfg["logging"]["results_csv"], index=False)

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                torch.save(model.state_dict(), best_ckpt)

            print(f"Epoch {epoch:03d} | loss {epoch_loss:.4f} | meanDice {mean_dice:.4f} | meanIoU {mean_iou:.4f} | best {best_mean_dice:.4f}")

    print("Training done. Best model @", best_ckpt, "best mean dice:", best_mean_dice)

if __name__ == "__main__":
    import numpy as np
    main()
