import argparse, os, random, pathlib, json
from glob import glob
import numpy as np

NPZ_EXT = (".npz",)

# 常见键名候选（不会强制全部存在，只做提示/统计）
IMAGE_KEY_CANDIDATES = ["image", "img", "images", "image_0", "vol", "data"]
MASK_KEY_CANDIDATES  = ["mask", "label", "labels", "seg", "segmentation"]

def light_probe_npz_keys(path, max_items=1):
    """轻量检查：读取少量条目（默认1个文件）以推断常用键名。"""
    try:
        with np.load(path, mmap_mode="r") as z:
            return sorted(list(z.files))
    except Exception:
        return []

def infer_common_keys(sample_keys_list):
    """从若干文件的键名中推断 image/mask 最可能的键（仅用于 meta 记录，不做强约束）"""
    def pick(cands, keys):
        for c in cands:
            if c in keys:
                return c
        return None
    image_keys, mask_keys = set(), set()
    for keys in sample_keys_list:
        kset = set(keys)
        ik = pick(IMAGE_KEY_CANDIDATES, kset)
        mk = pick(MASK_KEY_CANDIDATES,  kset)
        if ik: image_keys.add(ik)
        if mk: mask_keys.add(mk)
    return sorted(list(image_keys)), sorted(list(mask_keys))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="MedSegBench 根目录，包含多个 .npz")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recursive", action="store_true", help="递归搜索 **/*.npz")
    ap.add_argument("--require_keys", type=str, default="",
                    help="可选，逗号分隔的必须包含的键名，例如: image,mask")
    ap.add_argument("--probe_files", type=int, default=8,
                    help="抽样若干文件进行键名探测与统计（不会影响切分）")
    args = ap.parse_args()

    root = pathlib.Path(args.data_root)
    pattern = "**/*.npz" if args.recursive else "*.npz"
    npz_files = sorted([str(p) for p in root.glob(pattern)])

    assert len(npz_files) > 0, f"No .npz files found under {root} (recursive={args.recursive})"

    # 过滤：若指定 require_keys，则仅保留包含这些键的 .npz
    req_keys = [k.strip() for k in args.require_keys.split(",") if k.strip()]
    if req_keys:
        kept, dropped = [], []
        for f in npz_files:
            keys = light_probe_npz_keys(f)
            if all(k in keys for k in req_keys):
                kept.append(f)
            else:
                dropped.append(f)
        if len(kept) == 0:
            raise AssertionError(f"No .npz contains all required keys: {req_keys}")
        if dropped:
            print(f"[Info] {len(dropped)} files dropped due to missing required keys.")
        npz_files = kept

    # 抽样探测常见键（仅统计信息）
    sample_paths = npz_files[:max(1, min(args.probe_files, len(npz_files)))]
    sample_keys_list = [light_probe_npz_keys(p) for p in sample_paths]
    image_keys, mask_keys = infer_common_keys(sample_keys_list)

    # 打乱与切分
    rng = random.Random(args.seed)
    rng.shuffle(npz_files)

    n = len(npz_files)
    n_test = int(n * args.test_ratio)
    n_val  = int(n * args.val_ratio)
    test_paths = npz_files[:n_test]
    val_paths  = npz_files[n_test:n_test+n_val]
    train_paths= npz_files[n_test+n_val:]

    split_dir = root / "splits"
    os.makedirs(split_dir, exist_ok=True)

    # 写相对路径（相对 data_root），更便于跨环境使用
    def rel(p): 
        try:
            return str(pathlib.Path(p).relative_to(root))
        except Exception:
            return str(p)  # 兜底用绝对路径

    def write_list(name, paths):
        with open(split_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for p in paths:
                f.write(rel(p) + "\n")

    write_list("train", train_paths)
    write_list("val",   val_paths)
    write_list("test",  test_paths)

    meta = {
        "num_samples": n,
        "train": len(train_paths),
        "val":   len(val_paths),
        "test":  len(test_paths),
        "data_root": str(root),
        "recursive": bool(args.recursive),
        "required_keys": req_keys,
        "sampled_key_stats": {
            "image_key_candidates_found": image_keys,
            "mask_key_candidates_found":  mask_keys,
            "examples_probed": len(sample_paths)
        }
    }
    with open(split_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done. Splits written to", str(split_dir))
    print(meta)

if __name__ == "__main__":
    main()