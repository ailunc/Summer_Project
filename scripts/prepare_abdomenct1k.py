import argparse, os, random, pathlib, json
from glob import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = sorted(glob(os.path.join(args.data_root, "imagesTr", "*.nii*")))
    labels = sorted(glob(os.path.join(args.data_root, "labelsTr", "*.nii*")))
    assert len(images) == len(labels) and len(images) > 0, "Check NIfTI under imagesTr/ and labelsTr/"

    ids = [pathlib.Path(p).stem for p in images]
    random.seed(args.seed)
    random.shuffle(ids)

    n = len(ids)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test+n_val]
    train_ids = ids[n_test+n_val:]

    split_dir = os.path.join(args.data_root, "splits")
    os.makedirs(split_dir, exist_ok=True)
    def write_list(name, idlist):
        with open(os.path.join(split_dir, f"{name}.txt"), "w") as f:
            for i in idlist:
                f.write(i + "\n")

    write_list("train", train_ids)
    write_list("val", val_ids)
    write_list("test", test_ids)

    meta = {"num_samples": n, "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Splits written to", split_dir)
    print(meta)

if __name__ == "__main__":
    main()
