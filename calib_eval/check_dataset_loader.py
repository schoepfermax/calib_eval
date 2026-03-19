import os
from universal_dataset_loader import UniversalCalibrationDataset

def check(split: str):
    ds = UniversalCalibrationDataset(
        dataset_root=os.path.expanduser("~/dataset_root/static_mount_h1"),
        split=split,
        combined_index=False,
    )
    print(f"[OK] split={split} -> n={len(ds)}")
    if len(ds) == 0:
        return
    sample = ds.load_sample(0)
    print(" sample_key:", sample["sample_key"])
    print(" image shape:", sample["image"].shape)
    print(" points shape:", sample["points"].shape)
    print(" has_gt_extrinsics:", sample["has_gt_extrinsics"])

if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        check(s)
