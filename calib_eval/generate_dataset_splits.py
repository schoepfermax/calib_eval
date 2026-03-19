###############################################
# DATASET SPLITTING & INDEX GENERATION
###############################################

import os
import re
import random
from typing import List, Tuple, Dict


def _is_run_folder(path: str) -> bool:
    """
    A "run folder" is considered valid if it has at least:
      - images/
      - lidar/
    """
    return (
        os.path.isdir(os.path.join(path, "images")) and
        os.path.isdir(os.path.join(path, "lidar"))
    )


def _discover_run_folders(parent_dir: str) -> List[str]:
    """
    Finds run folders under parent_dir matching run_###.
    Returns a sorted list of absolute paths.
    """
    run_dirs: List[str] = []
    pat = re.compile(r"^run_(\d{3})$")

    if not os.path.isdir(parent_dir):
        return run_dirs

    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if os.path.isdir(full) and pat.match(name) and _is_run_folder(full):
            run_dirs.append(full)

    # Sort numerically by run index
    run_dirs.sort(key=lambda p: int(os.path.basename(p).split("_")[1]))
    return run_dirs


def _list_sample_ids_from_images(run_dir: str) -> List[str]:
    """
    Lists sample IDs by scanning images/*.png.
    We assume filename stems are the sample IDs (e.g. 000123).
    """
    image_dir = os.path.join(run_dir, "images")
    if not os.path.isdir(image_dir):
        return []
    ids = []
    for f in os.listdir(image_dir):
        if f.endswith(".png"):
            ids.append(os.path.splitext(f)[0])
    return sorted(ids)


def _validate_samples(run_dir: str, sample_ids: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Filters sample IDs to keep only those with image + lidar present.

    Returns:
      valid_ids: List[str]
      stats: dict with missing counts (for logging)
    """
    img_dir = os.path.join(run_dir, "images")
    lidar_dir = os.path.join(run_dir, "lidar")
    gt_dir = os.path.join(run_dir, "extrinsics_gt")

    missing_img = 0
    missing_lidar = 0
    missing_gt = 0

    valid: List[str] = []

    for sid in sample_ids:
        img_path = os.path.join(img_dir, f"{sid}.png")
        lidar_path = os.path.join(lidar_dir, f"{sid}.pcd")
        gt_path = os.path.join(gt_dir, f"{sid}.yaml")

        ok = True

        if not os.path.exists(img_path):
            missing_img += 1
            ok = False

        if not os.path.exists(lidar_path):
            missing_lidar += 1
            ok = False

        # GT extrinsics are OPTIONAL (supervised may have them, KITTI conversion may not).
        if os.path.isdir(gt_dir) and (not os.path.exists(gt_path)):
            missing_gt += 1

        if ok:
            valid.append(sid)

    stats = {
        "missing_img": missing_img,
        "missing_lidar": missing_lidar,
        "missing_gt_yaml_optional": missing_gt,
    }
    return valid, stats


def _write_split(index_dir: str, name: str, split_ids: List[str]) -> None:
    """
    Writes split IDs into index/<name>.txt, one ID per line.
    """
    os.makedirs(index_dir, exist_ok=True)
    out_path = os.path.join(index_dir, f"{name}.txt")
    with open(out_path, "w") as f:
        for sid in split_ids:
            f.write(sid + "\n")


def generate_splits(
    dataset_root: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Generates train/val/test splits for:
      - a single run folder OR
      - a parent folder containing run_### subfolders

    Splits are deterministic due to `seed`.
    """
    dataset_root = os.path.expanduser(dataset_root)

    # -----------------------------
    # Step 1: Resolve runs
    # -----------------------------
    if _is_run_folder(dataset_root):
        run_dirs = [dataset_root]
    else:
        run_dirs = _discover_run_folders(dataset_root)

    if not run_dirs:
        raise RuntimeError(
            f"No valid run folders found under: {dataset_root}\n"
            f"Expected either a run folder (images/, lidar/) or a parent folder with run_### subfolders."
        )

    rng = random.Random(seed)

    # -----------------------------
    # Step 2: Process each run
    # -----------------------------
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        # (a) Discover candidate IDs from images/
        candidate_ids = _list_sample_ids_from_images(run_dir)

        # (b) Validate that lidar exists
        valid_ids, stats = _validate_samples(run_dir, candidate_ids)

        n_total = len(valid_ids)
        if n_total == 0:
            print(f"[WARN] {run_name}: 0 valid samples after validation. Skipping.")
            continue

        # (c) Deterministic shuffle
        ids_shuffled = valid_ids[:]
        rng.shuffle(ids_shuffled)

        # (d) Split
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        train_set = ids_shuffled[:n_train]
        val_set = ids_shuffled[n_train:n_train + n_val]
        test_set = ids_shuffled[n_train + n_val:]

        # (e) Write
        index_dir = os.path.join(run_dir, "index")
        _write_split(index_dir, "train", train_set)
        _write_split(index_dir, "val", val_set)
        _write_split(index_dir, "test", test_set)

        # (f) Report
        print(
            f"[OK] {run_name}: split completed "
            f"(total_valid={n_total}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}), "
            f"missing(img={stats['missing_img']}, lidar={stats['missing_lidar']}, gt_yaml_optional={stats['missing_gt_yaml_optional']})"
        )


if __name__ == "__main__":

    generate_splits(
        dataset_root="~/dataset_root/static_mount_h1",
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )
