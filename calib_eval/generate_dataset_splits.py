###############################################
# DATASET SPLITTING & INDEX GENERATION
###############################################

import os
import re
import random
from typing import List, Tuple, Dict


def _is_static_run_folder(path: str) -> bool:
    """
    A static-style run folder is considered valid if it has at least:
      - images/
      - lidar/
    """
    return (
        os.path.isdir(os.path.join(path, "images")) and
        os.path.isdir(os.path.join(path, "lidar"))
    )


def _is_dynamic_run_folder(path: str) -> bool:
    """
    A dynamic-rig run folder is considered valid if it has:
      - images/
      - scans/
      - lidar/
      - odom/
      - intrinsics/
      - meta/

    We keep this explicit because dynamic-rig samples are expected to be
    multi-modal and keyed by the same sample ID across these folders.
    """
    required_subdirs = [
        "images",
        "scans",
        "lidar",
        "odom",
        "intrinsics",
        "meta",
    ]
    return all(os.path.isdir(os.path.join(path, name)) for name in required_subdirs)


def _is_run_folder(path: str) -> bool:
    """
    A "run folder" is valid if it matches either:
      - static-style structure
      - dynamic-rig structure
    """
    return _is_static_run_folder(path) or _is_dynamic_run_folder(path)


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

    ids: List[str] = []
    for f in os.listdir(image_dir):
        if f.endswith(".png"):
            ids.append(os.path.splitext(f)[0])

    return sorted(ids)


def _validate_static_samples(run_dir: str, sample_ids: List[str]) -> Tuple[List[str], Dict[str, int]]:
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


def _validate_dynamic_samples(run_dir: str, sample_ids: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Filters dynamic-rig sample IDs to keep only those with all required
    per-sample files present.

    Required per-sample files:
      - images/<sid>.png
      - scans/<sid>.npz
      - lidar/<sid>.pcd
      - odom/<sid>.yaml
      - meta/<sid>.yaml

    Required per-run file:
      - intrinsics/camera_intrinsics.yaml

    Returns:
      valid_ids: List[str]
      stats: dict with missing counts (for logging)
    """
    img_dir = os.path.join(run_dir, "images")
    scans_dir = os.path.join(run_dir, "scans")
    lidar_dir = os.path.join(run_dir, "lidar")
    odom_dir = os.path.join(run_dir, "odom")
    meta_dir = os.path.join(run_dir, "meta")
    intrinsics_path = os.path.join(run_dir, "intrinsics", "camera_intrinsics.yaml")

    missing_img = 0
    missing_scans = 0
    missing_lidar = 0
    missing_odom = 0
    missing_meta = 0
    missing_intrinsics = 0

    valid: List[str] = []

    for sid in sample_ids:
        img_path = os.path.join(img_dir, f"{sid}.png")
        scan_path = os.path.join(scans_dir, f"{sid}.npz")
        lidar_path = os.path.join(lidar_dir, f"{sid}.pcd")
        odom_path = os.path.join(odom_dir, f"{sid}.yaml")
        meta_path = os.path.join(meta_dir, f"{sid}.yaml")

        ok = True

        if not os.path.exists(img_path):
            missing_img += 1
            ok = False

        if not os.path.exists(scan_path):
            missing_scans += 1
            ok = False

        if not os.path.exists(lidar_path):
            missing_lidar += 1
            ok = False

        if not os.path.exists(odom_path):
            missing_odom += 1
            ok = False

        if not os.path.exists(meta_path):
            missing_meta += 1
            ok = False

        if not os.path.exists(intrinsics_path):
            missing_intrinsics += 1
            ok = False

        if ok:
            valid.append(sid)

    stats = {
        "missing_img": missing_img,
        "missing_scans": missing_scans,
        "missing_lidar": missing_lidar,
        "missing_odom": missing_odom,
        "missing_meta": missing_meta,
        "missing_intrinsics_file": missing_intrinsics,
    }
    return valid, stats


def _validate_samples(run_dir: str, sample_ids: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Dispatches validation based on the detected run-folder style.
    """
    if _is_dynamic_run_folder(run_dir):
        return _validate_dynamic_samples(run_dir, sample_ids)

    return _validate_static_samples(run_dir, sample_ids)


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

    Supports:
      - static-style runs
      - dynamic-rig runs

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
            f"Expected either a run folder or a parent folder with run_### subfolders."
        )

    rng = random.Random(seed)

    # -----------------------------
    # Step 2: Process each run
    # -----------------------------
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_type = "dynamic" if _is_dynamic_run_folder(run_dir) else "static"

        # (a) Discover candidate IDs from images/
        candidate_ids = _list_sample_ids_from_images(run_dir)

        # (b) Validate required per-sample files
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
            f"(type={run_type}, total_valid={n_total}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}), "
            f"missing={stats}"
        )


if __name__ == "__main__":

    generate_splits(
        dataset_root="~/dataset_root/static_mount_h1",
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )