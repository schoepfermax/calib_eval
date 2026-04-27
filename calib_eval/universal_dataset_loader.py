###############################################
# UNIVERSAL DATA LOADER
###############################################

import os
import re
import yaml
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


def _is_run_folder(path: str) -> bool:
    """
    A valid run folder must include:
      - images/
      - lidar/

    Notes:
      - lidar/ remains mandatory because both the static rig and the dynamic rig
        may store pseudo point clouds there.
      - scans/ and odom/ are optional because older datasets may not have them.
    """
    return (
        os.path.isdir(os.path.join(path, "images")) and
        os.path.isdir(os.path.join(path, "lidar"))
    )


def _discover_run_folders(parent_dir: str) -> List[str]:
    """
    Finds run folders under parent_dir matching run_###.
    Returns absolute paths sorted by run number.
    """
    run_dirs: List[str] = []
    pat = re.compile(r"^run_(\d{3})$")

    if not os.path.isdir(parent_dir):
        return run_dirs

    for name in os.listdir(parent_dir):
        m = pat.match(name)
        if not m:
            continue
        run_dir = os.path.join(parent_dir, name)
        if _is_run_folder(run_dir):
            run_dirs.append(run_dir)

    # Sort by run number
    def _run_num(d: str) -> int:
        bn = os.path.basename(d)
        m = pat.match(bn)
        return int(m.group(1)) if m else 999999

    run_dirs.sort(key=_run_num)
    return run_dirs


def _read_nonempty_noncomment_lines(txt_path: str) -> List[str]:
    """
    Reads a text file containing one token per line (split index).
    Ignores empty lines and comment lines starting with '#'.
    """
    out: List[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _extract_intrinsics_from_yaml_dict(d: Dict) -> Optional[Dict]:
    """
    Best-effort extraction of camera intrinsics from a YAML dict.

    Supported patterns (checked in this order):
      1) Flat keys: {width,height,fx,fy,cx,cy}
      2) Nested camera: {camera: {width,height,fx,fy,cx,cy}}
      3) ROS CameraInfo-like:
         - {width,height,K:[9]}
         - {camera_matrix:{data:[9]}, image_width/image_height}

    Returns a dict with keys: width,height,fx,fy,cx,cy or None.
    """
    if not isinstance(d, dict):
        return None

    # 1) Flat
    w = d.get("width", None)
    h = d.get("height", None)
    fx = d.get("fx", None)
    fy = d.get("fy", None)
    cx = d.get("cx", None)
    cy = d.get("cy", None)
    if all(v is not None for v in [w, h, fx, fy, cx, cy]):
        return {
            "width": int(w),
            "height": int(h),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }

    # 2) Nested camera
    cam = d.get("camera", None)
    if isinstance(cam, dict):
        w = cam.get("width", d.get("width", None))
        h = cam.get("height", d.get("height", None))
        fx = cam.get("fx", None)
        fy = cam.get("fy", None)
        cx = cam.get("cx", None)
        cy = cam.get("cy", None)
        if all(v is not None for v in [w, h, fx, fy, cx, cy]):
            return {
                "width": int(w),
                "height": int(h),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }

    # 3a) CameraInfo-like with K
    K = d.get("K", d.get("k", None))  # support ROS-style lower-case k
    if isinstance(K, (list, tuple)) and len(K) == 9:
        w = d.get("width", d.get("image_width", None))
        h = d.get("height", d.get("image_height", None))
        if w is not None and h is not None:
            return {
                "width": int(w),
                "height": int(h),
                "fx": float(K[0]),
                "fy": float(K[4]),
                "cx": float(K[2]),
                "cy": float(K[5]),
            }

    # 3b) ROS calibration yaml style
    if isinstance(d.get("camera_matrix", None), dict) and isinstance(d["camera_matrix"].get("data", None), (list, tuple)):
        data = d["camera_matrix"]["data"]
        if len(data) == 9:
            w = d.get("image_width", d.get("width", None))
            h = d.get("image_height", d.get("height", None))
            if w is not None and h is not None:
                return {
                    "width": int(w),
                    "height": int(h),
                    "fx": float(data[0]),
                    "fy": float(data[4]),
                    "cx": float(data[2]),
                    "cy": float(data[5]),
                }

    return None


class UniversalCalibrationDataset:

    def __init__(
        self,
        dataset_root: str,
        split: Optional[str] = None,
        combined_index: bool = False,
    ):
        """
        Args:
          dataset_root:
            - run folder path OR parent folder path
          split:
            - None -> load all samples (no index filtering)
            - 'train'/'val'/'test' -> use index files in splits/ OR index/
          combined_index:
            - If True, allow a single combined index file at dataset_root/(splits|index)/<split>.txt
              that references keys like run_###/sample_id.
        """
        self.dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
        self.split = split
        self.combined_index = combined_index

        # Determine whether dataset_root is a run folder or a parent directory
        if _is_run_folder(self.dataset_root):
            self.parent_dir = os.path.dirname(self.dataset_root)
            self.run_dirs = [self.dataset_root]
        else:
            self.parent_dir = self.dataset_root
            self.run_dirs = _discover_run_folders(self.parent_dir)

        if not self.run_dirs:
            raise RuntimeError(f"No run folders found under: {self.dataset_root}")

        # Build list of (run_dir, sample_id) pairs
        self.samples: List[Tuple[str, str]] = self._build_sample_list()

        if self.split is not None:
            self.samples = self._apply_split_filter(self.samples, self.split)

        if not self.samples:
            raise RuntimeError(f"No samples found after filtering (split={self.split}).")

    def __len__(self) -> int:
        return len(self.samples)

    def make_key(self, run_dir: str, sample_id: str) -> str:
        return f"{os.path.basename(run_dir)}/{sample_id}"

    def _build_sample_list(self) -> List[Tuple[str, str]]:
        """
        Discover all sample IDs by reading the images/ folder (png files).
        LiDAR + intrinsics are expected to match these sample IDs.
        """
        all_samples: List[Tuple[str, str]] = []

        for run_dir in self.run_dirs:
            img_dir = os.path.join(run_dir, "images")
            if not os.path.isdir(img_dir):
                continue

            for name in os.listdir(img_dir):
                if not name.endswith(".png"):
                    continue
                sample_id = name[:-4]
                all_samples.append((run_dir, sample_id))

        # Sort deterministically by run name then sample_id
        all_samples.sort(key=lambda x: (os.path.basename(x[0]), x[1]))
        return all_samples

    def _apply_split_filter(self, samples: List[Tuple[str, str]], split: str) -> List[Tuple[str, str]]:
        """
        Filters (run_dir, sample_id) pairs using precomputed splits.

        Supported formats:
          A) Per-run index:
             run_###/splits/<split>.txt containing sample_id lines
             run_###/index/<split>.txt  containing sample_id lines
          B) Combined index:
             dataset_root/splits/<split>.txt containing run_###/sample_id keys
             dataset_root/index/<split>.txt  containing run_###/sample_id keys
        """
        split = str(split).strip().lower()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split} (expected train/val/test)")

        allowed_keys = set()

        if self.combined_index:
            # Combined index file: dataset_root/(splits|index)/<split>.txt
            cand_paths = [
                os.path.join(self.dataset_root, "splits", f"{split}.txt"),
                os.path.join(self.dataset_root, "index", f"{split}.txt"),
            ]
            split_path = None
            for p in cand_paths:
                if os.path.exists(p):
                    split_path = p
                    break
            if split_path is None:
                raise RuntimeError(f"Combined split file not found. Tried: {cand_paths}")

            for key in _read_nonempty_noncomment_lines(split_path):
                allowed_keys.add(key)
        else:
            # Per-run split files
            for run_dir in self.run_dirs:
                cand_paths = [
                    os.path.join(run_dir, "splits", f"{split}.txt"),
                    os.path.join(run_dir, "index", f"{split}.txt"),
                ]
                split_path = None
                for p in cand_paths:
                    if os.path.exists(p):
                        split_path = p
                        break
                if split_path is None:
                    continue

                run_name = os.path.basename(run_dir)
                for sid in _read_nonempty_noncomment_lines(split_path):
                    allowed_keys.add(f"{run_name}/{sid}")

        out = []
        for run_dir, sid in samples:
            if self.make_key(run_dir, sid) in allowed_keys:
                out.append((run_dir, sid))
        return out

    def load_sample(self, idx: int) -> Dict:
        """
        Loads a sample by integer index into self.samples.
        """
        run_dir, sample_id = self.samples[idx]
        return self._load_from_run_and_id(run_dir, sample_id)

    def load_sample_by_key(self, key: str) -> Dict:
        """
        Loads a sample by string key "run_###/sample_id".
        Useful when model / evaluation / calibration code stores keys in logs or summaries.        """
        if "/" not in key:
            raise ValueError(f"Expected key format 'run_###/sample_id', got: {key}")

        run_name, sample_id = key.split("/", 1)
        run_dir = os.path.join(self.parent_dir, run_name)

        if not _is_run_folder(run_dir):
            raise RuntimeError(f"Run folder not found or invalid: {run_dir}")

        return self._load_from_run_and_id(run_dir, sample_id)

    def _load_from_run_and_id(self, run_dir: str, sample_id: str) -> Dict:
        """
        Core loader: given (run_dir, sample_id), read files and return a universal dict.
        """
        run_name = os.path.basename(run_dir)

        # ------------------------------------------
        # 1) Load intrinsics from dataset-native convention ONLY:
        #    run_###/intrinsics/camera_intrinsics.yaml
        # ------------------------------------------
        intr_path = os.path.join(run_dir, "intrinsics", "camera_intrinsics.yaml")

        if not os.path.exists(intr_path):
            raise RuntimeError(
                f"Missing camera intrinsics yaml: {intr_path}\n"
                "Expected dataset-native path: run_###/intrinsics/camera_intrinsics.yaml"
            )

        with open(intr_path, "r") as f:
            intr_d = yaml.safe_load(f) or {}

        parsed = _extract_intrinsics_from_yaml_dict(intr_d)
        if parsed is None:
            raise RuntimeError(
                f"Found intrinsics yaml, but could not parse intrinsics fields: {intr_path}"
            )
        intrinsics = parsed

        # ------------------------------------------
        # 2) Load image
        # ------------------------------------------
        img_path = os.path.join(run_dir, "images", f"{sample_id}.png")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # ------------------------------------------
        # 3) Load LiDAR / scan data
        # ------------------------------------------
        # Static rig:
        #   - stored as ASCII PCD (x y z) under lidar/{sample_id}.pcd
        #
        # Dynamic rig:
        #   - pseudo point cloud may be stored under:
        #       lidar/{sample_id}.pcd
        #   - raw 2D scan may be stored under:
        #       scans/{sample_id}.npz
        #       scans/{sample_id}.yaml
        #
        # Backward compatibility:
        #   - older dynamic datasets may still store raw scan under lidar/
        #
        # Returned fields:
        #   - points: Nx3 float32 OR None
        #   - scan: dict OR None
        #
        # Both may coexist for the same dynamic sample.
        # Preferred current paths
        pcd_path = os.path.join(run_dir, "lidar", f"{sample_id}.pcd")
        scan_npz_path = os.path.join(run_dir, "scans", f"{sample_id}.npz")
        scan_yaml_path = os.path.join(run_dir, "scans", f"{sample_id}.yaml")
        odom_yaml_path = os.path.join(run_dir, "odom", f"{sample_id}.yaml")

        # Legacy scan paths
        legacy_scan_npz_path = os.path.join(run_dir, "lidar", f"{sample_id}.npz")
        legacy_scan_yaml_path = os.path.join(run_dir, "lidar", f"{sample_id}.yaml")

        points = None
        scan = None
        odom = None

        # Pseudo / true point cloud if available
        if os.path.exists(pcd_path):
            points = self._read_ascii_pcd_xyz(pcd_path)

        # Raw scan if available (prefer scans/, fall back to legacy lidar/)
        if os.path.exists(scan_npz_path):
            scan = self._read_scan_npz(scan_npz_path)
        elif os.path.exists(scan_yaml_path):
            scan = self._read_scan_yaml(scan_yaml_path)
        elif os.path.exists(legacy_scan_npz_path):
            scan = self._read_scan_npz(legacy_scan_npz_path)
        elif os.path.exists(legacy_scan_yaml_path):
            scan = self._read_scan_yaml(legacy_scan_yaml_path)

        # Optional odometry for dynamic / sequence-based refinement
        if os.path.exists(odom_yaml_path):
            odom = self._read_odom_yaml(odom_yaml_path)

        # At least one geometric representation must exist
        if points is None and scan is None:
            raise RuntimeError(
                "No LiDAR/scan data found for sample.\n"
                f"  tried point cloud: {pcd_path}\n"
                f"  tried scan: {scan_npz_path}\n"
                f"  tried scan: {scan_yaml_path}\n"
                f"  tried legacy scan: {legacy_scan_npz_path}\n"
                f"  tried legacy scan: {legacy_scan_yaml_path}"
            )

        # ------------------------------------------
        # 4) Optional: Load GT extrinsics (supervised)
        # ------------------------------------------
        gt_path = os.path.join(run_dir, "extrinsics_gt", f"{sample_id}.yaml")
        has_gt = os.path.exists(gt_path)

        gt_translation = None
        gt_rotation_quat = None

        if has_gt:
            with open(gt_path, "r") as f:
                gt = yaml.safe_load(f) or {}
            # Use the schema we standardized earlier:
            #   translation: [tx, ty, tz]
            #   rotation_quat: [qx, qy, qz, qw]
            if "translation" in gt and "rotation_quat" in gt:
                gt_translation = gt["translation"]
                gt_rotation_quat = gt["rotation_quat"]

        # ------------------------------------------
        # 5) Return universal dict (explicit fields)
        # ------------------------------------------
        return {
            # Identity
            "run_name": run_name,
            "run_dir": run_dir,
            "sample_id": sample_id,
            "sample_key": self.make_key(run_dir, sample_id),

            # File paths
            "image_path": img_path,
            "pcd_path": pcd_path,
            "scan_npz_path": scan_npz_path,
            "scan_yaml_path": scan_yaml_path,
            "legacy_scan_npz_path": legacy_scan_npz_path,
            "legacy_scan_yaml_path": legacy_scan_yaml_path,
            "odom_yaml_path": odom_yaml_path,
            "gt_path": gt_path,

            # Core data
            "image": image,             # BGR (OpenCV convention)
            "points": points,           # Nx3 float32 OR None
            "scan": scan,               # dict OR None (raw dynamic rig scan)
            "odom": odom,               # dict OR None
            "intrinsics": intrinsics,   # dict

            # Optional supervised label
            "has_gt_extrinsics": has_gt,
            "gt_translation": gt_translation,
            "gt_rotation_quat": gt_rotation_quat,
        }

    def _read_scan_npz(self, npz_path: str) -> Dict:
        """
        Reads a LaserScan-like sample stored as .npz.

        Expected keys (best-effort):
          - ranges (required)
          - angle_min, angle_increment (required)

        Optional keys:
          - angle_max
          - range_min, range_max
          - time_increment
          - scan_time
          - intensities

        Returns a dict compatible with the data preprocessor dynamic-rig path.
        """
        data = np.load(npz_path, allow_pickle=False)

        if "ranges" not in data:
            raise RuntimeError(f"Scan npz missing 'ranges': {npz_path}")
        if "angle_min" not in data or "angle_increment" not in data:
            raise RuntimeError(f"Scan npz missing angle_min/angle_increment: {npz_path}")

        out = {
            "ranges": data["ranges"].astype(np.float32).tolist(),
            "angle_min": float(np.asarray(data["angle_min"]).reshape(())),
            "angle_increment": float(np.asarray(data["angle_increment"]).reshape(())),
        }

        if "angle_max" in data:
            out["angle_max"] = float(np.asarray(data["angle_max"]).reshape(()))
        else:
            n = len(out["ranges"])
            out["angle_max"] = out["angle_min"] + max(0, n - 1) * out["angle_increment"]

        if "range_min" in data:
            out["range_min"] = float(np.asarray(data["range_min"]).reshape(()))
        if "range_max" in data:
            out["range_max"] = float(np.asarray(data["range_max"]).reshape(()))
        if "time_increment" in data:
            out["time_increment"] = float(np.asarray(data["time_increment"]).reshape(()))
        if "scan_time" in data:
            out["scan_time"] = float(np.asarray(data["scan_time"]).reshape(()))
        if "intensities" in data:
            out["intensities"] = data["intensities"].astype(np.float32).tolist()

        return out

    def _read_scan_yaml(self, yaml_path: str) -> Dict:
        """
        Reads a LaserScan-like sample stored as .yaml.

        Expected keys (best-effort):
          - ranges (required)
          - angle_min, angle_increment (required)

        Optional keys:
          - angle_max
          - range_min, range_max
          - time_increment
          - scan_time
          - intensities
        """
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f) or {}

        if "ranges" not in d:
            raise RuntimeError(f"Scan yaml missing 'ranges': {yaml_path}")
        if "angle_min" not in d or "angle_increment" not in d:
            raise RuntimeError(f"Scan yaml missing angle_min/angle_increment: {yaml_path}")

        out = {
            "ranges": list(d["ranges"]),
            "angle_min": float(d["angle_min"]),
            "angle_increment": float(d["angle_increment"]),
        }

        if "angle_max" in d:
            out["angle_max"] = float(d["angle_max"])
        else:
            n = len(out["ranges"])
            out["angle_max"] = out["angle_min"] + max(0, n - 1) * out["angle_increment"]

        if "range_min" in d:
            out["range_min"] = float(d["range_min"])
        if "range_max" in d:
            out["range_max"] = float(d["range_max"])
        if "time_increment" in d:
            out["time_increment"] = float(d["time_increment"])
        if "scan_time" in d:
            out["scan_time"] = float(d["scan_time"])
        if "intensities" in d:
            out["intensities"] = list(d["intensities"])

        return out
    
    def _read_odom_yaml(self, yaml_path: str) -> Dict:
        """
        Reads odometry stored as YAML.

        Expected flexible structure (best-effort):
          position:
            x: ...
            y: ...
            z: ...
          orientation:
            x: ...
            y: ...
            z: ...
            w: ...
          linear_velocity:
            x: ...
            y: ...
            z: ...
          angular_velocity:
            x: ...
            y: ...
            z: ...

        Optional:
          header_frame_id
          child_frame_id

        Returns a permissive dict for playback / online2d refinement.
        """
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f) or {}

        out = {
            "position": d.get("position", {}) or {},
            "orientation": d.get("orientation", {}) or {},
            "linear_velocity": d.get("linear_velocity", {}) or {},
            "angular_velocity": d.get("angular_velocity", {}) or {},
            "header_frame_id": d.get("header_frame_id", "odom"),
            "child_frame_id": d.get("child_frame_id", "base_link"),
        }
        return out

    def _read_ascii_pcd_xyz(self, pcd_path: str) -> np.ndarray:
        """
        Reads a simple ASCII PCD with x y z fields.
        This matches what our extractor writes for the static rig.
        """
        if not os.path.exists(pcd_path):
            raise RuntimeError(f"PCD file not found: {pcd_path}")

        with open(pcd_path, "r") as f:
            lines = f.readlines()

        data_start = None
        for i, line in enumerate(lines):
            if line.strip().lower() == "data ascii":
                data_start = i + 1
                break

        if data_start is None:
            raise RuntimeError(f"Invalid PCD (no 'DATA ascii'): {pcd_path}")

        pts = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except Exception:
                continue
            pts.append([x, y, z])

        if not pts:
            return np.zeros((0, 3), dtype=np.float32)

        return np.asarray(pts, dtype=np.float32)