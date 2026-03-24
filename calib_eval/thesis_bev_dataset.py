###############################################
# BEV TRAINING DATASET FOR THESIS PROJECT
###############################################

import os
import yaml
import numpy as np
from typing import Dict, Optional

from calib_eval.universal_dataset_loader import UniversalCalibrationDataset
from calib_eval.geometry_utils import (
    quaternion_to_rotation_matrix,
    invert_extrinsics_cam_to_lidar_to_lidar_to_cam,
)


def _project_root_from_this_file() -> str:
    """
    Returns the package root:
        .../src/calib_eval
    when this file lives at:
        .../src/calib_eval/calib_eval/thesis_bev_dataset.py
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_reference_yaml_paths() -> Dict[str, str]:
    project_root = _project_root_from_this_file()
    ref_dir = os.path.join(project_root, "config", "reference")
    return {
        "static_mount_h1": os.path.join(ref_dir, "static_mount_h1_reference.yaml"),
        "static_mount_h2": os.path.join(ref_dir, "static_mount_h2_reference.yaml"),
        "dynamic_rig": os.path.join(ref_dir, "dynamic_reference.yaml"),
    }


def _load_reference_yaml(reference_yaml_path: str) -> Dict:
    if not os.path.exists(reference_yaml_path):
        raise RuntimeError(f"Reference YAML path does not exist: {reference_yaml_path}")

    with open(reference_yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    for key in ["translation", "rotation_quat"]:
        if key not in data:
            raise RuntimeError(f"Reference YAML missing required key: '{key}'")

    t = data["translation"]
    q = data["rotation_quat"]

    if not isinstance(t, list) or len(t) != 3:
        raise RuntimeError(
            f"Reference YAML translation must be a list of length 3: {reference_yaml_path}"
        )
    if not isinstance(q, list) or len(q) != 4:
        raise RuntimeError(
            f"Reference YAML rotation_quat must be a list of length 4: {reference_yaml_path}"
        )

    return {
        "translation": [float(v) for v in t],
        "rotation_quat": [float(v) for v in q],
        "parent_frame": str(data.get("parent_frame", "camera")),
        "child_frame": str(data.get("child_frame", "lidar")),
    }


def _make_transform_matrix_lidar_to_camera(
    t_cam_lidar_xyz,
    q_cam_lidar_xyzw,
) -> np.ndarray:
    """
    BEVCalib train_kitti.py expects gt_T_to_camera as LiDAR -> camera.

    Our package reference YAMLs store camera -> lidar.
    Therefore we invert them first.
    """
    t_lidar_cam, q_lidar_cam = invert_extrinsics_cam_to_lidar_to_lidar_to_cam(
        t_cam_lidar=t_cam_lidar_xyz,
        q_cam_lidar_xyzw=q_cam_lidar_xyzw,
    )

    R_lidar_cam = quaternion_to_rotation_matrix(q_lidar_cam)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_lidar_cam.astype(np.float32)
    T[:3, 3] = np.asarray(t_lidar_cam, dtype=np.float32).reshape(3,)
    return T


class ThesisBEVDataset:
    """
    Thin adapter that exposes the exact sample contract expected by the
    official BEVCalib training loop:

        item[0] -> image
        item[1] -> point cloud
        item[2] -> gt_T_to_camera   (LiDAR -> camera, 4x4)
        item[3] -> intrinsics       (3x3)

    Internally this reuses UniversalCalibrationDataset so we do not duplicate
    any dataset-structure parsing logic from the thesis pipeline.
    """

    def __init__(
        self,
        dataset_root: str,
        split: Optional[str] = None,
        combined_index: bool = False,
        static_mount_h1_reference_yaml: str = "",
        static_mount_h2_reference_yaml: str = "",
        dynamic_reference_yaml: str = "",
    ):
        self.dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
        self.split = split
        self.combined_index = bool(combined_index)

        self.base_ds = UniversalCalibrationDataset(
            dataset_root=self.dataset_root,
            split=self.split,
            combined_index=self.combined_index,
        )

        default_refs = _default_reference_yaml_paths()
        self.reference_yaml_paths = {
            "static_mount_h1": (
                static_mount_h1_reference_yaml.strip()
                if static_mount_h1_reference_yaml.strip()
                else default_refs["static_mount_h1"]
            ),
            "static_mount_h2": (
                static_mount_h2_reference_yaml.strip()
                if static_mount_h2_reference_yaml.strip()
                else default_refs["static_mount_h2"]
            ),
            "dynamic_rig": (
                dynamic_reference_yaml.strip()
                if dynamic_reference_yaml.strip()
                else default_refs["dynamic_rig"]
            ),
        }

        self.reference_yaml_data = {
            "static_mount_h1": _load_reference_yaml(
                self.reference_yaml_paths["static_mount_h1"]
            ),
            "static_mount_h2": _load_reference_yaml(
                self.reference_yaml_paths["static_mount_h2"]
            ),
            "dynamic_rig": _load_reference_yaml(
                self.reference_yaml_paths["dynamic_rig"]
            ),
        }

        self.reference_T_lidar_to_camera = {
            rig_id: _make_transform_matrix_lidar_to_camera(
                self.reference_yaml_data[rig_id]["translation"],
                self.reference_yaml_data[rig_id]["rotation_quat"],
            )
            for rig_id in ["static_mount_h1", "static_mount_h2", "dynamic_rig"]
        }

    def __len__(self) -> int:
        return len(self.base_ds)

    def _infer_rig_config_id(self, run_name: str) -> str:
        """
        Determine which reference to use.

        Supported cases:
          1) dataset_root points directly to static_mount_h1 or static_mount_h2
          2) dataset_root points to the combined symlink dataset static_mount_h1_h2
             with the agreed mapping:
                 run_001..008 -> h1
                 run_009..016 -> h2
          3) dataset_root points to dynamic_rig
        """
        root_name = os.path.basename(self.dataset_root.rstrip(os.sep))

        if root_name == "static_mount_h1":
            return "static_mount_h1"
        if root_name == "static_mount_h2":
            return "static_mount_h2"

        if root_name == "static_mount_h1_h2":
            try:
                run_num = int(run_name.split("_")[-1])
            except Exception:
                raise RuntimeError(
                    f"Could not infer run number from combined dataset run name: {run_name}"
                )

            if 1 <= run_num <= 8:
                return "static_mount_h1"
            if 9 <= run_num <= 16:
                return "static_mount_h2"

            raise RuntimeError(
                f"Combined dataset run number outside expected range 1..16: {run_name}"
            )

        if root_name == "dynamic_rig":
            return "dynamic_rig"

        run_name_l = run_name.lower()
        root_l = self.dataset_root.lower()

        if "dynamic" in root_l or "dynamic" in run_name_l:
            return "dynamic_rig"
        if "h1" in root_l or "h1" in run_name_l:
            return "static_mount_h1"
        if "h2" in root_l or "h2" in run_name_l:
            return "static_mount_h2"

        raise RuntimeError(
            "Could not infer rig_config_id for BEV training dataset.\n"
            f"  dataset_root={self.dataset_root}\n"
            f"  run_name={run_name}\n"
            "Expected static_mount_h1, static_mount_h2, static_mount_h1_h2, or dynamic_rig."
        )

    def __getitem__(self, idx: int):
        sample = self.base_ds.load_sample(idx)

        image_bgr = sample["image"]
        points = sample["points"]
        intrinsics_dict = sample["intrinsics"]
        run_name = sample["run_name"]

        if image_bgr is None:
            raise RuntimeError(f"Sample has no image at idx={idx}")
        if points is None:
            raise RuntimeError(
                f"BEV training currently expects point clouds. Sample has no points at idx={idx}"
            )

        rig_config_id = self._infer_rig_config_id(run_name)
        gt_T_to_camera = self.reference_T_lidar_to_camera[rig_config_id].copy()

        K = np.array(
            [
                [float(intrinsics_dict["fx"]), 0.0, float(intrinsics_dict["cx"])],
                [0.0, float(intrinsics_dict["fy"]), float(intrinsics_dict["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        points_xyz = np.asarray(points, dtype=np.float32)
        if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
            raise RuntimeError(
                f"Expected points shape Nx3 or Nx>=3, got {points_xyz.shape} at idx={idx}"
            )
        points_xyz = points_xyz[:, :3].astype(np.float32)

        return (
            image_bgr,
            points_xyz,
            gt_T_to_camera.astype(np.float32),
            K.astype(np.float32),
        )