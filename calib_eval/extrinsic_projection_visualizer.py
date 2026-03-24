#!/usr/bin/env python3

###############################################
# VISUALIZER
###############################################
# Primary purpose:
#   - thesis-grade qualitative visualization for archived calibration runs
#   - side-by-side image-plane projection comparison:
#       * reference extrinsics
#       * estimated extrinsics
#
# Current supported input modes:
#   1) live
#      - subscribe to live pipeline topics
#   2) archive_reconstruct
#      - read archived run_info.txt
#      - load one chosen sample from dataset
#      - load checkpoint from archive metadata
#      - reconstruct ONE estimated transform by single-frame inference
#
# IMPORTANT:
#   - archived runs do not store estimated extrinsics directly
#   - therefore archive_reconstruct mode performs deterministic single-frame
#     inference for the selected sample; it does NOT rerun the full experiment
#
# Published topics:
#   /viz/reference_overlay
#   /viz/estimated_overlay
#
# Notes:
#   - This file intentionally reuses existing project math/helpers rather than
#     re-implementing calibration geometry from scratch.
#   - Projection-based interpretation for dynamic pseudo-PCD remains limited;
#     this visualizer is still useful for qualitative comparison and thesis
#     figures.
###############################################

import os
import math
import yaml
import numpy as np
import cv2
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2

import torch

from calib_eval.calib_model_node import (
    _K_from_camera_info,
    _resize_and_scale_intrinsics,
    _make_T_from_yaml_camera_to_lidar,
)

from calib_eval.geometry_utils import (
    quaternion_to_rotation_matrix,
    project_lidar_to_image,
    invert_extrinsics_cam_to_lidar_to_lidar_to_cam,
)
from calib_eval.universal_dataset_loader import UniversalCalibrationDataset
from calib_eval.models.registry import create_model_from_registry


def quat_normalize_xyzw(q):
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_mul_xyzw(q1, q2):
    """
    Hamilton product q = q1 ⊗ q2, with xyzw convention.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return quat_normalize_xyzw([x, y, z, w])


class ExtrinsicProjectionVisualizer(Node):
    """
    Archive-first visualizer with optional live mode.

    Naming note:
      We keep the file path for minimal change, but the node itself is called
      simply "visualizer" to match project preference.
    """

    def __init__(self):
        super().__init__("visualizer")

        self.bridge = CvBridge()

        ###############################################
        # Parameters
        ###############################################
        self.declare_parameter("input_mode", "archive_reconstruct")  # live | archive_reconstruct

        # Live topics
        self.declare_parameter("image_topic", "/eval/clean/image")
        self.declare_parameter("points_topic", "/eval/clean/points")
        self.declare_parameter("camera_info_topic", "/eval/camera_info")
        self.declare_parameter("reference_extrinsics_topic", "/eval/ref_extrinsics")
        self.declare_parameter("estimated_extrinsics_topic", "/eval/estimated_extrinsics")

        # Archive reconstruction inputs
        self.declare_parameter("archive_dir", "")
        self.declare_parameter("dataset_root", "")
        self.declare_parameter("run_name", "")
        self.declare_parameter("sample_id", "")
        self.declare_parameter("reference_yaml_path", "")
        self.declare_parameter("device", "auto")

        # General / thesis display
        self.declare_parameter("use_dynamic_rig", False)
        self.declare_parameter("dynamic_representation", "pseudo_points")
        self.declare_parameter("metadata_text", "")
        self.declare_parameter("reference_color_bgr", [0, 255, 0])   # green
        self.declare_parameter("estimated_color_bgr", [0, 0, 255])   # red
        self.declare_parameter("point_radius_px", 1)
        self.declare_parameter("bottom_banner_height_px", 44)

        # Output topics
        self.declare_parameter("reference_overlay_topic", "/viz/reference_overlay")
        self.declare_parameter("estimated_overlay_topic", "/viz/estimated_overlay")

        self.input_mode = str(self.get_parameter("input_mode").value).strip().lower()
        self.use_dynamic_rig = bool(self.get_parameter("use_dynamic_rig").value)
        self.dynamic_representation = str(self.get_parameter("dynamic_representation").value).strip().lower()

        self.reference_overlay_topic = str(self.get_parameter("reference_overlay_topic").value)
        self.estimated_overlay_topic = str(self.get_parameter("estimated_overlay_topic").value)

        self.reference_color_bgr = tuple(int(v) for v in self.get_parameter("reference_color_bgr").value)
        self.estimated_color_bgr = tuple(int(v) for v in self.get_parameter("estimated_color_bgr").value)
        self.point_radius_px = int(self.get_parameter("point_radius_px").value)
        self.bottom_banner_height_px = int(self.get_parameter("bottom_banner_height_px").value)

        # Published image topics for RViz panels
        self.ref_pub = self.create_publisher(Image, self.reference_overlay_topic, 10)
        self.est_pub = self.create_publisher(Image, self.estimated_overlay_topic, 10)

        # Live-mode buffers
        self.latest_image_msg = None
        self.latest_points_msg = None
        self.latest_caminfo_msg = None
        self.latest_ref_msg = None
        self.latest_est_msg = None

        # Archive-mode cached sample/inference outputs
        self.archive_meta: Dict = {}
        self.archive_sample: Optional[Dict] = None
        self.archive_ref_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.archive_est_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.archive_render_done = False

        self.model = None
        self.model_name = ""
        self.device = self._resolve_device(str(self.get_parameter("device").value).strip().lower())

        if self.input_mode == "live":
            self._setup_live_mode()
        elif self.input_mode == "archive_reconstruct":
            self._setup_archive_mode()
        else:
            raise RuntimeError(
                f"input_mode must be one of ['live', 'archive_reconstruct'], got: {self.input_mode}"
            )

    ###############################################
    # Setup helpers
    ###############################################
    def _resolve_device(self, dev_param: str) -> torch.device:
        if dev_param == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' requested, but torch.cuda.is_available() is False.")
            return torch.device("cuda")
        if dev_param == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_live_mode(self):
        image_topic = str(self.get_parameter("image_topic").value)
        points_topic = str(self.get_parameter("points_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        reference_extrinsics_topic = str(self.get_parameter("reference_extrinsics_topic").value)
        estimated_extrinsics_topic = str(self.get_parameter("estimated_extrinsics_topic").value)

        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.create_subscription(PointCloud2, points_topic, self._points_cb, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._caminfo_cb, 10)
        self.create_subscription(TransformStamped, reference_extrinsics_topic, self._ref_cb, 10)
        self.create_subscription(TransformStamped, estimated_extrinsics_topic, self._est_cb, 10)

        self.timer = self.create_timer(0.25, self._render_live_tick)

        self.get_logger().info(
            "Visualizer started in live mode:\n"
            f"  image_topic={image_topic}\n"
            f"  points_topic={points_topic}\n"
            f"  camera_info_topic={camera_info_topic}\n"
            f"  reference_extrinsics_topic={reference_extrinsics_topic}\n"
            f"  estimated_extrinsics_topic={estimated_extrinsics_topic}\n"
            f"  reference_overlay_topic={self.reference_overlay_topic}\n"
            f"  estimated_overlay_topic={self.estimated_overlay_topic}\n"
        )

    def _setup_archive_mode(self):
        archive_dir = str(self.get_parameter("archive_dir").value).strip()
        if not archive_dir:
            raise RuntimeError("archive_dir must be provided for archive_reconstruct mode.")

        self.archive_meta = self._load_run_info(os.path.join(archive_dir, "run_info.txt"))

        # Allow parameter override for portability across workstation/laptop.
        dataset_root_override = str(self.get_parameter("dataset_root").value).strip()
        if dataset_root_override:
            self.archive_meta["dataset_root"] = dataset_root_override

        ref_yaml_override = str(self.get_parameter("reference_yaml_path").value).strip()
        if ref_yaml_override:
            self.archive_meta["reference_yaml_path"] = ref_yaml_override
        else:
            self.archive_meta["reference_yaml_path"] = self._resolve_reference_yaml_from_meta(self.archive_meta)

        run_name = str(self.get_parameter("run_name").value).strip()
        sample_id = str(self.get_parameter("sample_id").value).strip()

        if not run_name or not sample_id:
            raise RuntimeError(
                "archive_reconstruct mode requires explicit run_name and sample_id for reproducible figure generation."
            )

        self.archive_meta["run_name"] = run_name
        self.archive_meta["sample_id"] = sample_id
        self.archive_meta["sample_key"] = f"{run_name}/{sample_id}"

        self.archive_sample = self._load_archive_sample(
            dataset_root=self.archive_meta["dataset_root"],
            sample_key=self.archive_meta["sample_key"],
        )

        self.archive_ref_pose = self._load_pose_from_yaml(self.archive_meta["reference_yaml_path"])

        self.model_name = str(self.archive_meta.get("model", "")).strip().lower()
        if self.model_name == "bevcalib":
            self.model_name = "bev"
        elif self.model_name == "lccnet":
            self.model_name = "lccnet"

        ckpt_path = str(self.archive_meta.get("checkpoint", "")).strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError(f"Checkpoint missing or invalid in run_info.txt: {ckpt_path}")

        self.model = create_model_from_registry(
            model_name=self.model_name,
            checkpoint_path=ckpt_path,
            device="cuda" if self.device.type == "cuda" else "cpu",
            logger=self.get_logger(),
        )

        # Compute estimated pose once for the chosen figure frame.
        self.archive_est_pose = self._infer_pose_for_sample(
            model_name=self.model_name,
            model_wrapper=self.model,
            sample=self.archive_sample,
            ref_pose=self.archive_ref_pose,
        )

        t_est, q_est = self.archive_est_pose
        self.get_logger().info(
            "Archive reconstruction estimated extrinsics (camera -> lidar):\n"
            f"  translation = [{float(t_est[0]):.9f}, {float(t_est[1]):.9f}, {float(t_est[2]):.9f}]\n"
            f"  quaternion_xyzw = [{float(q_est[0]):.9f}, {float(q_est[1]):.9f}, {float(q_est[2]):.9f}, {float(q_est[3]):.9f}]"
        )

        # Publish the rendered images periodically so RViz can attach later.
        self.timer = self.create_timer(0.5, self._render_archive_tick)

        self.get_logger().info(
            "Visualizer started in archive_reconstruct mode:\n"
            f"  archive_dir={archive_dir}\n"
            f"  model={self.archive_meta.get('model', '')}\n"
            f"  checkpoint={self.archive_meta.get('checkpoint', '')}\n"
            f"  dataset_root={self.archive_meta.get('dataset_root', '')}\n"
            f"  sample_key={self.archive_meta.get('sample_key', '')}\n"
            f"  reference_yaml_path={self.archive_meta.get('reference_yaml_path', '')}\n"
            f"  reference_overlay_topic={self.reference_overlay_topic}\n"
            f"  estimated_overlay_topic={self.estimated_overlay_topic}\n"
        )

    ###############################################
    # Run-info / archive helpers
    ###############################################
    def _load_run_info(self, txt_path: str) -> Dict:
        if not os.path.isfile(txt_path):
            raise RuntimeError(f"run_info.txt not found: {txt_path}")

        out = {}
        with open(txt_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip()

        return out

    def _resolve_reference_yaml_from_meta(self, meta: Dict) -> str:
        # meta may store only "static_mount_h2_reference.yaml", not absolute path
        ref_raw = str(meta.get("reference_yaml", "")).strip()
        if not ref_raw:
            raise RuntimeError("reference_yaml not present in run_info.txt")

        if os.path.isabs(ref_raw):
            return ref_raw

        # Resolve relative to the package config/reference location.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(repo_root, "config", "reference", ref_raw)
        if os.path.isfile(cand):
            return cand

        raise RuntimeError(f"Could not resolve reference YAML from run_info.txt: {ref_raw}")

    def _load_archive_sample(self, dataset_root: str, sample_key: str) -> Dict:
        ds = UniversalCalibrationDataset(dataset_root=dataset_root, split=None, combined_index=True)
        return ds.load_sample_by_key(sample_key)

    def _load_pose_from_yaml(self, yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)

        t = np.asarray(d["translation"], dtype=np.float32).reshape(3,)
        q = quat_normalize_xyzw(d["rotation_quat"])
        return t, q

    ###############################################
    # Live-mode callbacks
    ###############################################
    def _image_cb(self, msg: Image):
        self.latest_image_msg = msg

    def _points_cb(self, msg: PointCloud2):
        self.latest_points_msg = msg

    def _caminfo_cb(self, msg: CameraInfo):
        self.latest_caminfo_msg = msg

    def _ref_cb(self, msg: TransformStamped):
        self.latest_ref_msg = msg

    def _est_cb(self, msg: TransformStamped):
        self.latest_est_msg = msg

    ###############################################
    # Common conversion helpers
    ###############################################
    def _caminfo_to_intrinsics(self, msg: CameraInfo) -> Dict:
        K = msg.k
        return {
            "fx": float(K[0]),
            "fy": float(K[4]),
            "cx": float(K[2]),
            "cy": float(K[5]),
            "width": int(msg.width),
            "height": int(msg.height),
        }

    def _sample_to_intrinsics(self, sample: Dict) -> Dict:
        intr = sample["intrinsics"]
        return {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "width": int(intr["width"]),
            "height": int(intr["height"]),
        }

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        pts = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([float(p[0]), float(p[1]), float(p[2])])

        if not pts:
            return np.zeros((0, 3), dtype=np.float32)

        return np.asarray(pts, dtype=np.float32)

    def _pose_from_tf_msg(self, msg: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
        t = np.array(
            [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z],
            dtype=np.float32,
        )
        q = np.array(
            [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w],
            dtype=np.float32,
        )
        q = quat_normalize_xyzw(q)
        return t, q

    ###############################################
    # LCC single-frame helpers
    ###############################################
    def _pad_bottom_right_to_official_canvas(self, img_bgr, depth_img):
        target_h = 384
        target_w = 1280

        h, w = img_bgr.shape[:2]
        pad_bottom = max(0, target_h - h)
        pad_right = max(0, target_w - w)

        if pad_bottom == 0 and pad_right == 0:
            return img_bgr, depth_img

        img_pad = np.pad(
            img_bgr,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        depth_pad = np.pad(
            depth_img,
            ((0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=0.0,
        )

        return img_pad, depth_pad

    def _build_sparse_depth_from_pose(self, points_lidar, intr_proj, image_shape_hw, t_cam_lidar, q_cam_lidar):
        t_lidar_cam, q_lidar_cam = invert_extrinsics_cam_to_lidar_to_lidar_to_cam(
            t_cam_lidar, q_cam_lidar
        )

        uv, cam_pts = project_lidar_to_image(
            points_lidar,
            translation=t_lidar_cam,
            quaternion=q_lidar_cam,
            intrinsics=intr_proj,
            image_shape=image_shape_hw,
        )

        image_h, image_w = image_shape_hw
        depth = np.zeros((image_h, image_w), dtype=np.float32)
        max_depth = 80.0

        for (u, v), pcam in zip(uv, cam_pts):
            ui = int(u)
            vi = int(v)
            z = float(pcam[2])
            if z <= 0.0 or z > max_depth:
                continue
            cur = depth[vi, ui]
            if cur == 0.0 or z < cur:
                depth[vi, ui] = z

        return depth

    def _prepare_rgb_depth_tensors(self, img_bgr_rs, depth):
        img_rgb = img_bgr_rs[:, :, ::-1].astype(np.float32) / 255.0
        rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_rgb = (img_rgb - rgb_mean) / rgb_std

        max_depth = 80.0
        depth_norm = depth.astype(np.float32) / max_depth

        rgb_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_t = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return rgb_t, depth_t

    def _infer_lcc_pose(self, model_wrapper, img_bgr, points_lidar, intr, ref_pose):
        t_ref_cam_lidar, q_ref_cam_lidar = ref_pose

        # Mirror the official/supervised_model_node preprocessing.
        if points_lidar.shape[0] > 0:
            x = points_lidar[:, 0]
            y = points_lidar[:, 1]
            keep = (x < -3.0) | (x > 3.0) | (y < -3.0) | (y > 3.0)
            points_lidar = points_lidar[keep]

        orig_h, orig_w = img_bgr.shape[:2]
        intr_proj = {
            "fx": intr["fx"],
            "fy": intr["fy"],
            "cx": intr["cx"],
            "cy": intr["cy"],
        }

        depth_orig = self._build_sparse_depth_from_pose(
            points_lidar=points_lidar,
            intr_proj=intr_proj,
            image_shape_hw=(orig_h, orig_w),
            t_cam_lidar=t_ref_cam_lidar,
            q_cam_lidar=q_ref_cam_lidar,
        )

        img_pad, depth_pad = self._pad_bottom_right_to_official_canvas(img_bgr, depth_orig)
        img_bgr_rs = cv2.resize(img_pad, (512, 256), interpolation=cv2.INTER_LINEAR)
        depth_rs = cv2.resize(depth_pad, (512, 256), interpolation=cv2.INTER_LINEAR)

        rgb_t, depth_t = self._prepare_rgb_depth_tensors(img_bgr_rs, depth_rs)
        transl_pred, rot_pred = model_wrapper.predict_delta(rgb_t, depth_t)

        transl_pred = transl_pred.detach().cpu().numpy().reshape(-1)
        rot_pred = rot_pred.detach().cpu().numpy().reshape(-1)
        rot_pred = quat_normalize_xyzw(rot_pred)

        R_delta = quaternion_to_rotation_matrix(rot_pred)
        t_new = (R_delta @ t_ref_cam_lidar.reshape(3, 1)).reshape(3,) + transl_pred.reshape(3,)
        q_new = quat_mul_xyzw(rot_pred, q_ref_cam_lidar)

        return t_new.astype(np.float32), q_new.astype(np.float32)

    ###############################################
    # BEV single-frame helpers
    ###############################################
    def _infer_bev_pose(self, model_wrapper, img_bgr, points_lidar, intr, reference_yaml_path):
        """
        BEV single-frame inference using the same preprocessing and transform
        conventions as calib_model_node.py.

        Output convention returned here:
          camera -> lidar
        """

        import torch

        # Match wrapper resize/intrinsics behavior exactly
        K = np.array(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        target_wh = (704, 256)
        img_rs, K_rs = _resize_and_scale_intrinsics(
            img_bgr,
            K,
            target_wh=target_wh,
        )

        # Match wrapper tensor formatting
        img_t = (
            torch.from_numpy(img_rs)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # (1,3,H,W)

        pc = np.asarray(points_lidar, dtype=np.float32).reshape(-1, 3)
        pc_t = (
            torch.from_numpy(pc)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # (1,N,3)

        K_t = (
            torch.from_numpy(K_rs)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # (1,3,3)

        # Match wrapper init-transform behavior exactly:
        # YAML is camera->lidar, BEV wants lidar->camera init.
        with open(reference_yaml_path, "r") as f:
            ref_yaml = yaml.safe_load(f) or {}

        T_camera_to_lidar = _make_T_from_yaml_camera_to_lidar(ref_yaml)
        T_lidar_to_camera = np.linalg.inv(T_camera_to_lidar).astype(np.float32)

        init_T = (
            torch.from_numpy(T_lidar_to_camera)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        pred_T_lidar_to_camera = model_wrapper.predict_T(
            img_bchw=img_t,
            pc_bnx3=pc_t,
            K_bx3x3=K_t,
            init_T_bx4x4=init_T,
        )

        T_lidar_to_camera_pred = (
            pred_T_lidar_to_camera[0].detach().cpu().numpy().astype(np.float32)
        )

        # Pipeline convention is camera -> lidar
        T_camera_to_lidar_pred = np.linalg.inv(T_lidar_to_camera_pred).astype(np.float32)

        t = T_camera_to_lidar_pred[:3, 3]
        R = T_camera_to_lidar_pred[:3, :3]
        q = self._rotation_matrix_to_quaternion_xyzw(R)
        q = quat_normalize_xyzw(q)

        return t.astype(np.float32), q.astype(np.float32)

    def _rotation_matrix_to_quaternion_xyzw(self, R):
        """
        Robust matrix -> quaternion conversion, xyzw convention.
        """
        R = np.asarray(R, dtype=np.float64)
        tr = float(np.trace(R))

        if tr > 0.0:
            S = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    ###############################################
    # Common inference entry
    ###############################################
    def _infer_pose_for_sample(self, model_name: str, model_wrapper, sample: Dict, ref_pose):
        img_bgr = sample["image"]
        intr = self._sample_to_intrinsics(sample)
        points = np.asarray(sample["points"], dtype=np.float32).reshape(-1, 3)

        if model_name in ["lccnet", "lcc"]:
            return self._infer_lcc_pose(
                model_wrapper=model_wrapper,
                img_bgr=img_bgr,
                points_lidar=points,
                intr=intr,
                ref_pose=ref_pose,
            )

        if model_name in ["bev", "bevcalib", "bev_calib"]:
            return self._infer_bev_pose(
                model_wrapper=model_wrapper,
                img_bgr=img_bgr,
                points_lidar=points,
                intr=intr,
                reference_yaml_path=self.archive_meta["reference_yaml_path"],
            )

        raise RuntimeError(f"Unsupported model_name for visualizer archive reconstruction: {model_name}")

    ###############################################
    # Drawing helpers
    ###############################################
    def _draw_projection(self, image_bgr, points_lidar, intr, pose_cam_to_lidar, color_bgr):
        """
        Input pose convention is camera -> lidar, matching archive/reference
        conventions in this project. Projection requires lidar -> camera, so we
        invert internally before calling project_lidar_to_image().
        """
        t_cam_lidar, q_cam_lidar = pose_cam_to_lidar
        t_lidar_cam, q_lidar_cam = invert_extrinsics_cam_to_lidar_to_lidar_to_cam(
            t_cam_lidar, q_cam_lidar
        )

        uv, _ = project_lidar_to_image(
            points_lidar,
            translation=t_lidar_cam,
            quaternion=q_lidar_cam,
            intrinsics=intr,
            image_shape=image_bgr.shape[:2],
        )

        out = image_bgr.copy()

        for u, v in uv:
            ui = int(u)
            vi = int(v)
            if 0 <= ui < out.shape[1] and 0 <= vi < out.shape[0]:
                cv2.circle(out, (ui, vi), self.point_radius_px, color_bgr, -1)

        return out

    def _make_metadata_text(self, default_mode_label: str) -> str:
        user_text = str(self.get_parameter("metadata_text").value).strip()
        if user_text:
            return user_text

        if self.input_mode == "archive_reconstruct":
            model = str(self.archive_meta.get("model", ""))
            train_ds = str(self.archive_meta.get("train_dataset", ""))
            eval_ds = str(self.archive_meta.get("eval_dataset", ""))
            ckpt = os.path.basename(str(self.archive_meta.get("checkpoint", "")))
            sample_key = str(self.archive_meta.get("sample_key", ""))
            return f"{default_mode_label} | {model} | train={train_ds} | eval={eval_ds} | {ckpt} | {sample_key}"

        return default_mode_label

    def _draw_bottom_banner(self, image_bgr, text: str) -> np.ndarray:
        out = image_bgr.copy()
        h, w = out.shape[:2]
        banner_h = min(self.bottom_banner_height_px, max(30, h // 6))

        # Dark grey banner
        cv2.rectangle(
            out,
            (0, h - banner_h),
            (w - 1, h - 1),
            color=(40, 40, 40),
            thickness=-1,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        ((tw, th), _) = cv2.getTextSize(text, font, scale, thickness)
        x = max(8, (w - tw) // 2)
        y = h - max(10, (banner_h - th) // 2)

        cv2.putText(
            out,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return out

    def _publish_overlay_pair(self, ref_img_bgr, est_img_bgr):
        ref_msg = self.bridge.cv2_to_imgmsg(ref_img_bgr, encoding="bgr8")
        est_msg = self.bridge.cv2_to_imgmsg(est_img_bgr, encoding="bgr8")

        now = self.get_clock().now().to_msg()
        ref_msg.header.stamp = now
        est_msg.header.stamp = now

        self.ref_pub.publish(ref_msg)
        self.est_pub.publish(est_msg)

    ###############################################
    # Live rendering
    ###############################################
    def _render_live_tick(self):
        if self.latest_image_msg is None:
            return
        if self.latest_points_msg is None:
            return
        if self.latest_caminfo_msg is None:
            return
        if self.latest_ref_msg is None:
            return
        if self.latest_est_msg is None:
            return

        try:
            img_bgr = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Failed to decode live image: {e}")
            return

        points = self._pointcloud2_to_xyz(self.latest_points_msg)
        intr = self._caminfo_to_intrinsics(self.latest_caminfo_msg)
        ref_pose = self._pose_from_tf_msg(self.latest_ref_msg)
        est_pose = self._pose_from_tf_msg(self.latest_est_msg)

        ref_img = self._draw_projection(img_bgr, points, intr, ref_pose, self.reference_color_bgr)
        est_img = self._draw_projection(img_bgr, points, intr, est_pose, self.estimated_color_bgr)

        ref_img = self._draw_bottom_banner(ref_img, self._make_metadata_text("REFERENCE"))
        est_img = self._draw_bottom_banner(est_img, self._make_metadata_text("ESTIMATED"))

        self._publish_overlay_pair(ref_img, est_img)

    ###############################################
    # Archive reconstruction rendering
    ###############################################
    def _render_archive_tick(self):
        if self.archive_sample is None or self.archive_ref_pose is None or self.archive_est_pose is None:
            return

        img_bgr = self.archive_sample["image"]
        intr = self._sample_to_intrinsics(self.archive_sample)
        points = np.asarray(self.archive_sample["points"], dtype=np.float32).reshape(-1, 3)

        ref_img = self._draw_projection(img_bgr, points, intr, self.archive_ref_pose, self.reference_color_bgr)
        est_img = self._draw_projection(img_bgr, points, intr, self.archive_est_pose, self.estimated_color_bgr)

        ref_img = self._draw_bottom_banner(ref_img, self._make_metadata_text("REFERENCE"))
        est_img = self._draw_bottom_banner(est_img, self._make_metadata_text("ESTIMATED"))

        self._publish_overlay_pair(ref_img, est_img)
        self.archive_render_done = True


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicProjectionVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()