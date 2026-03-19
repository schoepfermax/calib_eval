###############################################
# Wraps a supervised calibration model and publishes estimated extrinsics.
###############################################
# Publishes estimated extrinsics to:
#   /eval/estimated_extrinsics  (TransformStamped)
#
# Inputs (default topics used by the pipeline):
#   /eval/clean/image     (sensor_msgs/Image)        BGR8
#   /eval/clean/points    (sensor_msgs/PointCloud2)  LiDAR points
#   /eval/camera_info     (sensor_msgs/CameraInfo)   intrinsics
#   /eval/ref_extrinsics  (geometry_msgs/TransformStamped) baseline estimate
#
# Current implementation:
#   - LCCNet inference (RGB + sparse depth image)
#   - Sparse depth: nearest-depth selection per pixel (KITTI-style projection)
#
# IMPORTANT:
#   - Reference extrinsics are NOT ground truth. They are Abdul tool baseline.
#   - By default we treat LCCNet output as a DELTA and compose it onto reference.
#
# Frame convention (IMPORTANT, deterministic):
#   - /eval/ref_extrinsics is expected to be a TF from camera -> lidar
#     (header.frame_id=camera, child_frame_id=lidar), consistent with
#     ReferenceExtrinsicsPublisher logs: parent_frame=camera, child_frame=lidar.
#   - Projection needs lidar -> camera, so we invert the reference transform for
#     depth projection only.
#   - Published /eval/estimated_extrinsics follows the same direction as the
#     reference topic (camera -> lidar).
###############################################

import os
import glob
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import TransformStamped

from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

import torch

from calib_eval.geometry_utils import (
    project_lidar_to_image,
    quaternion_to_rotation_matrix,
)

# NOTE:
# For consistency across all integrated models, we load LCCNet via the registry
# (calib_eval.models.registry), not by importing third_party directly in this node.
from calib_eval.models.registry import create_model_from_registry


# -----------------------------
# Small quaternion utilities
# -----------------------------
def quat_normalize_xyzw(q):
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


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


def invert_transform_xyzw(t, q):
    """
    Invert transform (parent->child) represented as child pose in parent:
      p_parent = R p_child + t
    Inverse:
      p_child = R^T (p_parent - t)
    Returns (t_inv, q_inv) for (child->parent).
    """
    q = quat_normalize_xyzw(q)
    R = quaternion_to_rotation_matrix(q)
    t = np.asarray(t, dtype=np.float32).reshape(3,)

    R_inv = R.T
    t_inv = (-R_inv @ t.reshape(3, 1)).reshape(3,)
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)  # inverse of unit quaternion
    q_inv = quat_normalize_xyzw(q_inv)
    return t_inv, q_inv


def _auto_find_lccnet_checkpoint(logger=None):
    """
    Find the highest available kitti_iter*.tar checkpoint in a workspace-safe way.

    Search order:
      1) If CALIB_EVAL_LCCNET_CKPT_DIR is set, search there.
      2) Search relative to this file location:
         <repo>/calib_eval/models/checkpoints/LCCNet/kitti_iter*.tar

    Returns "" if nothing found (caller decides how to handle).
    """
    ckpt_dir_env = os.environ.get("CALIB_EVAL_LCCNET_CKPT_DIR", "").strip()
    candidates = []

    if ckpt_dir_env:
        candidates += glob.glob(os.path.join(ckpt_dir_env, "kitti_iter*.tar"))

    # Relative-to-repo search (works on laptop/workstation if repo copied similarly)
    here = os.path.abspath(__file__)
    # this file: <repo>/calib_eval/supervised_model_node.py
    # checkpoints: <repo>/calib_eval/models/checkpoints/LCCNet/
    repo_root_guess = os.path.dirname(here)
    candidates += glob.glob(os.path.join(repo_root_guess, "models", "checkpoints", "LCCNet", "kitti_iter*.tar"))

    # Pick max iter number
    best = ""
    best_iter = -1
    for p in candidates:
        base = os.path.basename(p)
        # kitti_iter3.tar -> 3
        try:
            it = int(base.replace("kitti_iter", "").replace(".tar", ""))
        except Exception:
            continue
        if it > best_iter:
            best_iter = it
            best = p

    if logger is not None:
        if best:
            logger.info(f"Auto-selected LCCNet checkpoint: {best} (iter={best_iter})")
        else:
            logger.warn("Auto checkpoint selection found no kitti_iter*.tar candidates.")

    return best


def _auto_find_iterative_lccnet_checkpoints(n_steps, preferred_dir='', logger=None):
    """
    Find the first n_steps LCCNet iterative checkpoints in order:
      kitti_iter1.tar, kitti_iter2.tar, ... , kitti_iterN.tar

    Search order:
      1) CALIB_EVAL_LCCNET_CKPT_DIR
      2) relative-to-repo checkpoints directory

    Returns:
      list[str]: ordered checkpoint paths

    Raises:
      RuntimeError if any required checkpoint is missing.
    """
    ckpt_dir_env = os.environ.get("CALIB_EVAL_LCCNET_CKPT_DIR", "").strip()

    here = os.path.abspath(__file__)
    repo_root_guess = os.path.dirname(here)
    repo_ckpt_dir = os.path.join(repo_root_guess, "models", "checkpoints", "LCCNet")

    search_dirs = []
    if preferred_dir:
        search_dirs.append(preferred_dir)
    if ckpt_dir_env:
        search_dirs.append(ckpt_dir_env)
    search_dirs.append(repo_ckpt_dir)

    found = []
    missing = []

    for step in range(1, int(n_steps) + 1):
        base = f"kitti_iter{step}.tar"
        hit = ""
        for d in search_dirs:
            cand = os.path.join(d, base)
            if os.path.exists(cand):
                hit = cand
                break
        if hit:
            found.append(hit)
        else:
            missing.append(base)

    if missing:
        raise RuntimeError(
            "Iterative LCCNet mode requires all staged checkpoints. Missing: "
            + ", ".join(missing)
        )

    if logger is not None:
        logger.info("Iterative LCCNet checkpoints:")
        for idx, p in enumerate(found, start=1):
            logger.info(f"  stage {idx}: {p}")

    return found


class SupervisedModelNode(Node):

    def __init__(self):
        super().__init__('supervised_model_node')

        # -----------------------------
        # Topics
        # -----------------------------
        self.declare_parameter('image_topic', '/eval/clean/image')
        self.declare_parameter('lidar_topic', '/eval/clean/points')
        self.declare_parameter('camera_info_topic', '/eval/camera_info')
        self.declare_parameter('reference_extrinsics_topic', '/eval/ref_extrinsics')

        # -----------------------------
        # Model settings
        # -----------------------------
        self.declare_parameter('model_name', 'lccnet')

        # LCCNet requires fixed image_size to match fc layer.
        # Derived from checkpoint fc1.weight shape (67712) and LCCNet's internal formula:
        #   expected image_size = (256, 512) (H,W)
        self.declare_parameter('lccnet_image_h', 256)
        self.declare_parameter('lccnet_image_w', 512)

        # LCCNet checkpoint:
        # - If empty, we auto-pick the highest available kitti_iter*.tar from the local repo
        #   (or from CALIB_EVAL_LCCNET_CKPT_DIR if set).
        # - If provided, it must exist.
        #
        # IMPORTANT:
        # Do NOT hardcode workstation-only paths here (e.g., /mnt/storage1/...).
        self.declare_parameter('checkpoint_path', "")

        # Max depth used in training config (commonly 80.0 for KITTI-style configs)
        self.declare_parameter('max_depth_m', 80.0)

        # If true, treat network output as delta and compose with reference extrinsics.
        # If false, publish raw network output as estimated extrinsics.
        self.declare_parameter('compose_with_reference', True)

        # LCCNet runtime mode:
        #   - single_pass : current behavior, one model forward pass
        #   - iterative   : staged official-style refinement using iter1..iterN
        self.declare_parameter('lccnet_mode', 'single_pass')
        self.declare_parameter('lccnet_iterative_steps', 5)

        # Optional: explicitly override device ("auto", "cpu", "cuda")
        self.declare_parameter('device', 'auto')

        # -----------------------------
        # ROS I/O
        # -----------------------------
        image_topic = str(self.get_parameter('image_topic').value)
        lidar_topic = str(self.get_parameter('lidar_topic').value)
        cam_info_topic = str(self.get_parameter('camera_info_topic').value)
        ref_topic = str(self.get_parameter('reference_extrinsics_topic').value)

        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.lidar_sub = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, cam_info_topic, self.caminfo_callback, 10)
        self.ref_sub = self.create_subscription(TransformStamped, ref_topic, self.ref_callback, 10)

        self.tf_pub = self.create_publisher(TransformStamped, '/eval/estimated_extrinsics', 10)

        self.bridge = CvBridge()

        self.latest_image_msg = None
        self.latest_lidar_msg = None
        self.latest_caminfo_msg = None
        self.latest_ref_msg = None

        # -----------------------------
        # Load model (registry-based)
        # -----------------------------
        dev_param = str(self.get_parameter('device').value).strip().lower()
        if dev_param == "cpu":
            self.device = torch.device("cpu")
        elif dev_param == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' requested, but torch.cuda.is_available() is False.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.get_logger().info(f"Using device: {self.device}")

        self.model_name = str(self.get_parameter('model_name').value).strip().lower()
        if self.model_name != 'lccnet':
            raise RuntimeError(f"Only model_name='lccnet' is supported right now. Got: {self.model_name}")

        self.image_h = int(self.get_parameter('lccnet_image_h').value)
        self.image_w = int(self.get_parameter('lccnet_image_w').value)

        self.lccnet_mode = str(self.get_parameter('lccnet_mode').value).strip().lower()
        self.lccnet_iterative_steps = int(self.get_parameter('lccnet_iterative_steps').value)

        if self.lccnet_mode not in ['single_pass', 'iterative']:
            raise RuntimeError(
                "lccnet_mode must be one of ['single_pass', 'iterative']. "
                f"Got: {self.lccnet_mode}"
            )

        ckpt_path = str(self.get_parameter('checkpoint_path').value).strip()
        if not ckpt_path:
            ckpt_path = _auto_find_lccnet_checkpoint(logger=self.get_logger())

        if ckpt_path and (not os.path.exists(ckpt_path)):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        device_str = "cuda" if self.device.type == "cuda" else "cpu"

        if self.lccnet_mode == 'single_pass':
            # Registry returns a wrapper with predict_delta(rgb_bchw, depth_b1hw).
            self.model = create_model_from_registry(
                model_name="lccnet",
                checkpoint_path=ckpt_path,
                device=device_str,
                logger=self.get_logger(),
            )
            self.iter_models = []
            self.iter_checkpoint_paths = []
        else:
            # IMPORTANT:
            # Do NOT preload all iterative stage models onto CUDA at startup.
            # On the laptop GPU this can exhaust VRAM / cuDNN workspace and crash
            # at the first convolution. Instead, keep only checkpoint paths and
            # load one stage at a time during the iterative loop.
            self.model = None

            # Thesis workflow requirement:
            # If the user explicitly passes checkpoint_path, iterative mode should
            # reuse THAT checkpoint for every refinement step instead of silently
            # defaulting to built-in KITTI staged checkpoints. This keeps the
            # experiment comparable to the single-pass run of the same model.
            if ckpt_path:
                self.iter_checkpoint_paths = [ckpt_path] * int(self.lccnet_iterative_steps)
                self.get_logger().info(
                    "Iterative LCCNet mode will reuse the provided checkpoint for all stages:"
                )
                for idx, p in enumerate(self.iter_checkpoint_paths, start=1):
                    self.get_logger().info(f"  stage {idx}: {p}")
            else:
                preferred_dir = ''
                self.iter_checkpoint_paths = _auto_find_iterative_lccnet_checkpoints(
                    self.lccnet_iterative_steps,
                    preferred_dir=preferred_dir,
                    logger=self.get_logger(),
                )

            self.iter_models = []

        # Timer for inference (keeps logic simple; no hidden loops)
        self.declare_parameter('inference_rate_hz', 10.0)
        rate = float(self.get_parameter('inference_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.5), self._tick)

        self.get_logger().info(
            "SupervisedModelNode ready:\n"
            f"  image_topic={image_topic}\n"
            f"  lidar_topic={lidar_topic}\n"
            f"  camera_info_topic={cam_info_topic}\n"
            f"  ref_extrinsics_topic={ref_topic}\n"
            f"  checkpoint={ckpt_path if ckpt_path else '<none>'}\n"
            f"  lccnet_mode={self.lccnet_mode}\n"
            f"  iterative_steps={self.lccnet_iterative_steps if self.lccnet_mode == 'iterative' else 1}\n"
            f"  LCCNet image_size(H,W)=({self.image_h},{self.image_w})\n"
            "  expected /eval/ref_extrinsics direction: camera -> lidar\n"
        )

    # -----------------------------
    # ROS callbacks
    # -----------------------------
    def image_callback(self, msg):
        self.latest_image_msg = msg

    def lidar_callback(self, msg):
        self.latest_lidar_msg = msg

    def caminfo_callback(self, msg):
        self.latest_caminfo_msg = msg

    def ref_callback(self, msg):
        self.latest_ref_msg = msg


    def _scale_intrinsics_for_resized_image(self, intr, orig_shape_hw):
        """
        Scale CameraInfo intrinsics from original image size to current LCCNet size.
        """
        orig_h, orig_w = orig_shape_hw
        scale_x = float(self.image_w) / float(orig_w)
        scale_y = float(self.image_h) / float(orig_h)
        return {
            'fx': intr['fx'] * scale_x,
            'fy': intr['fy'] * scale_y,
            'cx': intr['cx'] * scale_x,
            'cy': intr['cy'] * scale_y,
        }

    def _pad_bottom_right_to_official_canvas(self, img_bgr, depth_img):
        """
        Match official KITTI evaluation preprocessing:
            - keep original top-left alignment
            - pad only on the bottom and right
            - target canvas = 384 x 1280

        Returns:
            img_pad   : padded BGR image
            depth_pad : padded sparse depth image
        """
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
            mode='constant',
            constant_values=0,
        )

        depth_pad = np.pad(
            depth_img,
            ((0, pad_bottom), (0, pad_right)),
            mode='constant',
            constant_values=0.0,
        )

        return img_pad, depth_pad

    def _build_sparse_depth_from_pose(self, points_lidar, intr_proj, image_shape_hw, t_cam_lidar, q_cam_lidar):
        """
        Build KITTI-style sparse depth image using the current camera->lidar pose.

        IMPORTANT:
        - Projection needs lidar->camera, so invert the pose internally.
        - Projection is done in the ORIGINAL image geometry first.
        - Padding/resizing are handled later to match official KITTI evaluation.
        """
        t_lidar_cam, q_lidar_cam = invert_transform_xyzw(t_cam_lidar, q_cam_lidar)

        uv, cam_pts = project_lidar_to_image(
            points_lidar,
            translation=t_lidar_cam,
            quaternion=q_lidar_cam,
            intrinsics=intr_proj,
            image_shape=image_shape_hw,
        )

        image_h, image_w = image_shape_hw
        depth = np.zeros((image_h, image_w), dtype=np.float32)
        max_depth = float(self.get_parameter('max_depth_m').value)

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
        """
        Prepare LCCNet input tensors using preprocessing aligned with the
        official KITTI evaluation path:

          rgb_t   : 1x3xHxW, RGB in [0,1] with ImageNet normalization
          depth_t : 1x1xHxW, sparse depth normalized by max_depth_m

        Official references:
          - Dataset loader normalizes RGB with mean/std:
                mean=[0.485, 0.456, 0.406]
                std =[0.229, 0.224, 0.225]
          - Evaluation path divides depth by max_depth before inference.
        """
        # RGB preprocessing: BGR -> RGB, [0,1], then ImageNet normalization.
        img_rgb = img_bgr_rs[:, :, ::-1].astype(np.float32) / 255.0
        rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_rgb = (img_rgb - rgb_mean) / rgb_std

        # Depth preprocessing: normalize by max_depth, matching official eval.
        max_depth = float(self.get_parameter('max_depth_m').value)
        if max_depth <= 0.0:
            raise RuntimeError(f"max_depth_m must be > 0. Got: {max_depth}")
        depth_norm = depth.astype(np.float32) / max_depth

        rgb_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_t = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return rgb_t, depth_t

    def _run_lccnet_once(self, model_wrapper, img_bgr_rs, depth):
        """
        Run one LCCNet stage and return numpy delta outputs.
        """
        rgb_t, depth_t = self._prepare_rgb_depth_tensors(img_bgr_rs, depth)

        transl_pred, rot_pred = model_wrapper.predict_delta(rgb_t, depth_t)

        transl_pred = transl_pred.detach().cpu().numpy().reshape(-1)
        rot_pred = rot_pred.detach().cpu().numpy().reshape(-1)
        rot_pred = quat_normalize_xyzw(rot_pred)

        return transl_pred, rot_pred

    def _load_iterative_stage_model(self, checkpoint_path):
        """
        Lazily load one iterative LCCNet stage model.

        This keeps peak GPU memory lower than preloading all stage models
        simultaneously. The returned wrapper is used for one forward pass and
        then released by the caller.
        """
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        return create_model_from_registry(
            model_name="lccnet",
            checkpoint_path=checkpoint_path,
            device=device_str,
            logger=self.get_logger(),
        )

    def _compose_delta_onto_pose(self, t_pose_cam_lidar, q_pose_cam_lidar, transl_pred, rot_pred):
        """
        Compose delta (camera->lidar convention) onto current pose.
        """
        R_delta = quaternion_to_rotation_matrix(rot_pred)
        t_new = (R_delta @ t_pose_cam_lidar.reshape(3, 1)).reshape(3,) + transl_pred.reshape(3,)
        q_new = quat_mul_xyzw(rot_pred, q_pose_cam_lidar)
        return t_new, q_new


    # -----------------------------
    # Inference tick
    # -----------------------------
    def _tick(self):
        if self.latest_image_msg is None:
            return
        if self.latest_lidar_msg is None:
            return
        if self.latest_caminfo_msg is None:
            return
        if self.latest_ref_msg is None:
            return

        # 1) Decode image (BGR uint8)
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to decode image: {e}")
            return

        # 2) Extract intrinsics from CameraInfo
        K = self.latest_caminfo_msg.k  # length 9
        intr = {
            'fx': float(K[0]),
            'fy': float(K[4]),
            'cx': float(K[2]),
            'cy': float(K[5]),
        }

        # 3) Convert PointCloud2 -> Nx3 float32
        pts = []
        try:
            for p in point_cloud2.read_points(self.latest_lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True):
                pts.append([float(p[0]), float(p[1]), float(p[2])])
        except Exception as e:
            self.get_logger().warn(f"Failed to read PointCloud2: {e}")
            return

        points_lidar = np.asarray(pts, dtype=np.float32)

        # Official KITTI loader removes near-sensor central points:
        # keep points where x < -3 or x > 3 or y < -3 or y > 3
        if points_lidar.shape[0] > 0:
            x = points_lidar[:, 0]
            y = points_lidar[:, 1]
            keep = (x < -3.0) | (x > 3.0) | (y < -3.0) | (y > 3.0)
            points_lidar = points_lidar[keep]

        if points_lidar.shape[0] == 0:
            self.get_logger().warn("No lidar points available for inference after KITTI filtering.")
            return

        # 4) Keep ORIGINAL image geometry first.
        # Official KITTI eval projects depth in original image coordinates,
        # then pads to 384x1280, then resizes to 256x512.
        import cv2
        orig_h, orig_w = img_bgr.shape[:2]

        # 5) Reference extrinsics (expected camera -> lidar)
        ref = self.latest_ref_msg
        t_ref_cam_lidar = np.array([ref.transform.translation.x,
                                    ref.transform.translation.y,
                                    ref.transform.translation.z], dtype=np.float32)

        q_ref_cam_lidar = np.array([ref.transform.rotation.x,
                                    ref.transform.rotation.y,
                                    ref.transform.rotation.z,
                                    ref.transform.rotation.w], dtype=np.float32)
        q_ref_cam_lidar = quat_normalize_xyzw(q_ref_cam_lidar)

        intr_proj = {
            'fx': intr['fx'],
            'fy': intr['fy'],
            'cx': intr['cx'],
            'cy': intr['cy'],
        }

        compose = bool(self.get_parameter('compose_with_reference').value)

        if self.lccnet_mode == 'single_pass':
            # 1) Build sparse depth in ORIGINAL image geometry
            depth_orig = self._build_sparse_depth_from_pose(
                points_lidar=points_lidar,
                intr_proj=intr_proj,
                image_shape_hw=(orig_h, orig_w),
                t_cam_lidar=t_ref_cam_lidar,
                q_cam_lidar=q_ref_cam_lidar,
            )

            # 2) Pad RGB + depth to official KITTI canvas
            img_pad, depth_pad = self._pad_bottom_right_to_official_canvas(img_bgr, depth_orig)

            # 3) Resize both to LCCNet input size
            img_bgr_rs = cv2.resize(img_pad, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
            depth_rs = cv2.resize(depth_pad, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

            transl_pred, rot_pred = self._run_lccnet_once(
                self.model,
                img_bgr_rs,
                depth_rs,
            )

            if compose:
                t_new, q_new = self._compose_delta_onto_pose(
                    t_ref_cam_lidar,
                    q_ref_cam_lidar,
                    transl_pred,
                    rot_pred,
                )
            else:
                t_new = transl_pred.reshape(3,)
                q_new = rot_pred

        else:
            # Iterative staged refinement:
            # start from current reference pose and update pose after each stage.
            t_cur = t_ref_cam_lidar.copy()
            q_cur = q_ref_cam_lidar.copy()

            for stage_idx, stage_ckpt in enumerate(self.iter_checkpoint_paths):
                # 1) Build sparse depth in ORIGINAL image geometry for current pose
                depth_orig = self._build_sparse_depth_from_pose(
                    points_lidar=points_lidar,
                    intr_proj=intr_proj,
                    image_shape_hw=(orig_h, orig_w),
                    t_cam_lidar=t_cur,
                    q_cam_lidar=q_cur,
                )

                # 2) Pad RGB + depth to official KITTI canvas
                img_pad, depth_pad = self._pad_bottom_right_to_official_canvas(img_bgr, depth_orig)

                # 3) Resize both to LCCNet input size
                img_bgr_rs = cv2.resize(img_pad, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
                depth_rs = cv2.resize(depth_pad, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

                model_stage = self._load_iterative_stage_model(stage_ckpt)
                try:
                    transl_pred, rot_pred = self._run_lccnet_once(
                        model_stage,
                        img_bgr_rs,
                        depth_rs,
                    )
                finally:
                    del model_stage
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                if compose:
                    t_cur, q_cur = self._compose_delta_onto_pose(
                        t_cur,
                        q_cur,
                        transl_pred,
                        rot_pred,
                    )
                else:
                    t_cur = transl_pred.reshape(3,)
                    q_cur = rot_pred

            t_new = t_cur
            q_new = q_cur

        # 11) Publish TransformStamped (same direction as reference: camera -> lidar)
        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = ref.header.frame_id
        out.child_frame_id = ref.child_frame_id

        out.transform.translation.x = float(t_new[0])
        out.transform.translation.y = float(t_new[1])
        out.transform.translation.z = float(t_new[2])

        out.transform.rotation.x = float(q_new[0])
        out.transform.rotation.y = float(q_new[1])
        out.transform.rotation.z = float(q_new[2])
        out.transform.rotation.w = float(q_new[3])

        self.tf_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SupervisedModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()