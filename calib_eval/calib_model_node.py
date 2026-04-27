###############################################
# BEVCalib wrapper node
###############################################

import os
import yaml

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped

from calib_eval.models.registry import create_model_from_registry, set_global_determinism


def _K_from_camera_info(msg: CameraInfo) -> np.ndarray:
    K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
    return K


def _resize_and_scale_intrinsics(
    img_bgr: np.ndarray,
    K: np.ndarray,
    target_wh=(704, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirrors inference_kitti.py behavior: resize + scale intrinsics.
    target_wh = (W, H)
    """
    H0, W0 = img_bgr.shape[:2]
    Wt, Ht = int(target_wh[0]), int(target_wh[1])

    img_resized = cv2.resize(img_bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    sx = float(Wt) / float(W0)
    sy = float(Ht) / float(H0)

    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return img_resized, K2


def _quat_xyzw_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = [float(v) for v in q]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _make_T_from_yaml_camera_to_lidar(yaml_dict: dict) -> np.ndarray:
    t = np.asarray(yaml_dict["translation"], dtype=np.float32).reshape(3)
    q = np.asarray(yaml_dict["rotation_quat"], dtype=np.float32).reshape(4)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_xyzw_to_R(q)
    T[:3, 3] = t
    return T


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class CalibModelNode(Node):

    def __init__(self):
        super().__init__("calib_model_node")
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter("image_topic", "/eval/clean/image")
        self.declare_parameter("points_topic", "/eval/clean/points")
        self.declare_parameter("lidar_topic", "")
        self.declare_parameter("camera_info_topic", "/eval/camera_info")
        self.declare_parameter("estimated_extrinsics_topic", "/eval/estimated_extrinsics")

        # Reference / initialization inputs
        self.declare_parameter("reference_yaml_path", "")
        self.declare_parameter("evaluation_config_path", "")
        self.declare_parameter("rig_config_id", "static_mount_h1")

        # Model
        self.declare_parameter("model_name", "bevcalib")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("seed", 42)
        self.declare_parameter("force_determinism", True)

        # Preprocess
        self.declare_parameter("target_width", 704)
        self.declare_parameter("target_height", 256)
        self.declare_parameter("max_points", 0)  # 0 = keep all (can be heavy)

        # Determinism
        set_global_determinism(
            seed=int(self.get_parameter("seed").value),
            force_determinism=bool(self.get_parameter("force_determinism").value),
            logger=self.get_logger(),
        )

        # Model load
        self.model = create_model_from_registry(
            model_name=str(self.get_parameter("model_name").value),
            checkpoint_path=str(self.get_parameter("checkpoint_path").value),
            device=str(self.get_parameter("device").value),
            logger=self.get_logger(),
        )

        # Buffers
        self._img = None
        self._pc = None
        self._K = None

        image_topic = str(self.get_parameter("image_topic").value)
        points_topic = str(self.get_parameter("points_topic").value)
        lidar_topic = str(self.get_parameter("lidar_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        estimated_extrinsics_topic = str(
            self.get_parameter("estimated_extrinsics_topic").value
        )

        resolved_points_topic = points_topic.strip()
        if not resolved_points_topic:
            resolved_points_topic = lidar_topic.strip()
        if not resolved_points_topic:
            resolved_points_topic = "/eval/clean/points"

        self._init_T_lidar_to_camera = self._load_bev_init_transform()

        init_t = self._init_T_lidar_to_camera[:3, 3]
        self.get_logger().info(
            f"BEVCalib topics: image={image_topic}, points={resolved_points_topic}, "
            f"camera_info={camera_info_topic}, estimated_extrinsics={estimated_extrinsics_topic}"
        )
        self.get_logger().info(
            "BEVCalib init_T_to_camera loaded from reference baseline "
            f"(lidar->camera). translation=[{init_t[0]:.4f}, {init_t[1]:.4f}, {init_t[2]:.4f}]"
        )

        # ROS I/O
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(
            PointCloud2, resolved_points_topic, self._on_points, 10
        )
        self.create_subscription(
            CameraInfo, camera_info_topic, self._on_caminfo, 10
        )

        self.pub = self.create_publisher(
            TransformStamped,
            estimated_extrinsics_topic,
            10,
        )

    def _resolve_reference_yaml_path(self) -> str:
        direct_ref_path = str(self.get_parameter("reference_yaml_path").value).strip()
        eval_cfg_path = str(self.get_parameter("evaluation_config_path").value).strip()
        rig_config_id = str(self.get_parameter("rig_config_id").value).strip()

        def _exists(p: str) -> bool:
            return bool(p) and os.path.exists(os.path.expanduser(p))

        if _exists(direct_ref_path):
            return os.path.expanduser(direct_ref_path)

        if not eval_cfg_path:
            pkg_share = get_package_share_directory("calib_eval")
            eval_cfg_path = os.path.join(pkg_share, "config", "evaluation.yaml")

        eval_cfg_path = os.path.expanduser(eval_cfg_path)
        if os.path.exists(eval_cfg_path):
            cfg = _load_yaml(eval_cfg_path)
            ref = cfg.get("reference", {}) if isinstance(cfg, dict) else {}
            ref_path = str(ref.get("reference_yaml_path", "")).strip()
            if _exists(ref_path):
                return os.path.expanduser(ref_path)

        pkg_share = get_package_share_directory("calib_eval")
        ref_dir = os.path.join(pkg_share, "config", "reference")
        candidate = os.path.join(ref_dir, f"{rig_config_id}_reference.yaml")
        if os.path.exists(candidate):
            return candidate

        raise RuntimeError(
            "Could not resolve BEV reference YAML path. Provide reference_yaml_path, "
            "or evaluation_config_path with reference.reference_yaml_path, or rig_config_id."
        )

    def _load_bev_init_transform(self) -> np.ndarray:
        """
        BEV training uses:
          - gt_T_to_camera     = lidar -> camera
          - init_T_to_camera   = perturbed lidar -> camera

        Package reference YAMLs are stored as camera -> lidar.
        For BEV inference we therefore load the package baseline, build camera->lidar,
        and invert it once so the model receives lidar->camera initialization matching
        the training contract.
        """
        ref_yaml_path = self._resolve_reference_yaml_path()
        ref_yaml = _load_yaml(ref_yaml_path)
        T_camera_to_lidar = _make_T_from_yaml_camera_to_lidar(ref_yaml)
        T_lidar_to_camera = np.linalg.inv(T_camera_to_lidar).astype(np.float32)

        self.get_logger().info(
            f"BEVCalib reference init yaml resolved to: {ref_yaml_path}"
        )
        return T_lidar_to_camera

    def _on_image(self, msg: Image):
        try:
            self._img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image decode failed: {e}")
            return
        self._try_infer()

    def _on_caminfo(self, msg: CameraInfo):
        try:
            self._K = _K_from_camera_info(msg)
        except Exception as e:
            self.get_logger().warn(f"CameraInfo parse failed: {e}")
            return
        self._try_infer()

    def _on_points(self, msg: PointCloud2):
        pts = []
        for p in point_cloud2.read_points(msg, skip_nans=True):
            pts.append([float(p[0]), float(p[1]), float(p[2])])

        if not pts:
            self._pc = None
            return

        pc = np.asarray(pts, dtype=np.float32)
        max_points = int(self.get_parameter("max_points").value)
        if max_points > 0 and pc.shape[0] > max_points:
            rng = np.random.RandomState(0)
            idx = rng.choice(pc.shape[0], size=max_points, replace=False)
            pc = pc[idx]

        self._pc = pc
        self._try_infer()

    def _try_infer(self):
        if self._img is None or self._pc is None or self._K is None:
            return

        tw = int(self.get_parameter("target_width").value)
        th = int(self.get_parameter("target_height").value)
        img_rs, K_rs = _resize_and_scale_intrinsics(
            self._img,
            self._K,
            target_wh=(tw, th),
        )

        import torch

        device = str(self.get_parameter("device").value).strip().lower()
        torch_device = torch.device(device if device in ["cpu", "cuda"] else "cpu")

        img = (
            torch.from_numpy(img_rs)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(torch_device)
        )  # (1,3,H,W)

        pc = (
            torch.from_numpy(self._pc)
            .unsqueeze(0)
            .float()
            .to(torch_device)
        )  # (1,N,3)

        K = (
            torch.from_numpy(K_rs)
            .unsqueeze(0)
            .float()
            .to(torch_device)
        )  # (1,3,3)

        init_T = (
            torch.from_numpy(self._init_T_lidar_to_camera)
            .unsqueeze(0)
            .float()
            .to(torch_device)
        )

        try:
            T_pred_lidar_to_camera = self.model.predict_T(
                img_bchw=img,
                pc_bnx3=pc,
                K_bx3x3=K,
                init_T_bx4x4=init_T,
            )
        except Exception as e:
            self.get_logger().error(f"BEVCalib inference failed: {e}")
            return

        T_lidar_to_camera = (
            T_pred_lidar_to_camera[0].detach().cpu().numpy().astype(np.float32)
        )

        # The BEV model predicts lidar -> camera (gt_T_to_camera, init_T_to_camera).
        #
        # The rest of the pipeline publishes / compares transforms in the
        # package reference convention: camera -> lidar.
        #
        # Therefore we invert once here before publishing to keep BEV consistent with
        # the evaluator / reference publisher contract.
        T_camera_to_lidar = np.linalg.inv(T_lidar_to_camera).astype(np.float32)

        t = T_camera_to_lidar[:3, 3]
        R = T_camera_to_lidar[:3, :3]
        q = self._quat_from_R(R)  # xyzw

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.child_frame_id = "lidar_estimated"

        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])

        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])

        self.pub.publish(msg)

    def _quat_from_R(self, R: np.ndarray) -> np.ndarray:
        # Robust rotation matrix -> quaternion (xyzw)
        m = R
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])

        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S

        q = np.array([qx, qy, qz, qw], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        return q


def main(args=None):
    rclpy.init(args=args)
    node = CalibModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()