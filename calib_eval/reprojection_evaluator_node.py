###############################################
# Evaluator: Reprojection metrics
###############################################
# Subscribes:
#   /eval/clean/image
#   /eval/clean/(points|scan)   (selected via use_dynamic_rig)
#   /eval/estimated_extrinsics  (TransformStamped)
#
# Publishes:
#   /eval/metrics/reprojection_visibility
#   /eval/metrics/reprojection_mean_edge_dist_px   (legacy name)
#   /eval/metrics/reprojection_pixel_error_px      (clearer name)
###############################################

import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, LaserScan
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import numpy as np
from calib_eval import dynamic_utils as dyn_utils

try:
    from geometry_utils import (
        project_lidar_to_image,
        compute_edge_distance_transform,
        mean_edge_distance_px,
        invert_extrinsics_cam_to_lidar_to_lidar_to_cam,
        laserscan_to_points_xy_plane
    )
except Exception:
    from .geometry_utils import (   # type: ignore
        project_lidar_to_image,
        compute_edge_distance_transform,
        mean_edge_distance_px,
        invert_extrinsics_cam_to_lidar_to_lidar_to_cam,
        laserscan_to_points_xy_plane
    )


class ReprojectionEvaluatorNode(Node):

    def __init__(self):
        super().__init__('reprojection_evaluator_node')
        self.bridge = CvBridge()

        # Rig selection + topics
        self.declare_parameter('use_dynamic_rig', False)
        self.declare_parameter('dynamic_representation', '')
        self.declare_parameter('image_topic', '/eval/clean/image')
        self.declare_parameter('lidar_topic', '')  # optional override
        self.declare_parameter('extrinsics_topic', '/eval/estimated_extrinsics')

        # Config
        self.declare_parameter('evaluation_config_path', '')

        # Load per-run intrinsics from CameraInfo if available
        self.declare_parameter('camera_info_topic', '/eval/camera_info')

        image_topic = str(self.get_parameter('image_topic').value).strip()
        lidar_topic = self._resolve_lidar_topic()
        extr_topic = str(self.get_parameter('extrinsics_topic').value).strip()
        cfg_path = self._resolve_eval_config_path(str(self.get_parameter('evaluation_config_path').value).strip())
        cam_info_topic = str(self.get_parameter('camera_info_topic').value).strip()

        self.config = self._load_eval_config(cfg_path)
        self.intrinsics = self.config.get('camera', {'fx': 500.0, 'fy': 500.0, 'cx': 320.0, 'cy': 240.0})
        self.canny_low = int(self.config.get('edge', {}).get('canny_low', 100))
        self.canny_high = int(self.config.get('edge', {}).get('canny_high', 200))

        # Subscribers
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

        use_scan_input = self._use_scan_input(lidar_topic)
        if use_scan_input:
            self.lidar_sub = self.create_subscription(LaserScan, lidar_topic, self.scan_callback, 10)
        else:
            self.lidar_sub = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, 10)

        self.tf_sub = self.create_subscription(TransformStamped, extr_topic, self.tf_callback, 10)

        # CameraInfo subscription (optional; YAML remains fallback)
        self.cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self.camera_info_callback, 10)

        # Publishers
        self.pub_vis = self.create_publisher(Float32, '/eval/metrics/reprojection_visibility', 10)
        self.pub_px_legacy = self.create_publisher(Float32, '/eval/metrics/reprojection_mean_edge_dist_px', 10)
        self.pub_px = self.create_publisher(Float32, '/eval/metrics/reprojection_pixel_error_px', 10)

        # Buffers
        self.latest_image = None
        self.latest_points = None
        self.latest_tf = None

        # store latest camera info (per-run intrinsics)
        self.latest_camera_info = None

        # Debug counters
        self.debug_counter = 0
        self.debug_log_every = 20

        self.get_logger().info(
            "ReprojectionEvaluatorNode started.\n"
            f"  image_topic={image_topic}\n"
            f"  lidar_topic={lidar_topic}\n"
            f"  extrinsics_topic={extr_topic}\n"
            f"  use_dynamic_rig={bool(self.get_parameter('use_dynamic_rig').value)}\n"
            f"  dynamic_representation={str(self.get_parameter('dynamic_representation').value).strip() or '<auto>'}\n"
            f"  evaluation_config_path={cfg_path}\n"
            f"  camera_info_topic={cam_info_topic} (preferred if available; YAML fallback)"
        )

    def _use_scan_input(self, resolved_lidar_topic=None):
        """
        Decide whether the lidar input topic should be treated as LaserScan or PointCloud2.

        Dynamic-rig evaluation is representation-dependent:
          - raw_scan      -> LaserScan
          - pseudo_points -> PointCloud2

        We also
        infer from the resolved topic name when possible:
          - .../scan   -> LaserScan
          - .../points -> PointCloud2

        If dynamic_representation is unset and the topic is ambiguous:
          - dynamic rig defaults to LaserScan
          - static rig defaults to PointCloud2
        """
        dynamic_representation = str(self.get_parameter('dynamic_representation').value).strip().lower()
        if dynamic_representation == 'raw_scan':
            return True
        if dynamic_representation == 'pseudo_points':
            return False

        topic = (resolved_lidar_topic or '').strip()
        if topic.endswith('/scan'):
            return True
        if topic.endswith('/points'):
            return False

        use_dynamic = bool(self.get_parameter('use_dynamic_rig').value)
        return use_dynamic

    def _resolve_lidar_topic(self):
        explicit = str(self.get_parameter('lidar_topic').value).strip()
        if explicit:
            return explicit
        return '/eval/clean/scan' if self._use_scan_input() else '/eval/clean/points'

    def _resolve_eval_config_path(self, user_path: str) -> str:
        """
        Finds evaluation.yaml robustly.
        Priority:
          1) user provided param path (if exists)
          2) ../config/evaluation.yaml relative to this script
          3) ./config/evaluation.yaml relative to CWD
          4) ~/evaluation.yaml (legacy)
        """
        def exists(p): return p and os.path.exists(os.path.expanduser(p))

        if exists(user_path):
            return os.path.expanduser(user_path)

        here = os.path.dirname(os.path.abspath(__file__))
        candidate_repo = os.path.normpath(os.path.join(here, '..', 'config', 'evaluation.yaml'))
        if os.path.exists(candidate_repo):
            return candidate_repo

        candidate_cwd = os.path.join(os.getcwd(), 'config', 'evaluation.yaml')
        if os.path.exists(candidate_cwd):
            return candidate_cwd

        legacy = os.path.expanduser('~/evaluation.yaml')
        return legacy

    def _load_eval_config(self, path):
        defaults = {
            'camera': {'fx': 500.0, 'fy': 500.0, 'cx': 320.0, 'cy': 240.0},
            'edge': {'canny_low': 100, 'canny_high': 200},
        }
        try:
            p = os.path.expanduser(path)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                for k, v in defaults.items():
                    if k not in loaded:
                        loaded[k] = v
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if sub_k not in loaded[k]:
                                loaded[k][sub_k] = sub_v
                return loaded
            else:
                self.get_logger().warn(f"evaluation.yaml not found at: {p} (using defaults)")
        except Exception as e:
            self.get_logger().warn(f"Failed to read evaluation config {path}: {e}")
        return defaults

    ###############################################
    # CameraInfo callback
    ###############################################
    def camera_info_callback(self, msg: CameraInfo):
        """
        Store latest CameraInfo. When available, we use this for intrinsics
        instead of evaluation.yaml, since it reflects per-run recorded intrinsics.
        """
        self.latest_camera_info = msg

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.try_compute()

    def lidar_callback(self, msg):
        pts = []
        for p in point_cloud2.read_points(msg, skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        self.latest_points = np.asarray(pts, dtype=np.float32)
        self.try_compute()

    def scan_callback(self, msg):
        """
        Dynamic-rig LaserScan conversion must use the same helper as the 2D
        calibration path. Earlier, the evaluator used geometry_utils
        laserscan_to_points_xy_plane(...), while offline2d/online2d used
        dynamic_utils.scan_dict_to_points_lidar_frame(...).

        That mismatch caused the projection/evaluation path to operate on a
        different scan embedding than the optimization path, which contributed
        to reprojection breakdown.
        """
        scan_dict = {
            "ranges": list(msg.ranges),
            "intensities": list(msg.intensities),
            "angle_min": float(msg.angle_min),
            "angle_max": float(msg.angle_max),
            "angle_increment": float(msg.angle_increment),
            "time_increment": float(msg.time_increment),
            "scan_time": float(msg.scan_time),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
        }

        try:
            pts = dyn_utils.scan_dict_to_points_lidar_frame(scan_dict)
        except Exception as e:
            self.get_logger().warn(f"dynamic_utils.scan_dict_to_points_lidar_frame failed ({e}); falling back.")
            pts = laserscan_to_points_xy_plane(
                scan_dict["ranges"],
                scan_dict["angle_min"],
                scan_dict["angle_increment"]
            )

        self.latest_points = np.asarray(pts, dtype=np.float32)
        self.try_compute()

    def tf_callback(self, msg):
        t = msg.transform.translation
        q = msg.transform.rotation
        self.latest_tf = (
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
        )
        self.try_compute()

    def _resolve_intrinsics(self):
        """
        Prefer live CameraInfo if available; otherwise use evaluation.yaml.
        """
        if self.latest_camera_info is not None:
            k = np.array(self.latest_camera_info.k, dtype=np.float32).reshape(3, 3)
            return {
                'fx': float(k[0, 0]),
                'fy': float(k[1, 1]),
                'cx': float(k[0, 2]),
                'cy': float(k[1, 2]),
            }
        return self.intrinsics

    def try_compute(self):
        if self.latest_image is None or self.latest_points is None or self.latest_tf is None:
            self.debug_counter += 1
            if self.debug_counter <= 10 or (self.debug_counter % self.debug_log_every == 0):
                self.get_logger().info(
                    "[DEBUG reproj precheck] "
                    f"have_image={self.latest_image is not None} "
                    f"have_points={self.latest_points is not None} "
                    f"have_tf={self.latest_tf is not None} "
                    f"have_camera_info={self.latest_camera_info is not None}"
                )
            return

        intrinsics = self._resolve_intrinsics()
        if intrinsics is None:
            self.debug_counter += 1
            if self.debug_counter <= 10 or (self.debug_counter % self.debug_log_every == 0):
                self.get_logger().info(
                    "[DEBUG reproj precheck] "
                    f"have_image={self.latest_image is not None} "
                    f"have_points={self.latest_points is not None} "
                    f"have_tf={self.latest_tf is not None} "
                    f"have_camera_info={self.latest_camera_info is not None} "
                    "intrinsics_resolved=False"
                )
            return

        t, q = self.latest_tf
        t_lidar_cam, q_lidar_cam = invert_extrinsics_cam_to_lidar_to_lidar_to_cam(t, q)

        projected_uv, _ = project_lidar_to_image(
            self.latest_points, t_lidar_cam, q_lidar_cam, intrinsics, self.latest_image.shape
        )

        visibility = float(len(projected_uv)) / float(len(self.latest_points))

        _, dist = compute_edge_distance_transform(self.latest_image, self.canny_low, self.canny_high)
        mean_px, _ = mean_edge_distance_px(dist, projected_uv)

        # --------------------------------------------------
        # Debug logging for dynamic pseudo-points diagnosis
        # --------------------------------------------------
        self.debug_counter += 1
        if self.debug_counter <= 5 or (self.debug_counter % self.debug_log_every == 0):
            z_min = float(np.min(self.latest_points[:, 2])) if self.latest_points.ndim == 2 and self.latest_points.shape[1] >= 3 else float('nan')
            z_max = float(np.max(self.latest_points[:, 2])) if self.latest_points.ndim == 2 and self.latest_points.shape[1] >= 3 else float('nan')
            sample_uv = projected_uv[:3].tolist() if len(projected_uv) > 0 else []

            self.get_logger().info(
                "[DEBUG reproj] "
                f"points_total={len(self.latest_points)} "
                f"projected_valid={len(projected_uv)} "
                f"visibility={visibility:.6f} "
                f"mean_px={float(mean_px)} "
                f"z_min={z_min:.6f} "
                f"z_max={z_max:.6f} "
                f"t_cam_to_lidar={[float(x) for x in t.tolist()]} "
                f"t_lidar_to_cam={[float(x) for x in t_lidar_cam.tolist()]} "
                f"sample_uv={sample_uv}"
            )

        self.pub_vis.publish(Float32(data=float(visibility)))
        self.pub_px_legacy.publish(Float32(data=float(mean_px)))
        self.pub_px.publish(Float32(data=float(mean_px)))

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ReprojectionEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()