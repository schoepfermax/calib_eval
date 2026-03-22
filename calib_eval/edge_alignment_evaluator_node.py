###############################################
# Evaluator: Edge alignment metrics
###############################################
# Subscribes:
#   /eval/clean/image
#   /eval/clean/(points|scan)   (selected via use_dynamic_rig)
#   /eval/estimated_extrinsics
#
# Publishes:
#   /eval/metrics/edge_hit_ratio
#   /eval/metrics/edge_mean_edge_dist_px
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
        edge_hit_ratio_with_threshold,
        invert_extrinsics_cam_to_lidar_to_lidar_to_cam,
        laserscan_to_points_xy_plane
    )
except Exception:
    from .geometry_utils import (   # type: ignore
        project_lidar_to_image,
        compute_edge_distance_transform,
        mean_edge_distance_px,
        edge_hit_ratio_with_threshold,
        invert_extrinsics_cam_to_lidar_to_lidar_to_cam,
        laserscan_to_points_xy_plane
    )


class EdgeAlignmentEvaluatorNode(Node):

    def __init__(self):
        super().__init__('edge_alignment_evaluator_node')
        self.bridge = CvBridge()

        self.declare_parameter('use_dynamic_rig', False)
        self.declare_parameter('image_topic', '/eval/clean/image')
        self.declare_parameter('lidar_topic', '')
        self.declare_parameter('extrinsics_topic', '/eval/estimated_extrinsics')
        self.declare_parameter('evaluation_config_path', '')

        # Load per-run intrinsics from CameraInfo if available
        self.declare_parameter('camera_info_topic', '/eval/camera_info')

        # Threshold used for edge_hit_ratio:
        # a point is counted as a hit if it lands within this many pixels
        # of the nearest image edge in the distance transform.
        # If this parameter is <= 0, we fall back to evaluation.yaml or default.
        self.declare_parameter('edge_hit_threshold_px', -1.0)

        image_topic = str(self.get_parameter('image_topic').value).strip()
        lidar_topic = self._resolve_lidar_topic()
        extr_topic = str(self.get_parameter('extrinsics_topic').value).strip()
        cfg_path = self._resolve_eval_config_path(str(self.get_parameter('evaluation_config_path').value).strip())
        cam_info_topic = str(self.get_parameter('camera_info_topic').value).strip()

        self.config = self._load_eval_config(cfg_path)
        self.intrinsics = self.config.get('camera', {'fx': 500.0, 'fy': 500.0, 'cx': 320.0, 'cy': 240.0})
        self.canny_low = int(self.config.get('edge', {}).get('canny_low', 100))
        self.canny_high = int(self.config.get('edge', {}).get('canny_high', 200))

        configured_hit_threshold = float(self.config.get('edge', {}).get('hit_threshold_px', 2.0))
        param_hit_threshold = float(self.get_parameter('edge_hit_threshold_px').value)
        if param_hit_threshold > 0.0:
            self.edge_hit_threshold_px = param_hit_threshold
        else:
            self.edge_hit_threshold_px = configured_hit_threshold

        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

        use_dynamic = bool(self.get_parameter('use_dynamic_rig').value)
        if use_dynamic:
            self.lidar_sub = self.create_subscription(LaserScan, lidar_topic, self.scan_callback, 10)
        else:
            self.lidar_sub = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, 10)

        self.tf_sub = self.create_subscription(TransformStamped, extr_topic, self.tf_callback, 10)

        # CameraInfo subscription (YAML remains fallback)
        self.cam_info_sub = self.create_subscription(CameraInfo, cam_info_topic, self.camera_info_callback, 10)

        self.pub_ratio = self.create_publisher(Float32, '/eval/metrics/edge_hit_ratio', 10)
        self.pub_px = self.create_publisher(Float32, '/eval/metrics/edge_mean_edge_dist_px', 10)

        self.latest_image = None
        self.latest_points = None
        self.latest_tf = None

        # store latest camera info (per-run intrinsics)
        self.latest_camera_info = None

        self.get_logger().info(
            "EdgeAlignmentEvaluatorNode started.\n"
            f"  image_topic={image_topic}\n"
            f"  lidar_topic={lidar_topic}\n"
            f"  extrinsics_topic={extr_topic}\n"
            f"  use_dynamic_rig={bool(self.get_parameter('use_dynamic_rig').value)}\n"
            f"  evaluation_config_path={cfg_path}\n"
            f"  camera_info_topic={cam_info_topic} (preferred if available; YAML fallback)\n"
            f"  edge_hit_threshold_px={self.edge_hit_threshold_px}"
        )

    def _resolve_lidar_topic(self):
        explicit = str(self.get_parameter('lidar_topic').value).strip()
        if explicit:
            return explicit
        use_dynamic = bool(self.get_parameter('use_dynamic_rig').value)
        return '/eval/clean/scan' if use_dynamic else '/eval/clean/points'

    def _resolve_eval_config_path(self, user_path: str) -> str:
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

        return os.path.expanduser('~/evaluation.yaml')

    def _load_eval_config(self, path):
        defaults = {
            'camera': {'fx': 500.0, 'fy': 500.0, 'cx': 320.0, 'cy': 240.0},
            'edge': {'canny_low': 100, 'canny_high': 200, 'hit_threshold_px': 2.0},
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
        Dynamic-rig LaserScan conversion must match the same helper used by the
        offline/online 2D calibration path. Using a different scan embedding in
        the evaluator makes the metric path geometrically inconsistent with the
        optimization path.
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
        pts = dyn_utils.scan_dict_to_points_lidar_frame(scan_dict)
        self.latest_points = np.asarray(pts, dtype=np.float32)
        self.try_compute()

    def tf_callback(self, msg):
        self.latest_tf = msg
        self.try_compute()

    def _intrinsics_from_camera_info(self, cam: CameraInfo):
        """
        Convert ROS CameraInfo into the intrinsics dict format expected by geometry_utils.

        geometry_utils in this repo expects:
          {'fx':..., 'fy':..., 'cx':..., 'cy':...}
        """
        K = list(cam.k)
        if len(K) != 9:
            return None
        fx = float(K[0])
        fy = float(K[4])
        cx = float(K[2])
        cy = float(K[5])
        return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    def try_compute(self):
        if self.latest_image is None or self.latest_points is None or self.latest_tf is None:
            return

        if len(self.latest_points) == 0:
            self.pub_ratio.publish(Float32(data=0.0))
            self.pub_px.publish(Float32(data=float('inf')))
            return

        # CameraInfo intrinsics if available; fall back to YAML intrinsics
        intrinsics = self.intrinsics
        if self.latest_camera_info is not None:
            intr_from_ci = self._intrinsics_from_camera_info(self.latest_camera_info)
            if intr_from_ci is not None:
                intrinsics = intr_from_ci

        t = np.array([
            self.latest_tf.transform.translation.x,
            self.latest_tf.transform.translation.y,
            self.latest_tf.transform.translation.z
        ], dtype=np.float32)

        q = np.array([
            self.latest_tf.transform.rotation.x,
            self.latest_tf.transform.rotation.y,
            self.latest_tf.transform.rotation.z,
            self.latest_tf.transform.rotation.w
        ], dtype=np.float32)

        # NOTE:
        #   Extrinsics topics are camera->lidar (frame_id=camera, child_frame_id=lidar).
        #   Projection expects lidar->camera. Invert here (adapter responsibility).
        t_lidar_cam, q_lidar_cam = invert_extrinsics_cam_to_lidar_to_lidar_to_cam(t, q)

        projected_uv, _ = project_lidar_to_image(
            self.latest_points, t_lidar_cam, q_lidar_cam, intrinsics, self.latest_image.shape
        )

        _, dist = compute_edge_distance_transform(self.latest_image, self.canny_low, self.canny_high)

        hit_ratio, _, _ = edge_hit_ratio_with_threshold(
            dist_transform=dist,
            projected_uv=projected_uv,
            hit_threshold_px=self.edge_hit_threshold_px
        )

        mean_px, _ = mean_edge_distance_px(dist, projected_uv)

        self.pub_ratio.publish(Float32(data=float(hit_ratio)))
        self.pub_px.publish(Float32(data=float(mean_px)))


def main(args=None):
    rclpy.init(args=args)
    node = EdgeAlignmentEvaluatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()