###############################################
# DATASET EXTRACTION: Dynamic rig
###############################################
# Purpose:
#   - Save synchronized image + raw LaserScan + odometry for the dynamic 2D rig.
#   - Also save a pseudo-3D point cloud export for future compatibility with the
#     existing 3D pipeline.
#   - Save camera intrinsics from CameraInfo once per run.
#   - Enforce practical quality checks so the recorded run is useful for:
#       1) offline edge-alignment calibration
#       2) online motion-based refinement
#   - Use the same run_### folder structure philosophy as the static extractor.
###############################################

import math
import os
import re
from collections import deque

import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class DynamicRigExtractor(Node):
    def __init__(self):
        super().__init__('dynamic_rig_extractor')
        self.bridge = CvBridge()

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('scan_topic', '/lidar/scan')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('output_root', os.path.expanduser('~/dataset_root/dynamic_rig'))

        # Save both by default. Raw scan is the primary 2D modality.
        self.declare_parameter('save_raw_scan', True)
        self.declare_parameter('save_pseudo_3d', True)
        self.declare_parameter('pseudo_z_m', 0.15)

        # Synchronization thresholds.
        self.declare_parameter('max_wait_sec', 2.0)
        self.declare_parameter('max_age_sec', 0.5)
        self.declare_parameter('sync_slop_sec', 0.10)
        self.declare_parameter('odom_sync_slop_sec', 0.15)

        # If header stamps are not comparable to ROS time, fall back to receive-time.
        self.declare_parameter('stamp_sanity_window_sec', 5.0)

        # Short rolling buffers for nearest-neighbour sync selection.
        self.declare_parameter('image_buffer_size', 120)
        self.declare_parameter('scan_buffer_size', 80)
        self.declare_parameter('odom_buffer_size', 160)

        # Scene quality gates.
        self.declare_parameter('min_valid_scan_points', 20)
        self.declare_parameter('min_edge_pixel_ratio', 0.005)
        self.declare_parameter('min_mean_gradient', 8.0)
        self.declare_parameter('min_save_interval_sec', 0.20)
        self.declare_parameter('min_image_change_mae', 4.0)

        # Motion tagging thresholds.
        self.declare_parameter('motion_translation_tag_m', 0.02)
        self.declare_parameter('motion_yaw_tag_deg', 3.0)

        camera_topic = str(self.get_parameter('camera_topic').value)
        scan_topic = str(self.get_parameter('scan_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.base_output_root = os.path.expanduser(str(self.get_parameter('output_root').value))

        self.save_raw_scan = bool(self.get_parameter('save_raw_scan').value)
        self.save_pseudo_3d = bool(self.get_parameter('save_pseudo_3d').value)
        self.pseudo_z_m = float(self.get_parameter('pseudo_z_m').value)

        self.max_wait_sec = float(self.get_parameter('max_wait_sec').value)
        self.max_age_sec = float(self.get_parameter('max_age_sec').value)
        self.sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)
        self.odom_sync_slop_sec = float(self.get_parameter('odom_sync_slop_sec').value)
        self.stamp_sanity_window_sec = float(self.get_parameter('stamp_sanity_window_sec').value)

        self.image_buffer_size = int(self.get_parameter('image_buffer_size').value)
        self.scan_buffer_size = int(self.get_parameter('scan_buffer_size').value)
        self.odom_buffer_size = int(self.get_parameter('odom_buffer_size').value)

        self.min_valid_scan_points = int(self.get_parameter('min_valid_scan_points').value)
        self.min_edge_pixel_ratio = float(self.get_parameter('min_edge_pixel_ratio').value)
        self.min_mean_gradient = float(self.get_parameter('min_mean_gradient').value)
        self.min_save_interval_sec = float(self.get_parameter('min_save_interval_sec').value)
        self.min_image_change_mae = float(self.get_parameter('min_image_change_mae').value)

        self.motion_translation_tag_m = float(self.get_parameter('motion_translation_tag_m').value)
        self.motion_yaw_tag_deg = float(self.get_parameter('motion_yaw_tag_deg').value)

        # ----------------------------
        # Explicit sensor QoS (BEST_EFFORT)
        # ----------------------------
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ----------------------------
        # Output folders
        # ----------------------------
        os.makedirs(self.base_output_root, exist_ok=True)
        self.run_dir = self._create_next_run_dir(self.base_output_root)

        self.images_dir = os.path.join(self.run_dir, 'images')
        self.lidar_dir = os.path.join(self.run_dir, 'lidar')            # pseudo-3D compatibility export
        self.scans_dir = os.path.join(self.run_dir, 'scans')            # raw 2D scan save
        self.odom_dir = os.path.join(self.run_dir, 'odom')              # odometry per sample
        self.intrinsics_dir = os.path.join(self.run_dir, 'intrinsics')
        self.index_dir = os.path.join(self.run_dir, 'index')
        self.meta_dir = os.path.join(self.run_dir, 'meta')

        for d in [
            self.images_dir,
            self.lidar_dir,
            self.scans_dir,
            self.odom_dir,
            self.intrinsics_dir,
            self.index_dir,
            self.meta_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        self.index_yaml_path = os.path.join(self.index_dir, 'samples.yaml')
        self.run_summary_path = os.path.join(self.run_dir, 'run_summary.yaml')

        # ----------------------------
        # Subscriptions
        # ----------------------------
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, self.sensor_qos)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, self.sensor_qos)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, self.sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, self.sensor_qos)

        # ----------------------------
        # Buffers / state
        # ----------------------------
        self.image_buffer = deque(maxlen=max(2, self.image_buffer_size))
        self.scan_buffer = deque(maxlen=max(2, self.scan_buffer_size))
        self.odom_buffer = deque(maxlen=max(2, self.odom_buffer_size))
        self.latest_camera_info_msg = None

        self.intrinsics_written = False

        self.last_image_receive_time = None
        self.last_scan_receive_time = None
        self.last_odom_receive_time = None
        self.last_caminfo_receive_time = None

        self.rx_image = 0
        self.rx_scan = 0
        self.rx_odom = 0
        self.rx_caminfo = 0

        self.counter = 1
        self.reject_counter = 0

        # Stamp fallback tracking.
        self.warned_image_stamp = False
        self.warned_scan_stamp = False
        self.warned_odom_stamp = False

        # Saved-sample state for duplicate suppression and motion tagging.
        self.last_saved_time_sec = None
        self.last_saved_small_gray = None
        self.last_saved_odom_pose = None
        self.last_saved_odom_time_sec = None

        # Run summary accumulators.
        self.rejection_reason_counts = {}
        self.saved_edge_usable = 0
        self.saved_motion_usable = 0
        self.saved_both_usable = 0
        self.total_saved_scan_points = 0
        self.total_saved_edge_ratio = 0.0
        self.total_saved_gradient = 0.0
        self.sum_sync_img_scan = 0.0
        self.sum_sync_img_odom = 0.0
        self.sum_sync_scan_odom = 0.0
        self.run_path_length_m = 0.0
        self.run_cumulative_yaw_deg = 0.0
        self.last_any_odom_pose_for_run = None
        self.last_any_odom_time_for_run = None

        # Periodic health timer.
        self.health_timer = self.create_timer(1.0, self.health_check)

        self.get_logger().info(
            "DynamicRigExtractor started.\n"
            f"  camera_topic={camera_topic}\n"
            f"  scan_topic={scan_topic}\n"
            f"  camera_info_topic={camera_info_topic}\n"
            f"  odom_topic={odom_topic}\n"
            f"  base_output_root={self.base_output_root}\n"
            f"  run_dir={self.run_dir}\n"
            f"  save_raw_scan={self.save_raw_scan}, save_pseudo_3d={self.save_pseudo_3d}\n"
            f"  pseudo_z_m={self.pseudo_z_m}\n"
            f"  sanity: max_wait_sec={self.max_wait_sec}, max_age_sec={self.max_age_sec}, "
            f"sync_slop_sec={self.sync_slop_sec}, odom_sync_slop_sec={self.odom_sync_slop_sec}\n"
            f"  stamp_sanity_window_sec={self.stamp_sanity_window_sec}\n"
            f"  buffers: image_buffer_size={self.image_buffer_size}, scan_buffer_size={self.scan_buffer_size}, odom_buffer_size={self.odom_buffer_size}\n"
            f"  scene gates: min_valid_scan_points={self.min_valid_scan_points}, "
            f"min_edge_pixel_ratio={self.min_edge_pixel_ratio}, min_mean_gradient={self.min_mean_gradient}, "
            f"min_save_interval_sec={self.min_save_interval_sec}, min_image_change_mae={self.min_image_change_mae}\n"
            f"  motion tags: translation>={self.motion_translation_tag_m} m OR yaw>={self.motion_yaw_tag_deg} deg\n"
            "  NOTE: raw scan is the primary 2D modality; pseudo-3D is saved as compatibility output.\n"
            f"  QoS(subscribers): BEST_EFFORT, VOLATILE, KEEP_LAST(depth=5)"
        )

    # ----------------------------
    # Folder helper
    # ----------------------------
    def _create_next_run_dir(self, base_root: str) -> str:
        pat = re.compile(r'^run_(\d{3})$')
        existing = []
        for name in os.listdir(base_root):
            p = os.path.join(base_root, name)
            if os.path.isdir(p):
                m = pat.match(name)
                if m:
                    existing.append(int(m.group(1)))
        next_id = (max(existing) + 1) if existing else 1
        run_dir = os.path.join(base_root, f"run_{next_id:03d}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    # ----------------------------
    # Time helpers
    # ----------------------------
    def _stamp_to_float_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _now_float_sec(self):
        t = self.get_clock().now().to_msg()
        return float(t.sec) + float(t.nanosec) * 1e-9

    def _effective_msg_time(self, msg_stamp, receive_time_sec, now_sec, warn_flag_name: str) -> float:
        """
        Returns a timestamp that is safe to compare against ROS time.

        Rules:
          - If stamp == 0 -> use receive time
          - If |now - stamp| > stamp_sanity_window_sec -> use receive time
          - Else -> use stamp
        """
        stamp_sec = self._stamp_to_float_sec(msg_stamp)

        if int(msg_stamp.sec) == 0 and int(msg_stamp.nanosec) == 0:
            if not getattr(self, warn_flag_name):
                self.get_logger().warn(
                    f"{warn_flag_name}: header stamp is zero; using receive-time for sync/age checks."
                )
                setattr(self, warn_flag_name, True)
            return float(receive_time_sec)

        if abs(now_sec - stamp_sec) > self.stamp_sanity_window_sec:
            if not getattr(self, warn_flag_name):
                self.get_logger().warn(
                    f"{warn_flag_name}: header stamp not comparable to ROS time "
                    f"(now={now_sec:.3f}, stamp={stamp_sec:.3f}). Using receive-time."
                )
                setattr(self, warn_flag_name, True)
            return float(receive_time_sec)

        return stamp_sec

    def _buffer_entry(self, msg, receive_time_sec: float):
        return {
            'msg': msg,
            'receive_time_sec': float(receive_time_sec),
        }

    def _entry_time(self, entry: dict, now: float, warn_flag_name: str) -> float:
        return self._effective_msg_time(
            entry['msg'].header.stamp,
            entry['receive_time_sec'],
            now,
            warn_flag_name,
        )

    def _find_best_match(self, reference_time_sec: float, buffer_obj: deque, max_dt_sec: float, now: float, warn_flag_name: str):
        best_entry = None
        best_time = None
        best_dt = None

        for entry in buffer_obj:
            entry_time = self._entry_time(entry, now, warn_flag_name)
            dt = abs(reference_time_sec - entry_time)
            if dt > max_dt_sec:
                continue
            if best_dt is None or dt < best_dt:
                best_entry = entry
                best_time = entry_time
                best_dt = dt

        return best_entry, best_time, best_dt

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(value):
            return float(default)
        return float(value)

    # ----------------------------
    # Geometry / motion helpers
    # ----------------------------
    def _yaw_from_quaternion_xyzw(self, x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _wrap_angle_rad(self, angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    def _odom_pose_dict(self, odom_msg: Odometry) -> dict:
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        yaw_rad = self._yaw_from_quaternion_xyzw(q.x, q.y, q.z, q.w)
        return {
            'x': self._safe_float(p.x),
            'y': self._safe_float(p.y),
            'z': self._safe_float(p.z),
            'qx': self._safe_float(q.x),
            'qy': self._safe_float(q.y),
            'qz': self._safe_float(q.z),
            'qw': self._safe_float(q.w, default=1.0),
            'yaw_rad': self._safe_float(yaw_rad),
            'yaw_deg': self._safe_float(math.degrees(yaw_rad)),
        }

    def _odom_twist_dict(self, odom_msg: Odometry) -> dict:
        return {
            'linear': {
                'x': self._safe_float(odom_msg.twist.twist.linear.x),
                'y': self._safe_float(odom_msg.twist.twist.linear.y),
                'z': self._safe_float(odom_msg.twist.twist.linear.z),
            },
            'angular': {
                'x': self._safe_float(odom_msg.twist.twist.angular.x),
                'y': self._safe_float(odom_msg.twist.twist.angular.y),
                'z': self._safe_float(odom_msg.twist.twist.angular.z),
            },
        }

    def _motion_delta(self, prev_pose: dict, curr_pose: dict):
        dx = curr_pose['x'] - prev_pose['x']
        dy = curr_pose['y'] - prev_pose['y']
        dz = curr_pose['z'] - prev_pose['z']
        translation_m = float(math.sqrt(dx * dx + dy * dy + dz * dz))

        yaw_delta_rad = self._wrap_angle_rad(curr_pose['yaw_rad'] - prev_pose['yaw_rad'])
        yaw_delta_deg = float(abs(math.degrees(yaw_delta_rad)))
        return translation_m, yaw_delta_deg

    # ----------------------------
    # Image / scan helpers
    # ----------------------------
    def _compute_image_quality_metrics(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_pixel_ratio = float(np.count_nonzero(edges)) / float(edges.size)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        mean_gradient = float(np.mean(grad_mag))

        small_gray = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)
        return edge_pixel_ratio, mean_gradient, small_gray

    def _image_change_mae(self, small_gray, prev_small_gray):
        if prev_small_gray is None:
            return None
        diff = cv2.absdiff(small_gray, prev_small_gray)
        return float(np.mean(diff))

    def _scan_to_valid_xy_and_ranges(self, scan_msg: LaserScan):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        valid = np.isfinite(ranges)
        valid &= (ranges >= float(scan_msg.range_min))
        valid &= (ranges <= float(scan_msg.range_max))

        if np.count_nonzero(valid) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        angles = scan_msg.angle_min + np.arange(ranges.size, dtype=np.float32) * np.float32(scan_msg.angle_increment)
        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        xs = valid_ranges * np.cos(valid_angles)
        ys = valid_ranges * np.sin(valid_angles)
        xy = np.stack([xs, ys], axis=1).astype(np.float32)
        return xy, valid_ranges.astype(np.float32), valid_angles.astype(np.float32)

    def _write_pcd_ascii(self, pcd_path: str, pts_xyz: np.ndarray):
        with open(pcd_path, 'w') as f:
            f.write("# .PCD v0.7 - Point Cloud Data\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {pts_xyz.shape[0]}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {pts_xyz.shape[0]}\n")
            f.write("DATA ascii\n")
            for x, y, z in pts_xyz:
                f.write(f"{x} {y} {z}\n")

    # ----------------------------
    # Summary helpers
    # ----------------------------
    def _count_rejection(self, reason_key: str):
        self.rejection_reason_counts[reason_key] = self.rejection_reason_counts.get(reason_key, 0) + 1

    def _write_index_entry(self, idx: str, meta: dict):
        with open(self.index_yaml_path, 'a') as f:
            yaml.safe_dump([{idx: meta}], f, sort_keys=False)

    def _build_run_summary_dict(self):
        saved_count = self.counter - 1
        mean_scan_points = float(self.total_saved_scan_points) / float(saved_count) if saved_count > 0 else 0.0
        mean_edge_ratio = float(self.total_saved_edge_ratio) / float(saved_count) if saved_count > 0 else 0.0
        mean_gradient = float(self.total_saved_gradient) / float(saved_count) if saved_count > 0 else 0.0
        mean_img_scan_dt = float(self.sum_sync_img_scan) / float(saved_count) if saved_count > 0 else 0.0
        mean_img_odom_dt = float(self.sum_sync_img_odom) / float(saved_count) if saved_count > 0 else 0.0
        mean_scan_odom_dt = float(self.sum_sync_scan_odom) / float(saved_count) if saved_count > 0 else 0.0

        return {
            'saved_samples': int(saved_count),
            'reject_counter_total': int(self.reject_counter),
            'rejection_reason_counts': dict(sorted(self.rejection_reason_counts.items())),
            'rx_counts': {
                'image': int(self.rx_image),
                'scan': int(self.rx_scan),
                'odom': int(self.rx_odom),
                'caminfo': int(self.rx_caminfo),
            },
            'saved_tags': {
                'edge_usable': int(self.saved_edge_usable),
                'motion_usable': int(self.saved_motion_usable),
                'both_usable': int(self.saved_both_usable),
            },
            'save_options': {
                'save_raw_scan': bool(self.save_raw_scan),
                'save_pseudo_3d': bool(self.save_pseudo_3d),
                'pseudo_z_m': float(self.pseudo_z_m),
            },
            'thresholds': {
                'max_wait_sec': float(self.max_wait_sec),
                'max_age_sec': float(self.max_age_sec),
                'sync_slop_sec': float(self.sync_slop_sec),
                'odom_sync_slop_sec': float(self.odom_sync_slop_sec),
                'stamp_sanity_window_sec': float(self.stamp_sanity_window_sec),
                'image_buffer_size': int(self.image_buffer_size),
                'scan_buffer_size': int(self.scan_buffer_size),
                'odom_buffer_size': int(self.odom_buffer_size),
                'min_valid_scan_points': int(self.min_valid_scan_points),
                'min_edge_pixel_ratio': float(self.min_edge_pixel_ratio),
                'min_mean_gradient': float(self.min_mean_gradient),
                'min_save_interval_sec': float(self.min_save_interval_sec),
                'min_image_change_mae': float(self.min_image_change_mae),
                'motion_translation_tag_m': float(self.motion_translation_tag_m),
                'motion_yaw_tag_deg': float(self.motion_yaw_tag_deg),
            },
            'means': {
                'valid_scan_points': mean_scan_points,
                'edge_pixel_ratio': mean_edge_ratio,
                'mean_gradient': mean_gradient,
                'sync_dt_img_scan_sec': mean_img_scan_dt,
                'sync_dt_img_odom_sec': mean_img_odom_dt,
                'sync_dt_scan_odom_sec': mean_scan_odom_dt,
            },
            'motion_summary': {
                'path_length_m': float(self.run_path_length_m),
                'cumulative_yaw_change_deg': float(self.run_cumulative_yaw_deg),
            },
        }

    def _write_run_summary(self):
        summary = self._build_run_summary_dict()
        with open(self.run_summary_path, 'w') as f:
            yaml.safe_dump(summary, f, sort_keys=False)

    # ----------------------------
    # Health timer
    # ----------------------------
    def health_check(self):
        now = self._now_float_sec()

        if self.last_image_receive_time is None:
            self.get_logger().warn("No camera images received yet. Check camera_topic.")
        else:
            dt = now - self.last_image_receive_time
            if dt > self.max_wait_sec:
                self.get_logger().warn(f"No camera images in last {dt:.2f}s (max_wait_sec={self.max_wait_sec}).")

        if self.last_scan_receive_time is None:
            self.get_logger().warn("No LaserScan received yet. Check scan_topic + QoS.")
        else:
            dt = now - self.last_scan_receive_time
            if dt > self.max_wait_sec:
                self.get_logger().warn(f"No LaserScan in last {dt:.2f}s (max_wait_sec={self.max_wait_sec}).")

        if self.last_odom_receive_time is None:
            self.get_logger().warn("No Odometry received yet. Check odom_topic + QoS.")
        else:
            dt = now - self.last_odom_receive_time
            if dt > self.max_wait_sec:
                self.get_logger().warn(f"No Odometry in last {dt:.2f}s (max_wait_sec={self.max_wait_sec}).")

        if self.last_caminfo_receive_time is None:
            self.get_logger().warn("No CameraInfo received yet. Check camera_info_topic.")
        else:
            dt = now - self.last_caminfo_receive_time
            if dt > self.max_wait_sec:
                self.get_logger().warn(f"No CameraInfo in last {dt:.2f}s (max_wait_sec={self.max_wait_sec}).")

        self._write_run_summary()
        self.get_logger().info(
            f"RX counts: image={self.rx_image}, scan={self.rx_scan}, odom={self.rx_odom}, caminfo={self.rx_caminfo} | "
            f"buffered(image/scan/odom)={len(self.image_buffer)}/{len(self.scan_buffer)}/{len(self.odom_buffer)} | "
            f"saved={self.counter - 1}"
        )

    # ----------------------------
    # Callbacks
    # ----------------------------
    def camera_info_callback(self, msg: CameraInfo):
        self.rx_caminfo += 1
        self.latest_camera_info_msg = msg
        self.last_caminfo_receive_time = self._now_float_sec()
        if not self.intrinsics_written:
            self._write_intrinsics_if_valid(msg)

    def image_callback(self, msg: Image):
        self.rx_image += 1
        receive_time_sec = self._now_float_sec()
        self.last_image_receive_time = receive_time_sec
        self.image_buffer.append(self._buffer_entry(msg, receive_time_sec))
        self.try_save()

    def scan_callback(self, msg: LaserScan):
        self.rx_scan += 1
        receive_time_sec = self._now_float_sec()
        self.last_scan_receive_time = receive_time_sec
        self.scan_buffer.append(self._buffer_entry(msg, receive_time_sec))

    def odom_callback(self, msg: Odometry):
        self.rx_odom += 1
        receive_time_sec = self._now_float_sec()
        self.last_odom_receive_time = receive_time_sec
        self.odom_buffer.append(self._buffer_entry(msg, receive_time_sec))

        current_pose = self._odom_pose_dict(msg)
        if self.last_any_odom_pose_for_run is not None:
            delta_translation_m, delta_yaw_deg = self._motion_delta(self.last_any_odom_pose_for_run, current_pose)
            self.run_path_length_m += delta_translation_m
            self.run_cumulative_yaw_deg += delta_yaw_deg
        self.last_any_odom_pose_for_run = current_pose
        self.last_any_odom_time_for_run = self.last_odom_receive_time

    # ----------------------------
    # Intrinsics save (once)
    # ----------------------------
    def _write_intrinsics_if_valid(self, cam_info: CameraInfo):
        if cam_info.width <= 0 or cam_info.height <= 0:
            self.get_logger().warn("CameraInfo invalid (width/height <= 0). Not writing intrinsics yet.")
            return

        if all(v == 0.0 for v in cam_info.k):
            self.get_logger().warn("CameraInfo K matrix is all zeros. Not writing intrinsics yet.")
            return

        intr_path = os.path.join(self.intrinsics_dir, 'camera_intrinsics.yaml')
        intr = {
            'width': int(cam_info.width),
            'height': int(cam_info.height),
            'distortion_model': str(cam_info.distortion_model),
            'd': [float(x) for x in cam_info.d],
            'k': [float(x) for x in cam_info.k],
            'r': [float(x) for x in cam_info.r],
            'p': [float(x) for x in cam_info.p],
            'frame_id': str(cam_info.header.frame_id),
            'note': 'Saved from CameraInfo topic for this run.',
        }

        with open(intr_path, 'w') as f:
            yaml.safe_dump(intr, f)

        self.intrinsics_written = True
        self.get_logger().info(f"Wrote camera intrinsics to: {intr_path}")

    # ----------------------------
    # Save synchronized sample
    # ----------------------------
    def try_save(self):
        # Hard gate: all streams required.
        if len(self.image_buffer) == 0 or len(self.scan_buffer) == 0 or len(self.odom_buffer) == 0:
            return

        # Hard gate: intrinsics required.
        if not self.intrinsics_written:
            return

        while len(self.image_buffer) > 0:
            now = self._now_float_sec()
            image_entry = self.image_buffer[0]
            img_t = self._entry_time(image_entry, now, 'warned_image_stamp')

            # Hard gate: image age. If the oldest buffered image is already stale, drop it.
            if (now - img_t) > self.max_age_sec:
                self._periodic_reject_log('Rejected: image too old (age gate).', img_t, img_t, img_t, now)
                self._count_rejection('age_image')
                self.image_buffer.popleft()
                continue

            scan_entry, scan_t, dt_img_scan = self._find_best_match(
                reference_time_sec=img_t,
                buffer_obj=self.scan_buffer,
                max_dt_sec=self.sync_slop_sec,
                now=now,
                warn_flag_name='warned_scan_stamp',
            )
            if scan_entry is None:
                latest_scan_t = self._entry_time(self.scan_buffer[-1], now, 'warned_scan_stamp')
                if latest_scan_t >= img_t:
                    self._periodic_reject_log('Rejected: |image-scan| too large (sync gate).', img_t, latest_scan_t, img_t, now)
                    self._count_rejection('sync_image_scan')
                    self.image_buffer.popleft()
                    continue
                return

            odom_entry, odom_t, dt_img_odom = self._find_best_match(
                reference_time_sec=img_t,
                buffer_obj=self.odom_buffer,
                max_dt_sec=self.odom_sync_slop_sec,
                now=now,
                warn_flag_name='warned_odom_stamp',
            )
            if odom_entry is None:
                latest_odom_t = self._entry_time(self.odom_buffer[-1], now, 'warned_odom_stamp')
                if latest_odom_t >= img_t:
                    self._periodic_reject_log('Rejected: |image-odom| too large (sync gate).', img_t, img_t, latest_odom_t, now)
                    self._count_rejection('sync_image_odom')
                    self.image_buffer.popleft()
                    continue
                return

            dt_scan_odom = abs(scan_t - odom_t)

            # Hard gate: age for matched scan/odom.
            if (now - scan_t) > self.max_age_sec:
                self._periodic_reject_log('Rejected: scan too old (age gate).', img_t, scan_t, odom_t, now)
                self._count_rejection('age_scan')
                self.scan_buffer.popleft()
                continue
            if (now - odom_t) > self.max_age_sec:
                self._periodic_reject_log('Rejected: odom too old (age gate).', img_t, scan_t, odom_t, now)
                self._count_rejection('age_odom')
                self.odom_buffer.popleft()
                continue

            # Hard gate: synchronization.
            if dt_img_scan > self.sync_slop_sec:
                self._periodic_reject_log('Rejected: |image-scan| too large (sync gate).', img_t, scan_t, odom_t, now)
                self._count_rejection('sync_image_scan')
                self.image_buffer.popleft()
                continue
            if dt_img_odom > self.odom_sync_slop_sec:
                self._periodic_reject_log('Rejected: |image-odom| too large (sync gate).', img_t, scan_t, odom_t, now)
                self._count_rejection('sync_image_odom')
                self.image_buffer.popleft()
                continue
            if dt_scan_odom > self.odom_sync_slop_sec:
                self._periodic_reject_log('Rejected: |scan-odom| too large (sync gate).', img_t, scan_t, odom_t, now)
                self._count_rejection('sync_scan_odom')
                self.image_buffer.popleft()
                continue

            # Scene gate: minimum spacing between accepted samples.
            ref_time_sec = max(img_t, scan_t, odom_t)
            if self.last_saved_time_sec is not None:
                if (ref_time_sec - self.last_saved_time_sec) < self.min_save_interval_sec:
                    self._periodic_reject_log('Rejected: accepted sample spacing too small.', img_t, scan_t, odom_t, now)
                    self._count_rejection('save_interval')
                    self.image_buffer.popleft()
                    continue

            # Convert image and compute image quality metrics.
            try:
                img = self.bridge.imgmsg_to_cv2(image_entry['msg'], 'bgr8')
            except Exception as exc:
                self.get_logger().warn(f'Image conversion failed: {exc}')
                self._count_rejection('image_conversion_failed')
                self.image_buffer.popleft()
                continue

            edge_pixel_ratio, mean_gradient, small_gray = self._compute_image_quality_metrics(img)
            if edge_pixel_ratio < self.min_edge_pixel_ratio:
                self._periodic_reject_log('Rejected: image edge ratio below threshold.', img_t, scan_t, odom_t, now)
                self._count_rejection('edge_ratio')
                self.image_buffer.popleft()
                continue
            if mean_gradient < self.min_mean_gradient:
                self._periodic_reject_log('Rejected: image mean gradient below threshold.', img_t, scan_t, odom_t, now)
                self._count_rejection('mean_gradient')
                self.image_buffer.popleft()
                continue

            image_change_mae = self._image_change_mae(small_gray, self.last_saved_small_gray)
            if image_change_mae is not None and image_change_mae < self.min_image_change_mae:
                self._periodic_reject_log('Rejected: image too similar to previous accepted sample.', img_t, scan_t, odom_t, now)
                self._count_rejection('image_similarity')
                self.image_buffer.popleft()
                continue

            # Scan gate + raw representation.
            xy_points, valid_ranges, valid_angles = self._scan_to_valid_xy_and_ranges(scan_entry['msg'])
            valid_scan_points = int(xy_points.shape[0])
            if valid_scan_points < self.min_valid_scan_points:
                self._periodic_reject_log('Rejected: too few valid scan points.', img_t, scan_t, odom_t, now)
                self._count_rejection('valid_scan_points')
                self.image_buffer.popleft()
                continue

            # Motion tags are computed relative to the last accepted sample.
            current_odom_pose = self._odom_pose_dict(odom_entry['msg'])
            current_odom_twist = self._odom_twist_dict(odom_entry['msg'])
            if self.last_saved_odom_pose is None:
                motion_translation_m = 0.0
                motion_yaw_deg = 0.0
            else:
                motion_translation_m, motion_yaw_deg = self._motion_delta(self.last_saved_odom_pose, current_odom_pose)

            edge_usable = True
            motion_usable = (
                motion_translation_m >= self.motion_translation_tag_m or
                motion_yaw_deg >= self.motion_yaw_tag_deg
            )
            both_usable = edge_usable and motion_usable

            idx = f"{self.counter:06d}"

            # Save image.
            cv2.imwrite(os.path.join(self.images_dir, f'{idx}.png'), img)

            # Save raw scan.
            scan_path = None
            if self.save_raw_scan:
                scan_path = os.path.join(self.scans_dir, f'{idx}.npz')
                np.savez_compressed(
                    scan_path,
                    ranges=np.asarray(scan_entry['msg'].ranges, dtype=np.float32),
                    intensities=np.asarray(scan_entry['msg'].intensities, dtype=np.float32),
                    angle_min=np.float32(scan_entry['msg'].angle_min),
                    angle_max=np.float32(scan_entry['msg'].angle_max),
                    angle_increment=np.float32(scan_entry['msg'].angle_increment),
                    time_increment=np.float32(scan_entry['msg'].time_increment),
                    scan_time=np.float32(scan_entry['msg'].scan_time),
                    range_min=np.float32(scan_entry['msg'].range_min),
                    range_max=np.float32(scan_entry['msg'].range_max),
                    valid_xy=xy_points,
                    valid_ranges=valid_ranges,
                    valid_angles=valid_angles,
                )

            # Save pseudo-3D compatibility export.
            pcd_path = None
            if self.save_pseudo_3d:
                zs = np.full((valid_scan_points,), self.pseudo_z_m, dtype=np.float32)
                pts_xyz = np.concatenate([xy_points, zs.reshape(-1, 1)], axis=1).astype(np.float32)
                pcd_path = os.path.join(self.lidar_dir, f'{idx}.pcd')
                self._write_pcd_ascii(pcd_path, pts_xyz)

            # Save odometry.
            odom_path = os.path.join(self.odom_dir, f'{idx}.yaml')
            odom_yaml = {
                'stamp_sec': float(odom_t),
                'header_frame_id': str(odom_entry['msg'].header.frame_id),
                'child_frame_id': str(odom_entry['msg'].child_frame_id),
                'pose': current_odom_pose,
                'twist': current_odom_twist,
            }
            with open(odom_path, 'w') as f:
                yaml.safe_dump(odom_yaml, f, sort_keys=False)

            # Save per-sample metadata.
            meta = {
                'sample_id': idx,
                'image_stamp_sec': float(img_t),
                'scan_stamp_sec': float(scan_t),
                'odom_stamp_sec': float(odom_t),
                'sync_dt_sec': {
                    'image_scan': float(dt_img_scan),
                    'image_odom': float(dt_img_odom),
                    'scan_odom': float(dt_scan_odom),
                },
                'paths': {
                    'image': os.path.relpath(os.path.join(self.images_dir, f'{idx}.png'), self.run_dir),
                    'scan_npz': os.path.relpath(scan_path, self.run_dir) if scan_path is not None else None,
                    'pseudo_3d_pcd': os.path.relpath(pcd_path, self.run_dir) if pcd_path is not None else None,
                    'odom_yaml': os.path.relpath(odom_path, self.run_dir),
                },
                'quality': {
                    'valid_scan_points': int(valid_scan_points),
                    'edge_pixel_ratio': float(edge_pixel_ratio),
                    'mean_gradient': float(mean_gradient),
                    'image_change_mae_vs_prev_saved': None if image_change_mae is None else float(image_change_mae),
                },
                'motion': {
                    'translation_from_prev_saved_m': float(motion_translation_m),
                    'yaw_from_prev_saved_deg': float(motion_yaw_deg),
                },
                'tags': {
                    'edge_usable': bool(edge_usable),
                    'motion_usable': bool(motion_usable),
                    'both_usable': bool(both_usable),
                },
                'notes': [
                    'raw LaserScan is the primary 2D modality',
                    'pseudo-3D is a compatibility export for future 3D-pipeline reuse',
                    'nearest-neighbour buffered sync used for scan/odom selection',
                    'non-finite odom twist values are sanitized to 0.0 before saving',
                ],
            }
            meta_path = os.path.join(self.meta_dir, f'{idx}.yaml')
            with open(meta_path, 'w') as f:
                yaml.safe_dump(meta, f, sort_keys=False)

            self._write_index_entry(idx, meta)

            # Update run counters.
            self.total_saved_scan_points += valid_scan_points
            self.total_saved_edge_ratio += edge_pixel_ratio
            self.total_saved_gradient += mean_gradient
            self.sum_sync_img_scan += dt_img_scan
            self.sum_sync_img_odom += dt_img_odom
            self.sum_sync_scan_odom += dt_scan_odom
            self.saved_edge_usable += int(edge_usable)
            self.saved_motion_usable += int(motion_usable)
            self.saved_both_usable += int(both_usable)

            self.last_saved_time_sec = ref_time_sec
            self.last_saved_small_gray = small_gray
            self.last_saved_odom_pose = current_odom_pose
            self.last_saved_odom_time_sec = odom_t

            self.get_logger().info(
                f"Saved DYNAMIC sample {idx} "
                f"(valid_scan_points={valid_scan_points}, "
                f"dt_img_scan={dt_img_scan:.3f}s, dt_img_odom={dt_img_odom:.3f}s, "
                f"edge_usable={edge_usable}, motion_usable={motion_usable})"
            )

            self.counter += 1
            self.image_buffer.popleft()
            self._write_run_summary()

    def _periodic_reject_log(self, reason: str, img_t: float, scan_t: float, odom_t: float, now: float):
        """
        Avoid log spam: print every 50th rejection.
        """
        self.reject_counter += 1
        if self.reject_counter % 50 != 0:
            return

        img_age = now - img_t
        scan_age = now - scan_t
        odom_age = now - odom_t
        dt_img_scan = abs(img_t - scan_t)
        dt_img_odom = abs(img_t - odom_t)
        dt_scan_odom = abs(scan_t - odom_t)

        self.get_logger().warn(
            f"{reason} (every-50)\n"
            f"  img_age={img_age:.3f}s, scan_age={scan_age:.3f}s, odom_age={odom_age:.3f}s\n"
            f"  |img-scan|={dt_img_scan:.3f}s, |img-odom|={dt_img_odom:.3f}s, |scan-odom|={dt_scan_odom:.3f}s\n"
            f"  gates: max_age_sec={self.max_age_sec}, sync_slop_sec={self.sync_slop_sec}, "
            f"odom_sync_slop_sec={self.odom_sync_slop_sec}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DynamicRigExtractor()
    try:
        rclpy.spin(node)
    finally:
        node._write_run_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
