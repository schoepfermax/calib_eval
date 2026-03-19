###############################################
# DATASET EXTRACTION: Static rig
###############################################
# Purpose:
#   - Record synchronized pairs:
#       camera image (sensor_msgs/Image)
#       3D LiDAR cloud (sensor_msgs/PointCloud2)
#   - Record camera intrinsics via CameraInfo and store them once per run
#   - Save into a unified dataset structure.
###############################################

import os
import re
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge


class StaticRigExtractor(Node):
    def __init__(self):
        super().__init__('static_rig_extractor')
        self.bridge = CvBridge()

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter('camera_topic', '/basler/camera/image_raw')
        self.declare_parameter('points_topic', '/ouster/points')
        self.declare_parameter('camera_info_topic', '/basler/camera/camera_info_calibrated')
        self.declare_parameter('output_root', os.path.expanduser('~/dataset_root/static_mount_h1'))

        # Sync + sanity
        self.declare_parameter('max_wait_sec', 2.0)
        self.declare_parameter('max_age_sec', 1.0)
        self.declare_parameter('sync_slop_sec', 0.2)

        # If header stamps are not comparable to ROS time, we fall back to receive-time.
        # This threshold defines what "too far from now" means.
        self.declare_parameter('stamp_sanity_window_sec', 5.0)

        camera_topic = str(self.get_parameter('camera_topic').value)
        points_topic = str(self.get_parameter('points_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)

        self.base_output_root = os.path.expanduser(str(self.get_parameter('output_root').value))
        self.max_wait_sec = float(self.get_parameter('max_wait_sec').value)
        self.max_age_sec = float(self.get_parameter('max_age_sec').value)
        self.sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)
        self.stamp_sanity_window_sec = float(self.get_parameter('stamp_sanity_window_sec').value)

        # ----------------------------
        # Explicit sensor QoS: BEST_EFFORT
        # ----------------------------
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ----------------------------
        # Output folder (run_###)
        # ----------------------------
        os.makedirs(self.base_output_root, exist_ok=True)
        self.run_dir = self._create_next_run_dir(self.base_output_root)

        self.images_dir = os.path.join(self.run_dir, 'images')
        self.lidar_dir = os.path.join(self.run_dir, 'lidar')
        self.intrinsics_dir = os.path.join(self.run_dir, 'intrinsics')
        self.index_dir = os.path.join(self.run_dir, 'index')
        self.meta_dir = os.path.join(self.run_dir, 'meta')

        for d in [self.images_dir, self.lidar_dir, self.intrinsics_dir, self.index_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)

        # ----------------------------
        # Subscriptions
        # ----------------------------
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, self.sensor_qos)
        self.pcl_sub = self.create_subscription(PointCloud2, points_topic, self.pcl_callback, self.sensor_qos)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, self.sensor_qos)

        # ----------------------------
        # Buffers / state
        # ----------------------------
        self.latest_image_msg = None
        self.latest_pcl_msg = None

        self.latest_camera_info_msg = None
        self.intrinsics_written = False

        self.last_image_receive_time = None
        self.last_pcl_receive_time = None
        self.last_caminfo_receive_time = None

        self.rx_image = 0
        self.rx_pcl = 0
        self.rx_caminfo = 0

        self.counter = 1
        self.reject_counter = 0

        # Stamp fallback tracking (to avoid spamming logs)
        self.warned_image_stamp = False
        self.warned_pcl_stamp = False

        self.health_timer = self.create_timer(1.0, self.health_check)

        self.get_logger().info(
            "StaticRigExtractor started.\n"
            f"  camera_topic={camera_topic}\n"
            f"  points_topic={points_topic}\n"
            f"  camera_info_topic={camera_info_topic}\n"
            f"  base_output_root={self.base_output_root}\n"
            f"  run_dir={self.run_dir}\n"
            f"  sanity: max_wait_sec={self.max_wait_sec}, max_age_sec={self.max_age_sec}, sync_slop_sec={self.sync_slop_sec}\n"
            f"  stamp_sanity_window_sec={self.stamp_sanity_window_sec}\n"
            "  NOTE: baseline/reference extrinsics are handled elsewhere (not written here).\n"
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

        Why this exists:
          Some sensors/drivers publish header stamps in a different clock domain
          (sensor time / PTP / other epoch). Those stamps are not comparable to ROS time.
          If the stamp is "too far" from now, we fall back to receive time.

        Rules:
          - If stamp == 0 -> use receive time
          - If |now - stamp| > stamp_sanity_window_sec -> use receive time
          - Else -> use stamp
        """
        stamp_sec = self._stamp_to_float_sec(msg_stamp)

        # Stamp is exactly zero -> invalid.
        if int(msg_stamp.sec) == 0 and int(msg_stamp.nanosec) == 0:
            if not getattr(self, warn_flag_name):
                self.get_logger().warn(
                    f"{warn_flag_name}: header stamp is zero; using receive-time for sync/age checks."
                )
                setattr(self, warn_flag_name, True)
            return float(receive_time_sec)

        # Stamp is wildly far from "now" -> likely a different clock epoch.
        if abs(now_sec - stamp_sec) > self.stamp_sanity_window_sec:
            if not getattr(self, warn_flag_name):
                self.get_logger().warn(
                    f"{warn_flag_name}: header stamp not comparable to ROS time "
                    f"(now={now_sec:.3f}, stamp={stamp_sec:.3f}). Using receive-time."
                )
                setattr(self, warn_flag_name, True)
            return float(receive_time_sec)

        return stamp_sec

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

        if self.last_pcl_receive_time is None:
            self.get_logger().warn("No point clouds received yet. Check points_topic + QoS.")
        else:
            dt = now - self.last_pcl_receive_time
            if dt > self.max_wait_sec:
                self.get_logger().warn(f"No point clouds in last {dt:.2f}s (max_wait_sec={self.max_wait_sec}).")

        if self.last_caminfo_receive_time is None:
            self.get_logger().warn("No CameraInfo received yet. Check camera_info_topic.")
        else:
            dt = now - self.last_caminfo_receive_time
            if dt > self.max_wait_sec:
                self.get_logger().warn(f"No CameraInfo in last {dt:.2f}s (max_wait_sec={self.max_wait_sec}).")

        self.get_logger().info(
            f"RX counts: image={self.rx_image}, points={self.rx_pcl}, caminfo={self.rx_caminfo} | saved={self.counter-1}"
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
        self.latest_image_msg = msg
        self.last_image_receive_time = self._now_float_sec()
        self.try_save()

    def pcl_callback(self, msg: PointCloud2):
        self.rx_pcl += 1
        self.latest_pcl_msg = msg
        self.last_pcl_receive_time = self._now_float_sec()
        self.try_save()

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
            'note': 'Saved from CameraInfo topic for this run.'
        }
        with open(intr_path, 'w') as f:
            yaml.safe_dump(intr, f)

        self.intrinsics_written = True
        self.get_logger().info(f"Wrote camera intrinsics to: {intr_path}")

    # ----------------------------
    # Save synchronized sample
    # ----------------------------
    def try_save(self):
        if self.latest_image_msg is None or self.latest_pcl_msg is None:
            return
        if not self.intrinsics_written:
            return

        now = self._now_float_sec()

        # IMPORTANT FIX:
        # Use "effective" timestamps that are guaranteed comparable to ROS time.
        img_t = self._effective_msg_time(
            self.latest_image_msg.header.stamp,
            self.last_image_receive_time,
            now,
            warn_flag_name='warned_image_stamp'
        )
        pcl_t = self._effective_msg_time(
            self.latest_pcl_msg.header.stamp,
            self.last_pcl_receive_time,
            now,
            warn_flag_name='warned_pcl_stamp'
        )

        # Age gate
        if (now - img_t) > self.max_age_sec or (now - pcl_t) > self.max_age_sec:
            self._periodic_reject_log("Rejected: message too old (age gate).", img_t, pcl_t, now)
            return

        # Sync gate
        dt = abs(img_t - pcl_t)
        if dt > self.sync_slop_sec:
            self._periodic_reject_log("Rejected: |Δt| too large (sync gate).", img_t, pcl_t, now, dt=dt)
            return

        idx = f"{self.counter:06d}"

        # Save image
        img = self.bridge.imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
        cv2.imwrite(os.path.join(self.images_dir, f"{idx}.png"), img)

        # Save point cloud
        pts = []
        for p in point_cloud2.read_points(self.latest_pcl_msg, skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        pts = np.asarray(pts, dtype=np.float32)

        if pts.shape[0] == 0:
            self._periodic_reject_log("Rejected: empty point cloud after conversion.", img_t, pcl_t, now, dt=dt)
            return

        pcd_path = os.path.join(self.lidar_dir, f"{idx}.pcd")
        with open(pcd_path, 'w') as f:
            f.write("# .PCD v0.7 - Point Cloud Data\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {pts.shape[0]}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {pts.shape[0]}\n")
            f.write("DATA ascii\n")
            for x, y, z in pts:
                f.write(f"{x} {y} {z}\n")

        # Save metadata
        meta_path = os.path.join(self.meta_dir, f"{idx}.yaml")
        meta = {'image_time_sec': float(img_t), 'points_time_sec': float(pcl_t), 'sync_dt_sec': float(dt)}
        with open(meta_path, 'w') as f:
            yaml.safe_dump(meta, f)

        self.get_logger().info(f"Saved STATIC sample {idx} (points={pts.shape[0]}, |Δt|={dt:.3f}s)")

        self.counter += 1
        self.latest_image_msg = None
        self.latest_pcl_msg = None

    def _periodic_reject_log(self, reason: str, img_t: float, pcl_t: float, now: float, dt: float = None):
        self.reject_counter += 1
        if self.reject_counter % 50 != 0:
            return

        if dt is None:
            dt = abs(img_t - pcl_t)

        img_age = now - img_t
        pcl_age = now - pcl_t
        self.get_logger().warn(
            f"{reason} (every-50)\n"
            f"  img_age={img_age:.3f}s, pcl_age={pcl_age:.3f}s, |Δt|={dt:.3f}s\n"
            f"  gates: max_age_sec={self.max_age_sec}, sync_slop_sec={self.sync_slop_sec}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = StaticRigExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
