###############################################
#!/usr/bin/env python3
###############################################
# Data Preprocessor Node
#
# This node is the "starting point" of the evaluation pipeline after dataset
# extraction + splitting are already done.
#
# OFFLINE MODE (default):
#   - Reads samples from unified dataset loader (train/val/test already exist).
#   - Applies deterministic, lightweight preprocessing to:
#       - image (optional mild denoise)
#       - point cloud (optional range/box crop + optional fixed-count sampling)
#   - Publishes normalized topics used downstream by model wrappers + evaluators:
#       /eval/clean/image   (sensor_msgs/Image)
#       /eval/clean/points  (sensor_msgs/PointCloud2)  [static rig]
#       /eval/clean/scan    (sensor_msgs/PointCloud2)  [dynamic rig]
#       /eval/camera_info   (sensor_msgs/CameraInfo)   [per-run intrinsics from dataset]
###############################################

import os
import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo, LaserScan
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry

from calib_eval.universal_dataset_loader import UniversalCalibrationDataset


class DataPreprocessorNode(Node):
    def __init__(self):
        super().__init__("data_preprocessor_node")
        self.bridge = CvBridge()

        ###############################################
        # MODE (offline by default)
        ###############################################
        # Kept as a parameter so online can be re-enabled if needed.
        self.declare_parameter("mode", "offline")

        self.mode = str(self.get_parameter("mode").value).strip().lower()
        if self.mode not in ["offline", "online"]:
            self.get_logger().warn(f"Unknown mode='{self.mode}', forcing offline.")
            self.mode = "offline"

        ###############################################
        # OFFLINE DATASET SETTINGS
        ###############################################
        self.declare_parameter("dataset_root", "")
        self.declare_parameter("split", "train")
        self.declare_parameter("start_index", 0)
        self.declare_parameter("max_samples", -1)         # -1 = all
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("use_dynamic_rig", False)  # false->points, true->scan
        self.declare_parameter("camera_frame_id", "camera")

        if self.mode == "online":
            # ------------------------------------------------------------
            # ONLINE MODE IS NOT USED IN CURRENT WORKFLOW.
            # Keeping it intentionally disabled to avoid confusion. Backed up.
            # ------------------------------------------------------------
            self.get_logger().error(
                "DataPreprocessorNode online mode is currently disabled/commented out.\n"
                "Set mode='offline' (default)."
            )
            raise RuntimeError("Online mode disabled in this thesis pipeline snapshot.")

        self.dataset_root = str(self.get_parameter("dataset_root").value).strip()
        self.split = str(self.get_parameter("split").value).strip()
        self.start_index = int(self.get_parameter("start_index").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.use_dynamic_rig = bool(self.get_parameter("use_dynamic_rig").value)
        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value).strip()

        if not self.dataset_root:
            self.get_logger().error("dataset_root is empty. Provide dataset_root param.")
            raise RuntimeError("dataset_root must be provided.")

        ###############################################
        # Output topics
        ###############################################
        self.declare_parameter("out_image_topic", "/eval/clean/image")
        self.declare_parameter("out_points_topic", "/eval/clean/points")
        self.declare_parameter("out_scan_topic", "/eval/clean/scan")
        self.declare_parameter("out_camera_info_topic", "/eval/camera_info")
        self.declare_parameter("out_odom_topic", "/eval/odom")

        # Dynamic-rig representation choice:
        #   - "raw_scan"     -> publish LaserScan on out_scan_topic, do not fabricate pseudo cloud
        #   - "pseudo_points"-> publish PointCloud2 from stored dataset points on out_points_topic
        # For static rig, this parameter is ignored and points are published as usual.
        self.declare_parameter("dynamic_representation", "raw_scan")

        self.out_image_topic = str(self.get_parameter("out_image_topic").value).strip()
        self.out_points_topic = str(self.get_parameter("out_points_topic").value).strip()
        self.out_scan_topic = str(self.get_parameter("out_scan_topic").value).strip()
        self.out_camera_info_topic = str(self.get_parameter("out_camera_info_topic").value).strip()
        self.out_odom_topic = str(self.get_parameter("out_odom_topic").value).strip()

        self.dynamic_representation = str(
            self.get_parameter("dynamic_representation").value
        ).strip().lower()

        if self.dynamic_representation not in ["raw_scan", "pseudo_points"]:
            self.get_logger().warn(
                f"Unknown dynamic_representation='{self.dynamic_representation}', forcing 'raw_scan'."
            )
            self.dynamic_representation = "raw_scan"

        self.image_pub = self.create_publisher(Image, self.out_image_topic, 10)
        self.points_pub = self.create_publisher(PointCloud2, self.out_points_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.out_scan_topic, 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, self.out_camera_info_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.out_odom_topic, 10)

        ###############################################
        # Preprocessing toggles
        ###############################################
        self.declare_parameter("enable_image_denoise", False)
        self.declare_parameter("denoise_kernel", 3)

        self.declare_parameter("enable_box_crop", False)
        self.declare_parameter("crop_x_min", -50.0)
        self.declare_parameter("crop_x_max", 50.0)
        self.declare_parameter("crop_y_min", -50.0)
        self.declare_parameter("crop_y_max", 50.0)
        self.declare_parameter("crop_z_min", -5.0)
        self.declare_parameter("crop_z_max", 5.0)

        self.declare_parameter("enable_fixed_sampling", False)
        self.declare_parameter("fixed_num_points", 4096)

        ###############################################
        # Dataset loader
        ###############################################
        self.ds = UniversalCalibrationDataset(
            dataset_root=self.dataset_root,
            split=self.split,
            combined_index=False,
        )

        self.i = max(0, self.start_index)
        self.n_total = len(self.ds)
        if self.max_samples > 0:
            self.n_end = min(self.n_total, self.i + self.max_samples)
        else:
            self.n_end = self.n_total

        self.get_logger().info(
            "DataPreprocessorNode started (offline).\n"
            f"  dataset_root={self.dataset_root}\n"
            f"  split={self.split}\n"
            f"  start_index={self.i}\n"
            f"  end_index={self.n_end}\n"
            f"  publish_rate_hz={self.publish_rate_hz}\n"
            f"  use_dynamic_rig={self.use_dynamic_rig}\n"
            f"  dynamic_representation={self.dynamic_representation}\n"
            f"  out_image_topic={self.out_image_topic}\n"
            f"  out_points_topic={self.out_points_topic}\n"
            f"  out_scan_topic={self.out_scan_topic}\n"
            f"  out_camera_info_topic={self.out_camera_info_topic}\n"
            f"  out_odom_topic={self.out_odom_topic}"
        )

        if self.publish_rate_hz <= 0.0:
            self.get_logger().error("publish_rate_hz must be > 0.")
            raise RuntimeError("Invalid publish_rate_hz.")

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._tick)

    def _build_camera_info_msg(self, intrinsics: dict) -> CameraInfo:
        """
        Build CameraInfo from the dataset loader intrinsics dict.

          - We control extraction + loader, and therefore enforce ONE intrinsics schema.
          - For the static rig dataset loader, intrinsics are expected as:
                {
                  "width":  <int>,
                  "height": <int>,
                  "fx": <float>,
                  "fy": <float>,
                  "cx": <float>,
                  "cy": <float>,
                }

        If any required field is missing, we raise immediately. Publishing a zero K silently
        would invalidate all projection-based metrics.
        """
        if not isinstance(intrinsics, dict):
            raise RuntimeError(f"intrinsics must be a dict, got: {type(intrinsics)}")

        required = ["width", "height", "fx", "fy", "cx", "cy"]
        missing = [k for k in required if k not in intrinsics]
        if missing:
            raise RuntimeError(
                "Missing required intrinsics keys: " + ", ".join(missing) +
                f". Got keys: {list(intrinsics.keys())}"
            )

        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame_id

        msg.width = width
        msg.height = height

        # No distortion in current dataset extraction.
        msg.distortion_model = "plumb_bob"
        msg.d = []

        # K (row-major):
        # [ fx  0  cx ]
        # [  0 fy  cy ]
        # [  0  0   1 ]
        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        # Rectification (identity)
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # Projection matrix P (3x4). We use K in the left 3x3, last column zeros.
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        return msg

    def _preprocess_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Deterministic, lightweight point filtering:
          - optional box crop
          - optional fixed-count sampling
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)

        # Box crop
        if bool(self.get_parameter("enable_box_crop").value) and pts.shape[0] > 0:
            x_min = float(self.get_parameter("crop_x_min").value)
            x_max = float(self.get_parameter("crop_x_max").value)
            y_min = float(self.get_parameter("crop_y_min").value)
            y_max = float(self.get_parameter("crop_y_max").value)
            z_min = float(self.get_parameter("crop_z_min").value)
            z_max = float(self.get_parameter("crop_z_max").value)

            m = (
                (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
                (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            )
            pts = pts[m]

        # Fixed-count sampling (deterministic)
        if bool(self.get_parameter("enable_fixed_sampling").value) and pts.shape[0] > 0:
            n = int(self.get_parameter("fixed_num_points").value)
            if n > 0 and pts.shape[0] > n:
                rng = np.random.RandomState(self.i)
                idx = rng.choice(pts.shape[0], size=n, replace=False)
                pts = pts[idx]

        return pts.astype(np.float32)
    
    def _build_laserscan_msg(self, scan: dict, stamp, frame_id: str) -> LaserScan:
        """
        Build LaserScan from loader-provided scan dict.

        Expected dynamic sample schema:
            {
                "ranges": [...],
                "angle_min": ...,
                "angle_max": ...,
                "angle_increment": ...,
                "range_min": ...,
                "range_max": ...
            }
        """
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.angle_min = float(scan.get("angle_min", 0.0))
        msg.angle_max = float(scan.get("angle_max", 0.0))
        msg.angle_increment = float(scan.get("angle_increment", 0.0))
        msg.time_increment = float(scan.get("time_increment", 0.0))
        msg.scan_time = float(scan.get("scan_time", 0.0))
        msg.range_min = float(scan.get("range_min", 0.0))
        msg.range_max = float(scan.get("range_max", 0.0))

        ranges = scan.get("ranges", [])
        intensities = scan.get("intensities", [])

        msg.ranges = [float(x) for x in ranges]
        msg.intensities = [float(x) for x in intensities] if intensities else []

        return msg

    def _build_odom_msg(self, odom: dict, stamp, frame_id: str) -> Odometry:
        """
        Build Odometry from loader-provided odom dict.
        This stays permissive so playback does not crash if some optional fields are absent.
        """
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = str(odom.get("header_frame_id", "odom"))
        msg.child_frame_id = str(odom.get("child_frame_id", frame_id))

        pos = odom.get("position", {})
        ori = odom.get("orientation", {})
        lin = odom.get("linear_velocity", {})
        ang = odom.get("angular_velocity", {})

        msg.pose.pose.position.x = float(pos.get("x", 0.0))
        msg.pose.pose.position.y = float(pos.get("y", 0.0))
        msg.pose.pose.position.z = float(pos.get("z", 0.0))

        msg.pose.pose.orientation.x = float(ori.get("x", 0.0))
        msg.pose.pose.orientation.y = float(ori.get("y", 0.0))
        msg.pose.pose.orientation.z = float(ori.get("z", 0.0))
        msg.pose.pose.orientation.w = float(ori.get("w", 1.0))

        msg.twist.twist.linear.x = float(lin.get("x", 0.0))
        msg.twist.twist.linear.y = float(lin.get("y", 0.0))
        msg.twist.twist.linear.z = float(lin.get("z", 0.0))

        msg.twist.twist.angular.x = float(ang.get("x", 0.0))
        msg.twist.twist.angular.y = float(ang.get("y", 0.0))
        msg.twist.twist.angular.z = float(ang.get("z", 0.0))

        return msg

    def _tick(self):
        if self.i >= self.n_end:
            self.get_logger().info("Reached end of dataset. Stopping timer.")
            self.timer.cancel()
            return

        s = self.ds.load_sample(self.i)

        # --- Image preprocessing ---
        img_bgr = s["image"]  # BGR uint8 from loader
        if bool(self.get_parameter("enable_image_denoise").value):
            k = int(self.get_parameter("denoise_kernel").value)
            if k < 1:
                k = 1
            if k % 2 == 0:
                k += 1
            # Mild blur ("noise suppression")
            img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)

        # Publish image
        msg_img = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        msg_img.header.stamp = self.get_clock().now().to_msg()
        msg_img.header.frame_id = self.camera_frame_id
        self.image_pub.publish(msg_img)

        # Publish camera info (per-run intrinsics)
        msg_cam = self._build_camera_info_msg(s["intrinsics"])
        msg_cam.header.stamp = msg_img.header.stamp
        msg_cam.header.frame_id = self.camera_frame_id
        self.cam_info_pub.publish(msg_cam)

        # --- LiDAR / scan preprocessing ---
        # Static rig:
        #   - always publish PointCloud2 from stored points
        #
        # Dynamic rig:
        #   - raw_scan mode: publish stored LaserScan on /eval/clean/scan
        #   - pseudo_points mode: publish stored pseudo point cloud on /eval/clean/points
        #
        #   The dataset extractor already stores both raw scan and pseudo-3D for the dynamic rig.

        if self.use_dynamic_rig:
            if self.dynamic_representation == "raw_scan":
                scan = s.get("scan", None)
                if scan is None:
                    raise RuntimeError(
                        "Dynamic rig sample has no raw scan, but dynamic_representation='raw_scan'."
                    )

                scan_msg = self._build_laserscan_msg(
                    scan=scan,
                    stamp=msg_img.header.stamp,
                    frame_id="lidar",
                )
                self.scan_pub.publish(scan_msg)

            elif self.dynamic_representation == "pseudo_points":
                if s.get("points", None) is None:
                    raise RuntimeError(
                        "Dynamic rig sample has no stored pseudo points, but dynamic_representation='pseudo_points'."
                    )

                pts = np.asarray(s["points"], dtype=np.float32)
                pts = self._preprocess_points(pts)

                fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                cloud = point_cloud2.create_cloud(msg_img.header, fields, pts.tolist())
                self.points_pub.publish(cloud)

        else:
            pts = np.asarray(s["points"], dtype=np.float32)
            pts = self._preprocess_points(pts)

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            cloud = point_cloud2.create_cloud(msg_img.header, fields, pts.tolist())
            self.points_pub.publish(cloud)

        # Optional odometry publish if present in dataset sample
        odom = s.get("odom", None)
        if odom is not None:
            odom_msg = self._build_odom_msg(
                odom=odom,
                stamp=msg_img.header.stamp,
                frame_id=self.camera_frame_id,
            )
            self.odom_pub.publish(odom_msg)

        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = DataPreprocessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()