#!/usr/bin/env python3
###############################################
# Offline2D Calibration Node
#
# Classical multi-frame edge-alignment calibration
# for 2D laser scanner + camera rigs.
#
# Pipeline position:
#   data_preprocessor_node
#       ↓
#   offline2d
#       ↓
#   evaluators
###############################################

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge

from calib_eval import dynamic_utils as utils

class Offline2DNode(Node):

    def __init__(self):
        super().__init__("offline2d")

        self.bridge = CvBridge()

        ###############################################
        # PARAMETERS
        ###############################################

        self.declare_parameter("image_topic", "/eval/clean/image")
        self.declare_parameter("scan_topic", "/eval/clean/scan")
        self.declare_parameter("camera_info_topic", "/eval/camera_info")
        self.declare_parameter("reference_extrinsics_topic", "/eval/ref_extrinsics")
        self.declare_parameter("output_extrinsics_topic", "/eval/offline2d_extrinsics")
        self.declare_parameter("max_frames", 150)
        self.declare_parameter("top_k", 40)
        self.declare_parameter("max_sync_dt_sec", 0.05)

        self.image_topic = str(self.get_parameter("image_topic").value).strip()
        self.scan_topic = str(self.get_parameter("scan_topic").value).strip()
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value).strip()
        self.reference_extrinsics_topic = str(self.get_parameter("reference_extrinsics_topic").value).strip()
        self.output_extrinsics_topic = str(self.get_parameter("output_extrinsics_topic").value).strip()
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.top_k = int(self.get_parameter("top_k").value)
        self.max_sync_dt_sec = float(self.get_parameter("max_sync_dt_sec").value)

        ###############################################
        # SUBSCRIBERS
        ###############################################

        self.sub_img = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            10
        )

        self.sub_scan = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_cb,
            10
        )

        self.sub_cam = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.caminfo_cb,
            10
        )

        self.sub_ref = self.create_subscription(
            TransformStamped,
            self.reference_extrinsics_topic,
            self.ref_tf_cb,
            10
        )

        ###############################################
        # PUBLISHERS
        ###############################################

        self.pub_extrinsics = self.create_publisher(
            TransformStamped,
            self.output_extrinsics_topic,
            10
        )

        ###############################################
        # INTERNAL BUFFERS
        ###############################################

        self.images = []
        self.scans = []
        self.timestamps = []

        self.intrinsics = None

        self.latest_scan = None
        self.latest_scan_stamp_ns = None
        self.latest_camera_info_stamp_ns = None
        self.latest_ref_tf = None

        self.finished = False

        self.get_logger().info(
            "offline2d node started.\n"
            f"  image_topic={self.image_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  reference_extrinsics_topic={self.reference_extrinsics_topic}\n"
            f"  output_extrinsics_topic={self.output_extrinsics_topic}\n"
            f"  max_frames={self.max_frames}\n"
            f"  top_k={self.top_k}\n"
            f"  max_sync_dt_sec={self.max_sync_dt_sec}"
        )

    ###############################################
    # CALLBACKS
    ###############################################

    def caminfo_cb(self, msg: CameraInfo):

        self.latest_camera_info_stamp_ns = self._stamp_to_ns(msg.header.stamp)

        if self.intrinsics is None:
            self.intrinsics = utils.intrinsics_from_camerainfo(msg)
            self.get_logger().info("Received intrinsics.")

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        self.latest_scan_stamp_ns = self._stamp_to_ns(msg.header.stamp)

    def ref_tf_cb(self, msg: TransformStamped):
        # Stored for future use / diagnostics.
        self.latest_ref_tf = msg

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

    def image_cb(self, msg: Image):

        if self.finished:
            return

        if self.latest_scan is None:
            return

        if self.intrinsics is None:
            return

        if self.latest_scan_stamp_ns is None or self.latest_camera_info_stamp_ns is None:
            return

        image_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        max_sync_dt_ns = int(self.max_sync_dt_sec * 1e9)

        if abs(image_stamp_ns - self.latest_scan_stamp_ns) > max_sync_dt_ns:
            return

        if abs(image_stamp_ns - self.latest_camera_info_stamp_ns) > max_sync_dt_ns:
            return

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        self.images.append(img)
        self.scans.append(self.latest_scan)
        self.timestamps.append(msg.header.stamp.sec)

        if len(self.images) >= self.max_frames:

            self.get_logger().info(
                f"Collected {len(self.images)} frames. Running calibration."
            )

            self.run_calibration()

            self.finished = True

    ###############################################
    # MAIN CALIBRATION
    ###############################################

    def run_calibration(self):

        frames = []

        for img, scan in zip(self.images, self.scans):

            edge_score = utils.edge_density(img)

            frames.append({
                "image": img,
                "scan": scan,
                "score": edge_score
            })

        frames = utils.select_top_k_frames(frames, self.top_k)

        imgs = []
        scans = []

        for f in frames:
            imgs.append(f["image"])
            scans.append(utils.scan_to_points(f["scan"]))

        imgs = np.array(imgs, dtype=object)
        scans = np.array(scans, dtype=object)

        extrinsic = utils.solve_multiframe_alignment(
            imgs,
            scans,
            self.intrinsics
        )

        self.publish_extrinsics(extrinsic)

    ###############################################
    # PUBLISH
    ###############################################

    def publish_extrinsics(self, T):

        msg = TransformStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.child_frame_id = "lidar"

        t = T[:3, 3]

        q = utils.rotation_to_quaternion(T[:3, :3])

        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])

        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]

        self.pub_extrinsics.publish(msg)

        self.get_logger().info(
            "offline2d calibration completed and published."
        )


def main(args=None):

    rclpy.init(args=args)

    node = Offline2DNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()