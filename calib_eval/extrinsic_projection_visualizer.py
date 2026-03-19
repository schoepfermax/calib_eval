#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import cv2

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation


class ExtrinsicProjectionVisualizer(Node):

    def __init__(self):
        super().__init__("extrinsic_projection_visualizer")

        self.bridge = CvBridge()

        # load extrinsic YAML
        self.declare_parameter("extrinsics_yaml", "")
        path = self.get_parameter("extrinsics_yaml").value

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        t = np.array(data["translation"], dtype=np.float32)
        q = np.array(data["rotation_quat"], dtype=np.float32)

        R = Rotation.from_quat(q).as_matrix()

        self.T_cam_lidar = np.eye(4)
        self.T_cam_lidar[:3, :3] = R
        self.T_cam_lidar[:3, 3] = t

        self.get_logger().info("Loaded reference extrinsics")

        self.K = None

        self.image = None
        self.points = None

        self.create_subscription(Image, "/eval/clean/image", self.img_cb, 10)
        self.create_subscription(PointCloud2, "/eval/clean/points", self.pc_cb, 10)
        self.create_subscription(CameraInfo, "/eval/camera_info", self.cam_cb, 10)

        self.timer = self.create_timer(0.2, self.render)

    def cam_cb(self, msg):
        K = np.array(msg.k).reshape(3, 3)
        self.K = K

    def img_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def pc_cb(self, msg):
        pts = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        self.points = np.array(pts)

    def render(self):

        if self.image is None or self.points is None or self.K is None:
            return

        img = self.image.copy()

        pts = self.points

        # transform lidar -> camera
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_cam = (self.T_cam_lidar @ pts_h.T).T[:, :3]

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        for X, Y, Z in pts_cam:

            if Z <= 0:
                continue

            u = int(fx * X / Z + cx)
            v = int(fy * Y / Z + cy)

            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (0, 255, 0), -1)

        cv2.imshow("LiDAR projection", img)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ExtrinsicProjectionVisualizer()
    rclpy.spin(node)


if __name__ == "__main__":
    main()