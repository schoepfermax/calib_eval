#!/usr/bin/env python3
"""

Publishes a correct sensor_msgs/CameraInfo at a fixed rate, loaded from a YAML file
(ROS camera_calibration format).

Why:
- This node guarantees valid intrinsics for dataset extraction + reprojection evaluation.
"""

import os
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class CameraInfoRepublisher(Node):
    def __init__(self):
        super().__init__('camera_info_republisher')

        # ---- Parameters ----
        self.declare_parameter('camera_info_yaml', '/home/khan/calib/basler.yaml')
        self.declare_parameter('output_topic', '/basler/camera/camera_info_calibrated')
        self.declare_parameter('frame_id', 'pylon_camera')
        self.declare_parameter('publish_rate_hz', 10.0)

        yaml_path = os.path.expanduser(self.get_parameter('camera_info_yaml').value)
        self.output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # ---- Load YAML once (not every publish) ----
        self.cam_info_msg = self._load_camera_info_yaml(yaml_path)
        self.cam_info_msg.header.frame_id = self.frame_id

        # ---- Publisher ----
        self.pub = self.create_publisher(CameraInfo, self.output_topic, 10)

        # ---- Timer ----
        period = 1.0 / max(rate_hz, 0.1)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            "CameraInfoRepublisher started.\n"
            f"  yaml={yaml_path}\n"
            f"  output_topic={self.output_topic}\n"
            f"  frame_id={self.frame_id}\n"
            f"  publish_rate_hz={rate_hz}\n"
        )

    def _load_camera_info_yaml(self, yaml_path: str) -> CameraInfo:
        """Reads ROS camera_calibration YAML and returns a CameraInfo message."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Camera calibration YAML not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        msg = CameraInfo()
        msg.width = int(data['image_width'])
        msg.height = int(data['image_height'])

        # Distortion
        msg.distortion_model = str(data.get('distortion_model', 'plumb_bob'))
        msg.d = list(map(float, data['distortion_coefficients']['data']))

        # Intrinsics K
        msg.k = list(map(float, data['camera_matrix']['data']))

        # Rectification R
        msg.r = list(map(float, data['rectification_matrix']['data']))

        # Projection P
        msg.p = list(map(float, data['projection_matrix']['data']))

        return msg

    def _on_timer(self):
        """Publishes the CameraInfo periodically with updated timestamp."""
        self.cam_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.cam_info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
