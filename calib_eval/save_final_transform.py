import os
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
from rclpy.node import Node
import yaml


class SaveTransform(Node):

    def __init__(self):
        super().__init__("save_transform")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("output_path", "estimated_extrinsics.yaml")
        self.declare_parameter("extrinsics_topic", "/eval/estimated_extrinsics")

        # Parameters for best selection
        self.declare_parameter("metric_topic", "/eval/metrics/reprojection_pixel_error_px")
        self.declare_parameter("metric_name", "reprojection_pixel_error_px")

        self.output_path = str(self.get_parameter("output_path").value)
        self.extrinsics_topic = str(self.get_parameter("extrinsics_topic").value)

        self.metric_topic = str(self.get_parameter("metric_topic").value)
        self.metric_name = str(self.get_parameter("metric_name").value)

        # -----------------------------
        # Internal state
        # -----------------------------
        self.write_count = 0
        self.output_path_logged = False

        self.latest_transform = None

        self.best_transform = None
        self.best_metric_value = float("inf")
        self.best_metric_valid = False

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.sub_transform = self.create_subscription(
            TransformStamped,
            self.extrinsics_topic,
            self.transform_callback,
            10
        )

        self.sub_metric = self.create_subscription(
            Float32,
            self.metric_topic,
            self.metric_callback,
            10
        )

    # -----------------------------
    # Callbacks
    # -----------------------------
    def transform_callback(self, msg):
        self.latest_transform = msg
        self.write_yaml()

    def metric_callback(self, msg):
        value = float(msg.data)

        if not math.isfinite(value):
            return

        if self.latest_transform is None:
            return

        if value < self.best_metric_value:
            self.best_metric_value = value
            self.best_transform = self.latest_transform
            self.best_metric_valid = True

    # -----------------------------
    # YAML writing
    # -----------------------------
    def transform_to_dict(self, msg):
        return {
            "parent_frame": msg.header.frame_id,
            "child_frame": msg.child_frame_id,
            "translation": {
                "x": float(msg.transform.translation.x),
                "y": float(msg.transform.translation.y),
                "z": float(msg.transform.translation.z),
            },
            "rotation": {
                "x": float(msg.transform.rotation.x),
                "y": float(msg.transform.rotation.y),
                "z": float(msg.transform.rotation.z),
                "w": float(msg.transform.rotation.w),
            },
        }

    def write_yaml(self):
        if self.latest_transform is None:
            return

        data = {
            "convention": {
                "description": "camera_to_lidar",
                "note": "Transform expresses LiDAR pose in camera frame"
            },
            "selection_metric": {
                "name": self.metric_name,
                "best_value": float(self.best_metric_value) if self.best_metric_valid else None,
                "best_value_valid": bool(self.best_metric_valid),
            },
            "last_transform": self.transform_to_dict(self.latest_transform),
        }

        if self.best_metric_valid and self.best_transform is not None:
            data["best_transform"] = self.transform_to_dict(self.best_transform)
        else:
            data["best_transform"] = None

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        self.write_count += 1

        if not self.output_path_logged:
            self.get_logger().info(f"Saving transforms (last + best) to {self.output_path}")
            self.output_path_logged = True

        if self.write_count % 25 == 0:
            self.get_logger().info(
                f"Updated transform YAML ({self.write_count} writes): {self.output_path}"
            )


def main():
    rclpy.init()
    node = SaveTransform()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()