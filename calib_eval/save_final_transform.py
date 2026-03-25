import os

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
import yaml


class SaveTransform(Node):

    def __init__(self):
        super().__init__("save_transform")

        self.declare_parameter("output_path", "estimated_extrinsics.yaml")
        self.declare_parameter("extrinsics_topic", "/eval/estimated_extrinsics")

        self.output_path = str(self.get_parameter("output_path").value)
        self.extrinsics_topic = str(self.get_parameter("extrinsics_topic").value)

        self.write_count = 0
        self.output_path_logged = False

        self.sub = self.create_subscription(
            TransformStamped,
            self.extrinsics_topic,
            self.callback,
            10
        )

    def callback(self, msg):
        data = {
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

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        self.write_count += 1

        if not self.output_path_logged:
            self.get_logger().info(f"Saving latest estimated extrinsics to {self.output_path}")
            self.output_path_logged = True

        if self.write_count % 25 == 0:
            self.get_logger().info(
                f"Updated estimated extrinsics YAML ({self.write_count} writes): {self.output_path}"
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