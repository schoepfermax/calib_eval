import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import yaml
import os

class SaveTransform(Node):

    def __init__(self):
        super().__init__("save_transform")

        self.declare_parameter("output_path", "estimated_extrinsics.yaml")

        self.output_path = self.get_parameter("output_path").value

        self.sub = self.create_subscription(
            TransformStamped,
            "/eval/estimated_extrinsics",
            self.callback,
            10
        )

        self.saved = False

    def callback(self, msg):

        if self.saved:
            return

        data = {
            "parent_frame": msg.header.frame_id,
            "child_frame": msg.child_frame_id,
            "translation": {
                "x": msg.transform.translation.x,
                "y": msg.transform.translation.y,
                "z": msg.transform.translation.z,
            },
            "rotation": {
                "x": msg.transform.rotation.x,
                "y": msg.transform.rotation.y,
                "z": msg.transform.rotation.z,
                "w": msg.transform.rotation.w,
            },
        }

        with open(self.output_path, "w") as f:
            yaml.dump(data, f)

        self.get_logger().info(f"Saved transform to {self.output_path}")

        self.saved = True


def main():
    rclpy.init()
    node = SaveTransform()
    rclpy.spin(node)

if __name__ == "__main__":
    main()