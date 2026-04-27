###############################################
# Loads a baseline/reference extrinsics YAML
# and publishes it as:
#   /eval/ref_extrinsics  (TransformStamped)
# This reference is NOT ground truth; it is a baseline estimate.
###############################################


import os
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory

from .geometry_utils import quat_normalize_xyzw


class ReferenceExtrinsicsPublisher(Node):

    def __init__(self):
        super().__init__('reference_extrinsics_publisher')

        # Primary input: direct reference YAML path
        self.declare_parameter('reference_yaml_path', '')

        # Optional: evaluation.yaml path (used as fallback to locate reference yaml)
        self.declare_parameter('evaluation_config_path', '')

        # Rig config ID used to pick a default reference YAML from package share:
        #   - "static_mount_h1" -> static_mount_h1_reference.yaml
        #   - "static_mount_h2" -> static_mount_h2_reference.yaml
        #   - "dynamic"         -> dynamic_reference.yaml
        self.declare_parameter('rig_config_id', 'static_mount_h1')

        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('parent_frame_override', '')
        self.declare_parameter('child_frame_override', '')

        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.parent_override = str(self.get_parameter('parent_frame_override').value).strip()
        self.child_override = str(self.get_parameter('child_frame_override').value).strip()
        self.rig_config_id = str(self.get_parameter('rig_config_id').value).strip()

        # Resolve reference YAML path (direct param wins; config fallback otherwise)
        yaml_path = self._resolve_reference_yaml_path(
            str(self.get_parameter('reference_yaml_path').value).strip(),
            str(self.get_parameter('evaluation_config_path').value).strip()
        )

        # Publisher
        self.pub = self.create_publisher(TransformStamped, '/eval/ref_extrinsics', 10)

        # Load & validate YAML once
        self.ref = self._load_and_validate_reference_yaml(yaml_path)

        # Optional overrides
        if self.parent_override:
            self.ref['parent_frame'] = self.parent_override
        if self.child_override:
            self.ref['child_frame'] = self.child_override

        # Timer loop
        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn("publish_rate_hz <= 0. Publishing disabled.")
            self.timer = None
        else:
            self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._tick)

        self.get_logger().info(
            "ReferenceExtrinsicsPublisher started.\n"
            f"  reference_yaml={yaml_path}\n"
            f"  publishing /eval/ref_extrinsics @ {self.publish_rate_hz} Hz\n"
            f"  parent_frame={self.ref['parent_frame']}\n"
            f"  child_frame={self.ref['child_frame']}\n"
            "  NOTE: This is baseline/reference (not GT)."
        )

    def _resolve_reference_yaml_path(self, direct_ref_path: str, eval_cfg_path: str) -> str:
        """
        Path resolution logic:
          1) If direct_ref_path exists -> use it
          2) Else, try to read eval_cfg_path (or package default) and use reference.reference_yaml_path
          3) Else, use rig_config_id + package share default reference YAML
        """
        def exists(p): return p and os.path.exists(os.path.expanduser(p))

        if exists(direct_ref_path):
            return os.path.expanduser(direct_ref_path)

        # If evaluation_config_path not provided, use package share default.
        if not eval_cfg_path.strip():
            pkg_share = get_package_share_directory("calib_eval")
            eval_cfg_path = os.path.join(pkg_share, "config", "evaluation.yaml")

        eval_cfg_path = os.path.expanduser(eval_cfg_path)

        if os.path.exists(eval_cfg_path):
            with open(eval_cfg_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            ref = cfg.get('reference', {}) if isinstance(cfg, dict) else {}
            ref_path = str(ref.get('reference_yaml_path', '')).strip()
            if exists(ref_path):
                return os.path.expanduser(ref_path)

        # Final fallback: rig_config_id-based reference YAML from package share.
        pkg_share = get_package_share_directory("calib_eval")
        ref_dir = os.path.join(pkg_share, "config", "reference")

        if self.rig_config_id.lower() in ["dynamic", "dynamic_rig"]:
            candidate = os.path.join(ref_dir, "dynamic_reference.yaml")
        else:
            # e.g. "static_mount_h1" -> "static_mount_h1_reference.yaml"
            candidate = os.path.join(ref_dir, f"{self.rig_config_id}_reference.yaml")

        if os.path.exists(candidate):
            return candidate

        raise RuntimeError(
            "Could not resolve reference YAML path.\n"
            "Provide either:\n"
            "  - ROS param: reference_yaml_path:=/path/to/reference.yaml\n"
            "or\n"
            "  - ROS param: evaluation_config_path:=/path/to/evaluation.yaml with reference.reference_yaml_path set."
            "\n"
            "or\n"
            "  - ROS param: rig_config_id:=static_mount_h1/static_mount_h2/dynamic (package share default)."
        )

    def _load_and_validate_reference_yaml(self, path: str) -> dict:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"Reference YAML path does not exist: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        for key in ['translation', 'rotation_quat', 'parent_frame', 'child_frame']:
            if key not in data:
                raise RuntimeError(f"Reference YAML missing required key: '{key}'")

        t = self._to_float_list(data['translation'], 'translation', 3)
        q = self._to_float_list(data['rotation_quat'], 'rotation_quat', 4)
        q = [float(x) for x in quat_normalize_xyzw(q)]

        parent_frame = str(data['parent_frame']).strip()
        child_frame = str(data['child_frame']).strip()
        if not parent_frame or not child_frame:
            raise RuntimeError("parent_frame and child_frame must be non-empty strings.")

        out = dict(data)
        out['translation'] = t
        out['rotation_quat'] = q
        out['parent_frame'] = parent_frame
        out['child_frame'] = child_frame
        return out

    def _to_float_list(self, arr, name: str, n_expected: int):
        if not isinstance(arr, list) or len(arr) != n_expected:
            raise RuntimeError(f"{name} must be a list of length {n_expected}.")
        out = []
        for i, v in enumerate(arr):
            try:
                out.append(float(v))
            except Exception:
                raise RuntimeError(
                    f"{name}[{i}] is not numeric ('{v}'). Replace placeholders with real numbers."
                )
        return out

    def _tick(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.ref['parent_frame']
        tf_msg.child_frame_id = self.ref['child_frame']

        tf_msg.transform.translation.x = float(self.ref['translation'][0])
        tf_msg.transform.translation.y = float(self.ref['translation'][1])
        tf_msg.transform.translation.z = float(self.ref['translation'][2])

        tf_msg.transform.rotation.x = float(self.ref['rotation_quat'][0])
        tf_msg.transform.rotation.y = float(self.ref['rotation_quat'][1])
        tf_msg.transform.rotation.z = float(self.ref['rotation_quat'][2])
        tf_msg.transform.rotation.w = float(self.ref['rotation_quat'][3])

        self.pub.publish(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceExtrinsicsPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()