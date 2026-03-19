###############################################
# SYSTEM-LEVEL EVALUATION PIPELINE
###############################################
# Subscribes to metric topics from atomic evaluators and prints a compact report.
##############################################################################################

import os
import csv
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class EvaluationPipelineNode(Node):

    def __init__(self):
        super().__init__('evaluation_pipeline_node')

        self.declare_parameter('evaluation_config_path', os.path.expanduser('~/evaluation.yaml'))
        self.declare_parameter('output_path_override', '')

        cfg_path = self.get_parameter('evaluation_config_path').get_parameter_value().string_value
        output_path_override = str(self.get_parameter('output_path_override').value).strip()

        self.config = self._load_eval_config(cfg_path)

        if output_path_override:
            self.output_path = os.path.expanduser(output_path_override)
        else:
            self.output_path = os.path.expanduser(
                self.config.get('output', {}).get('output_path', '~/evaluation_results/')
            )

        os.makedirs(self.output_path, exist_ok=True)

        csv_path = os.path.join(self.output_path, 'evaluation_metrics.csv')
        file_exists = os.path.exists(csv_path)

        self.csv_file = open(csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header if file is new/empty
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            self.csv_writer.writerow([
                'timestamp',
                'reproj_visibility',
                'reproj_pixel_error_px',
                'edge_hit_ratio',
                'ref_translation_dev_m',
                'ref_rotation_dev_deg',
                'ref_translation_dev_std_m',
                'ref_rotation_dev_std_deg'
            ])
            self.csv_file.flush()

        # ------------------------------------------------------------------

        self.sub_reproj_vis = self.create_subscription(
            Float32, '/eval/metrics/reprojection_visibility', self.cb_reproj_vis, 10
        )
        self.sub_reproj_pixel = self.create_subscription(
            Float32, '/eval/metrics/reprojection_pixel_error_px', self.cb_reproj_pixel, 10
        )

        self.sub_edge_ratio = self.create_subscription(
            Float32, '/eval/metrics/edge_hit_ratio', self.cb_edge_ratio, 10
        )

        self.sub_ref_t = self.create_subscription(
            Float32, '/eval/metrics/ref_translation_dev_m', self.cb_ref_t, 10
        )
        self.sub_ref_r = self.create_subscription(
            Float32, '/eval/metrics/ref_rotation_dev_deg', self.cb_ref_r, 10
        )
        self.sub_ref_t_std = self.create_subscription(
            Float32, '/eval/metrics/ref_translation_dev_std_m', self.cb_ref_t_std, 10
        )
        self.sub_ref_r_std = self.create_subscription(
            Float32, '/eval/metrics/ref_rotation_dev_std_deg', self.cb_ref_r_std, 10
        )

        # --- Latest values ---
        self.reproj_visibility = None
        self.reproj_pixel_error_px = None
        self.edge_hit_ratio = None
        self.ref_t_dev = None
        self.ref_r_dev = None
        self.ref_t_std = None
        self.ref_r_std = None

        # Report timer
        self.declare_parameter('report_rate_hz', 2.0)
        report_rate = float(self.get_parameter('report_rate_hz').value)
        self.report_timer = self.create_timer(1.0 / max(report_rate, 0.1), self.report)

        self.get_logger().info(
            f"EvaluationPipelineNode cfg={cfg_path} "
            f"output_path_override={output_path_override if output_path_override else '<none>'} "
            f"CSV_path={csv_path}"
        )

    def _load_eval_config(self, path):
        defaults = {
            'output': {'output_path': os.path.expanduser('~/evaluation_results/')},
        }
        try:
            p = os.path.expanduser(path)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                for k, v in defaults.items():
                    if k not in loaded:
                        loaded[k] = v
                return loaded
        except Exception as e:
            self.get_logger().warn(f"Failed to load eval config: {e}")
        return defaults

    # Callbacks
    def cb_reproj_vis(self, msg): self.reproj_visibility = float(msg.data)
    def cb_reproj_pixel(self, msg): self.reproj_pixel_error_px = float(msg.data)
    def cb_edge_ratio(self, msg): self.edge_hit_ratio = float(msg.data)
    def cb_ref_t(self, msg): self.ref_t_dev = float(msg.data)
    def cb_ref_r(self, msg): self.ref_r_dev = float(msg.data)
    def cb_ref_t_std(self, msg): self.ref_t_std = float(msg.data)
    def cb_ref_r_std(self, msg): self.ref_r_std = float(msg.data)

    def report(self):
        parts = []

        if self.reproj_visibility is not None: parts.append(f"reproj_vis={self.reproj_visibility:.3f}")
        if self.reproj_pixel_error_px is not None: parts.append(f"reproj_px_err={self.reproj_pixel_error_px:.2f}")
        if self.edge_hit_ratio is not None: parts.append(f"edge_hit={self.edge_hit_ratio:.3f}")
        if self.ref_t_dev is not None: parts.append(f"dev_t_m={self.ref_t_dev:.4f}")
        if self.ref_r_dev is not None: parts.append(f"dev_r_deg={self.ref_r_dev:.2f}")
        if self.ref_t_std is not None: parts.append(f"std_t_m={self.ref_t_std:.4f}")
        if self.ref_r_std is not None: parts.append(f"std_r_deg={self.ref_r_std:.2f}")

        if parts:
            self.get_logger().info("[SYSTEM EVAL] " + " | ".join(parts))

        # Mandatory CSV write
        ts = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([
            ts,
            self.reproj_visibility,
            self.reproj_pixel_error_px,
            self.edge_hit_ratio,
            self.ref_t_dev,
            self.ref_r_dev,
            self.ref_t_std,
            self.ref_r_std
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationPipelineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()