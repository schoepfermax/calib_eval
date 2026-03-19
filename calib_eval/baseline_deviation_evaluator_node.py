###############################################
# Evaluator: Baseline deviation
###############################################
#
# Compare a model's estimated extrinsics against a reference/baseline
# extrinsics (e.g., Abdul Haq tool output) and track repeatability.
#
# Subscribes:
#   /eval/estimated_extrinsics  (TransformStamped)  <- from model wrapper
#   /eval/ref_extrinsics        (TransformStamped)  <- from baseline publisher node
#
# Publishes:
#   /eval/metrics/ref_translation_dev_m
#   /eval/metrics/ref_rotation_dev_deg
#   /eval/metrics/ref_translation_dev_std_m
#   /eval/metrics/ref_rotation_dev_std_deg
###############################################

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TransformStamped
import numpy as np
from calib_eval.geometry_utils import (
    quat_normalize_xyzw,
    quat_geodesic_angle_deg_xyzw,
    welford_update,
    welford_std,
)


class BaselineDeviationEvaluatorNode(Node):
    """
    Computes deviation of estimated extrinsics from a reference baseline.
    """

    def __init__(self):
        super().__init__('baseline_deviation_evaluator_node')

        ###############################################
        # Parameters
        ###############################################
        self.declare_parameter('estimated_extrinsics_topic', '/eval/estimated_extrinsics')
        self.declare_parameter('reference_extrinsics_topic', '/eval/ref_extrinsics')

        est_topic = str(self.get_parameter('estimated_extrinsics_topic').value).strip()
        ref_topic = str(self.get_parameter('reference_extrinsics_topic').value).strip()

        ###############################################
        # Subscriptions
        ###############################################
        self.est_sub = self.create_subscription(
            TransformStamped, est_topic, self.est_callback, 10
        )
        self.ref_sub = self.create_subscription(
            TransformStamped, ref_topic, self.ref_callback, 10
        )

        ###############################################
        # Publishers: deviation + running std
        ###############################################
        self.pub_t_dev = self.create_publisher(Float32, '/eval/metrics/ref_translation_dev_m', 10)
        self.pub_r_dev = self.create_publisher(Float32, '/eval/metrics/ref_rotation_dev_deg', 10)

        self.pub_t_std = self.create_publisher(Float32, '/eval/metrics/ref_translation_dev_std_m', 10)
        self.pub_r_std = self.create_publisher(Float32, '/eval/metrics/ref_rotation_dev_std_deg', 10)

        ###############################################
        # Internal state (latest messages)
        ###############################################
        self.latest_est = None
        self.latest_ref = None

        ###############################################
        # Running stats (Welford) for translation deviation
        ###############################################
        self.n_t = 0
        self.mean_t = 0.0
        self.m2_t = 0.0

        ###############################################
        # Running stats (Welford) for rotation deviation
        ###############################################
        self.n_r = 0
        self.mean_r = 0.0
        self.m2_r = 0.0

        self.get_logger().info(
            "BaselineDeviationEvaluatorNode started (Option C).\n"
            f"  estimated_extrinsics_topic={est_topic}\n"
            f"  reference_extrinsics_topic={ref_topic}\n"
            "Waiting for both topics..."
        )

    ###############################################
    # Helper methods (math utilities)
    ###############################################
    @staticmethod
    def _quat_normalize(q):
        """
        Normalizes a quaternion. If the norm is ~0, returns identity quaternion.

        NOTE:
          Implementation is centralized in geometry_utils.py to ensure consistent
          conventions across the pipeline.
        """
        return quat_normalize_xyzw(q)

    @staticmethod
    def _translation_from_tf(tf_msg: TransformStamped):
        """
        Extracts translation vector [x, y, z] from a TransformStamped.
        """
        return np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z
        ], dtype=np.float32)

    @staticmethod
    def _quat_from_tf(tf_msg: TransformStamped):
        """
        Extracts and normalizes quaternion [x, y, z, w] from a TransformStamped.
        """
        return BaselineDeviationEvaluatorNode._quat_normalize([
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w
        ])

    @staticmethod
    def _quat_angle_deg(q1, q2):
        """
        Computes the geodesic angle between two unit quaternions in degrees.

        Using abs(dot) makes it invariant to sign flip (q and -q same rotation).

        NOTE:
          Implementation is centralized in geometry_utils.py to ensure consistent
          conventions across the pipeline.
        """
        return quat_geodesic_angle_deg_xyzw(q1, q2)

    @staticmethod
    def _welford_update(n, mean, m2, x):
        """
        One-step Welford update for running mean/variance.
        Returns updated (n, mean, m2).

        NOTE:
          Implementation is centralized in geometry_utils.py.
        """
        return welford_update(n, mean, m2, x)

    @staticmethod
    def _welford_std(n, m2):
        """
        Returns sample standard deviation if n >= 2, else 0.0.

        NOTE:
          Implementation is centralized in geometry_utils.py.
        """
        return welford_std(n, m2)

    ###############################################
    # ROS callbacks
    ###############################################
    def est_callback(self, msg):
        self.latest_est = msg
        self.try_eval()

    def ref_callback(self, msg):
        self.latest_ref = msg
        self.try_eval()

    ###############################################
    # Main evaluation
    ###############################################
    def try_eval(self):
        """
        Runs deviation computation once both reference and estimate are available.
        """
        if self.latest_est is None or self.latest_ref is None:
            return

        # Warn if frames are inconsistent
        if self.latest_est.header.frame_id and self.latest_ref.header.frame_id:
            if self.latest_est.header.frame_id != self.latest_ref.header.frame_id:
                self.get_logger().warn(
                    f"Frame mismatch: est.frame_id='{self.latest_est.header.frame_id}' "
                    f"ref.frame_id='{self.latest_ref.header.frame_id}'. "
                    "This may indicate inconsistent conventions."
                )

        # Extract transforms
        t_est = self._translation_from_tf(self.latest_est)
        q_est = self._quat_from_tf(self.latest_est)

        t_ref = self._translation_from_tf(self.latest_ref)
        q_ref = self._quat_from_tf(self.latest_ref)

        # Instant deviation metrics
        t_dev = float(np.linalg.norm(t_est - t_ref))
        r_dev = self._quat_angle_deg(q_est, q_ref)

        # Publish instant deviation
        self.pub_t_dev.publish(Float32(data=t_dev))
        self.pub_r_dev.publish(Float32(data=r_dev))

        # Update running stats and publish running std
        self.n_t, self.mean_t, self.m2_t = self._welford_update(self.n_t, self.mean_t, self.m2_t, t_dev)
        self.n_r, self.mean_r, self.m2_r = self._welford_update(self.n_r, self.mean_r, self.m2_r, r_dev)

        t_std = self._welford_std(self.n_t, self.m2_t)
        r_std = self._welford_std(self.n_r, self.m2_r)

        self.pub_t_std.publish(Float32(data=t_std))
        self.pub_r_std.publish(Float32(data=r_std))


def main(args=None):
    rclpy.init(args=args)
    node = BaselineDeviationEvaluatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()