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
#   online2d
#       ↓
#   evaluators
#
# Design notes:
#   - This node computes the initial 2D scan-camera extrinsics.
#   - It uses the dynamic pipeline helpers from dynamic_utils.py.
#   - The implementation is explicit and bounded rather than using hidden
#     optimizer behavior.
#   - Reference extrinsics are used only as the initialization / baseline.
#   - Synchronization is done with a small explicit "pending latest sample"
#     strategy so that image-first publish order does not starve collection.
###############################################

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan

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

        self.declare_parameter("camera_frame_id", "camera")
        self.declare_parameter("lidar_frame_id", "lidar")

        self.declare_parameter("max_frames", 150)
        self.declare_parameter("top_k", 40)
        self.declare_parameter("max_sync_dt_sec", 0.05)

        # Evaluation settings
        self.declare_parameter("canny_low", 100)
        self.declare_parameter("canny_high", 200)
        self.declare_parameter("blur_kernel", 3)
        self.declare_parameter("clip_px", 30.0)
        self.declare_parameter("min_projected_points", 20)
        self.declare_parameter("visibility_penalty_value", 100.0)

        # Search bounds around the reference / initialization.
        self.declare_parameter("translation_bound_x_m", 0.30)
        self.declare_parameter("translation_bound_y_m", 0.30)
        self.declare_parameter("translation_bound_z_m", 0.30)

        self.declare_parameter("rotation_bound_roll_deg", 10.0)
        self.declare_parameter("rotation_bound_pitch_deg", 10.0)
        self.declare_parameter("rotation_bound_yaw_deg", 15.0)

        # Stage A = coarse constrained search.
        self.declare_parameter("stage_a_translation_steps_x_m", [-0.10, -0.05, 0.0, 0.05, 0.10])
        self.declare_parameter("stage_a_translation_steps_y_m", [-0.10, -0.05, 0.0, 0.05, 0.10])
        self.declare_parameter("stage_a_translation_steps_z_m", [-0.05, 0.0, 0.05])
        self.declare_parameter("stage_a_rotation_steps_roll_deg", [-2.0, 0.0, 2.0])
        self.declare_parameter("stage_a_rotation_steps_pitch_deg", [-2.0, 0.0, 2.0])
        self.declare_parameter("stage_a_rotation_steps_yaw_deg", [-4.0, -2.0, 0.0, 2.0, 4.0])

        # Stage B = full local refinement.
        self.declare_parameter("stage_b_translation_steps_x_m", [-0.02, -0.01, 0.0, 0.01, 0.02])
        self.declare_parameter("stage_b_translation_steps_y_m", [-0.02, -0.01, 0.0, 0.01, 0.02])
        self.declare_parameter("stage_b_translation_steps_z_m", [-0.01, 0.0, 0.01])
        self.declare_parameter("stage_b_rotation_steps_roll_deg", [-0.5, 0.0, 0.5])
        self.declare_parameter("stage_b_rotation_steps_pitch_deg", [-0.5, 0.0, 0.5])
        self.declare_parameter("stage_b_rotation_steps_yaw_deg", [-1.0, -0.5, 0.0, 0.5, 1.0])

        # Search mask for Stage A. We keep all enabled by default.
        self.declare_parameter("stage_a_enable_tx", True)
        self.declare_parameter("stage_a_enable_ty", True)
        self.declare_parameter("stage_a_enable_tz", True)
        self.declare_parameter("stage_a_enable_roll", True)
        self.declare_parameter("stage_a_enable_pitch", True)
        self.declare_parameter("stage_a_enable_yaw", True)

        self.image_topic = str(self.get_parameter("image_topic").value).strip()
        self.scan_topic = str(self.get_parameter("scan_topic").value).strip()
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value).strip()
        self.reference_extrinsics_topic = str(self.get_parameter("reference_extrinsics_topic").value).strip()
        self.output_extrinsics_topic = str(self.get_parameter("output_extrinsics_topic").value).strip()

        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value).strip()
        self.lidar_frame_id = str(self.get_parameter("lidar_frame_id").value).strip()

        self.max_frames = int(self.get_parameter("max_frames").value)
        self.top_k = int(self.get_parameter("top_k").value)
        self.max_sync_dt_sec = float(self.get_parameter("max_sync_dt_sec").value)

        self.eval_kwargs = {
            "canny_low": int(self.get_parameter("canny_low").value),
            "canny_high": int(self.get_parameter("canny_high").value),
            "blur_kernel": int(self.get_parameter("blur_kernel").value),
            "clip_px": float(self.get_parameter("clip_px").value),
            "min_projected_points": int(self.get_parameter("min_projected_points").value),
            "visibility_penalty_value": float(self.get_parameter("visibility_penalty_value").value),
        }

        self.translation_bounds_xyz = [
            (
                -float(self.get_parameter("translation_bound_x_m").value),
                float(self.get_parameter("translation_bound_x_m").value),
            ),
            (
                -float(self.get_parameter("translation_bound_y_m").value),
                float(self.get_parameter("translation_bound_y_m").value),
            ),
            (
                -float(self.get_parameter("translation_bound_z_m").value),
                float(self.get_parameter("translation_bound_z_m").value),
            ),
        ]

        self.rotation_bound_halfwidths_rpy = [
            math.radians(float(self.get_parameter("rotation_bound_roll_deg").value)),
            math.radians(float(self.get_parameter("rotation_bound_pitch_deg").value)),
            math.radians(float(self.get_parameter("rotation_bound_yaw_deg").value)),
        ]

        self.stage_a_mask = [
            bool(self.get_parameter("stage_a_enable_tx").value),
            bool(self.get_parameter("stage_a_enable_ty").value),
            bool(self.get_parameter("stage_a_enable_tz").value),
            bool(self.get_parameter("stage_a_enable_roll").value),
            bool(self.get_parameter("stage_a_enable_pitch").value),
            bool(self.get_parameter("stage_a_enable_yaw").value),
        ]

        self.stage_a_translation_steps_xyz = [
            [float(v) for v in self.get_parameter("stage_a_translation_steps_x_m").value],
            [float(v) for v in self.get_parameter("stage_a_translation_steps_y_m").value],
            [float(v) for v in self.get_parameter("stage_a_translation_steps_z_m").value],
        ]
        self.stage_b_translation_steps_xyz = [
            [float(v) for v in self.get_parameter("stage_b_translation_steps_x_m").value],
            [float(v) for v in self.get_parameter("stage_b_translation_steps_y_m").value],
            [float(v) for v in self.get_parameter("stage_b_translation_steps_z_m").value],
        ]

        self.stage_a_rotation_steps_rpy = [
            [math.radians(float(v)) for v in self.get_parameter("stage_a_rotation_steps_roll_deg").value],
            [math.radians(float(v)) for v in self.get_parameter("stage_a_rotation_steps_pitch_deg").value],
            [math.radians(float(v)) for v in self.get_parameter("stage_a_rotation_steps_yaw_deg").value],
        ]
        self.stage_b_rotation_steps_rpy = [
            [math.radians(float(v)) for v in self.get_parameter("stage_b_rotation_steps_roll_deg").value],
            [math.radians(float(v)) for v in self.get_parameter("stage_b_rotation_steps_pitch_deg").value],
            [math.radians(float(v)) for v in self.get_parameter("stage_b_rotation_steps_yaw_deg").value],
        ]

        ###############################################
        # SUBSCRIBERS
        ###############################################

        self.sub_img = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.sub_cam = self.create_subscription(CameraInfo, self.camera_info_topic, self.caminfo_cb, 10)
        self.sub_ref = self.create_subscription(
            TransformStamped,
            self.reference_extrinsics_topic,
            self.ref_tf_cb,
            10,
        )

        ###############################################
        # PUBLISHERS
        ###############################################

        self.pub_extrinsics = self.create_publisher(
            TransformStamped,
            self.output_extrinsics_topic,
            10,
        )

        ###############################################
        # INTERNAL BUFFERS
        ###############################################

        self.frames = []

        self.intrinsics = None

        self.latest_image_bgr = None
        self.latest_image_stamp_ns = None

        self.latest_scan = None
        self.latest_scan_stamp_ns = None
        self.latest_camera_info_stamp_ns = None

        self.last_collected_image_stamp_ns = None

        self.reference_t = None
        self.reference_q = None

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

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

    def _camera_info_to_intrinsics_dict(self, msg: CameraInfo):
        return utils.intrinsics_dict_from_camera_info_or_loader_dict({
            "width": int(msg.width),
            "height": int(msg.height),
            "k": list(msg.k),
        })

    def _scan_msg_to_scan_dict(self, msg: LaserScan):
        return {
            "angle_min": float(msg.angle_min),
            "angle_max": float(msg.angle_max),
            "angle_increment": float(msg.angle_increment),
            "time_increment": float(msg.time_increment),
            "scan_time": float(msg.scan_time),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
            "ranges": list(msg.ranges),
            "intensities": list(msg.intensities),
        }

    def caminfo_cb(self, msg: CameraInfo):
        self.latest_camera_info_stamp_ns = self._stamp_to_ns(msg.header.stamp)

        if self.intrinsics is None:
            self.intrinsics = self._camera_info_to_intrinsics_dict(msg)
            self.get_logger().info("offline2d received intrinsics.")

        self._try_collect_frame()

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = self._scan_msg_to_scan_dict(msg)
        self.latest_scan_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        self._try_collect_frame()

    def ref_tf_cb(self, msg: TransformStamped):
        self.reference_t = np.array([
            float(msg.transform.translation.x),
            float(msg.transform.translation.y),
            float(msg.transform.translation.z),
        ], dtype=np.float32)
        self.reference_q = utils.normalize_quaternion_xyzw([
            float(msg.transform.rotation.x),
            float(msg.transform.rotation.y),
            float(msg.transform.rotation.z),
            float(msg.transform.rotation.w),
        ])
        self._try_collect_frame()

    def image_cb(self, msg: Image):
        if self.finished:
            return

        self.latest_image_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        self.latest_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._try_collect_frame()

    def _try_collect_frame(self):
        if self.finished:
            return
        if self.latest_image_bgr is None:
            return
        if self.latest_image_stamp_ns is None:
            return
        if self.latest_scan is None:
            return
        if self.intrinsics is None:
            return
        if self.reference_t is None or self.reference_q is None:
            return
        if self.latest_scan_stamp_ns is None or self.latest_camera_info_stamp_ns is None:
            return

        image_stamp_ns = int(self.latest_image_stamp_ns)
        max_sync_dt_ns = int(self.max_sync_dt_sec * 1e9)

        if self.last_collected_image_stamp_ns == image_stamp_ns:
            return

        if abs(image_stamp_ns - self.latest_scan_stamp_ns) > max_sync_dt_ns:
            return

        if abs(image_stamp_ns - self.latest_camera_info_stamp_ns) > max_sync_dt_ns:
            return

        image_bgr = self.latest_image_bgr

        _, _, _, edge_ratio, mean_grad = utils.compute_edge_map_and_distance_transform(
            image_bgr,
            canny_low=self.eval_kwargs["canny_low"],
            canny_high=self.eval_kwargs["canny_high"],
            blur_kernel=self.eval_kwargs["blur_kernel"],
        )

        scan_points = utils.scan_dict_to_points_lidar_frame(self.latest_scan)
        entry = {
            "dataset_index": len(self.frames),
            "image": image_bgr,
            "scan": self.latest_scan,
            "intrinsics": self.intrinsics,
            "quality_score": utils.metadata_quality_score(
                sample_meta=None,
                default_edge_ratio=edge_ratio,
                default_grad=mean_grad,
                default_scan_count=scan_points.shape[0],
            ),
        }
        self.frames.append(entry)
        self.last_collected_image_stamp_ns = image_stamp_ns

        if len(self.frames) == 1 or (len(self.frames) % 25) == 0:
            self.get_logger().info(
                f"offline2d collected frames: {len(self.frames)}/{self.max_frames}"
            )

        if len(self.frames) >= self.max_frames:
            self.get_logger().info(
                f"Collected {len(self.frames)} frames. Running offline 2D calibration."
            )
            self.run_calibration()
            self.finished = True

    ###############################################
    # MAIN CALIBRATION
    ###############################################

    def run_calibration(self):
        selected_frames = utils.select_top_k_diverse_samples(
            sample_entries=self.frames,
            top_k=self.top_k,
            min_index_gap=3,
        )

        if len(selected_frames) == 0:
            self.get_logger().warn("offline2d: no valid frames selected. Publishing reference extrinsics.")
            self.publish_extrinsics(self.reference_t, self.reference_q)
            return

        bundle_list = []
        for frame in selected_frames:
            bundle_list.append({
                "image": frame["image"],
                "scan": frame["scan"],
                "intrinsics": frame["intrinsics"],
                "weight": 1.0,
            })

        init_param = utils.make_parameter_vector_from_translation_quaternion(
            self.reference_t,
            self.reference_q,
        )

        ref_rpy = utils.quaternion_xyzw_to_rpy(self.reference_q)
        rotation_bounds_rpy = [
            (
                float(ref_rpy[0] - self.rotation_bound_halfwidths_rpy[0]),
                float(ref_rpy[0] + self.rotation_bound_halfwidths_rpy[0]),
            ),
            (
                float(ref_rpy[1] - self.rotation_bound_halfwidths_rpy[1]),
                float(ref_rpy[1] + self.rotation_bound_halfwidths_rpy[1]),
            ),
            (
                float(ref_rpy[2] - self.rotation_bound_halfwidths_rpy[2]),
                float(ref_rpy[2] + self.rotation_bound_halfwidths_rpy[2]),
            ),
        ]

        result = utils.coarse_to_fine_search(
            sample_bundle_list=bundle_list,
            init_param_vec=init_param,
            translation_bounds_xyz=self.translation_bounds_xyz,
            rotation_bounds_rpy=rotation_bounds_rpy,
            stage_a_mask=self.stage_a_mask,
            stage_a_translation_steps_xyz=self.stage_a_translation_steps_xyz,
            stage_a_rotation_steps_rpy=self.stage_a_rotation_steps_rpy,
            stage_b_translation_steps_xyz=self.stage_b_translation_steps_xyz,
            stage_b_rotation_steps_rpy=self.stage_b_rotation_steps_rpy,
            evaluation_kwargs=self.eval_kwargs,
        )

        best_t = result["best_translation"]
        best_q = result["best_quaternion"]
        best_cost = float(result["best_eval"]["mean_total_cost"])

        self.get_logger().info(
            f"offline2d calibration completed. selected_frames={len(bundle_list)}, best_cost={best_cost:.6f}"
        )
        self.publish_extrinsics(best_t, best_q)

    ###############################################
    # OUTPUT
    ###############################################

    def publish_extrinsics(self, translation_xyz, quaternion_xyzw):
        msg = TransformStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame_id
        msg.child_frame_id = self.lidar_frame_id

        msg.transform.translation.x = float(translation_xyz[0])
        msg.transform.translation.y = float(translation_xyz[1])
        msg.transform.translation.z = float(translation_xyz[2])

        msg.transform.rotation.x = float(quaternion_xyzw[0])
        msg.transform.rotation.y = float(quaternion_xyzw[1])
        msg.transform.rotation.z = float(quaternion_xyzw[2])
        msg.transform.rotation.w = float(quaternion_xyzw[3])

        self.pub_extrinsics.publish(msg)
        self.get_logger().info("offline2d published initial extrinsics.")


def main(args=None):
    rclpy.init(args=args)
    node = Offline2DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()