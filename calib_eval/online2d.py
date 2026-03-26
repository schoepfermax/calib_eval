#!/usr/bin/env python3
###############################################
# Online2D Calibration Refinement Node
#
# Mandatory second stage for the 2D calibration path.
#
# This node consumes offline-played topics from the dataset-backed
# data_preprocessor_node and refines the initial offline extrinsics over
# rolling windows of image + scan + odometry.
#
# Design notes:
#   - still offline / dataset-driven in the thesis pipeline
#   - sequential / online-style in its update logic
#   - keeps the best-so-far transform
#   - freezes refinement once meaningful deterioration is detected
#
# Diagnostic notes:
#   - This version adds explicit logging so we can see whether online2d
#     is:
#         * receiving usable samples
#         * entering window evaluation
#         * finding candidate improvements
#         * rejecting updates due to gating / limits
#         * freezing due to deterioration
#   - The actual refinement behavior is intentionally unchanged here.
###############################################

import math
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from collections import deque

from calib_eval import dynamic_utils as utils


class Online2DNode(Node):
    def __init__(self):
        super().__init__("online2d")

        self.bridge = CvBridge()

        ###############################################
        # PARAMETERS
        ###############################################
        self.declare_parameter("image_topic", "/eval/clean/image")
        self.declare_parameter("scan_topic", "/eval/clean/scan")
        self.declare_parameter("camera_info_topic", "/eval/camera_info")
        self.declare_parameter("odom_topic", "/eval/odom")
        self.declare_parameter("input_extrinsics_topic", "/eval/offline2d_extrinsics")
        self.declare_parameter("output_extrinsics_topic", "/eval/estimated_extrinsics")

        self.declare_parameter("camera_frame_id", "camera")
        self.declare_parameter("lidar_frame_id", "lidar")

        self.declare_parameter("window_size", 20)
        self.declare_parameter("min_window_size", 6)
        self.declare_parameter("evaluation_stride", 1)
        self.declare_parameter("deterioration_patience", 5)
        self.declare_parameter("deterioration_tolerance", 1e-4)

        self.declare_parameter("min_translation_for_update_m", 0.02)
        self.declare_parameter("min_yaw_for_update_deg", 2.0)

        self.declare_parameter("max_translation_step_m", 0.05)
        self.declare_parameter("max_rotation_step_deg", 2.0)
        self.declare_parameter("min_improvement", 1e-4)
        self.declare_parameter("max_sync_dt_sec", 1.0)

        self.declare_parameter("canny_low", 100)
        self.declare_parameter("canny_high", 200)
        self.declare_parameter("blur_kernel", 3)
        self.declare_parameter("clip_px", 30.0)
        self.declare_parameter("min_projected_points", 20)
        self.declare_parameter("visibility_penalty_value", 100.0)

        self.declare_parameter("edge_ratio_gate", 0.0025)
        self.declare_parameter("mean_grad_gate", 8.0)

        self.declare_parameter("translation_step_candidates", [0.0, 0.01, 0.03])
        self.declare_parameter("rotation_step_candidates_deg", [0.0, 0.5, 1.0])

        self.image_topic = str(self.get_parameter("image_topic").value).strip()
        self.scan_topic = str(self.get_parameter("scan_topic").value).strip()
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value).strip()
        self.odom_topic = str(self.get_parameter("odom_topic").value).strip()
        self.input_extrinsics_topic = str(self.get_parameter("input_extrinsics_topic").value).strip()
        self.output_extrinsics_topic = str(self.get_parameter("output_extrinsics_topic").value).strip()

        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value).strip()
        self.lidar_frame_id = str(self.get_parameter("lidar_frame_id").value).strip()

        self.window_size = int(self.get_parameter("window_size").value)
        self.min_window_size = int(self.get_parameter("min_window_size").value)
        self.evaluation_stride = int(self.get_parameter("evaluation_stride").value)
        self.patience = int(self.get_parameter("deterioration_patience").value)
        self.deterioration_tolerance = float(self.get_parameter("deterioration_tolerance").value)

        self.min_translation_for_window = float(self.get_parameter("min_translation_for_update_m").value)
        self.min_rotation_deg_for_window = float(self.get_parameter("min_yaw_for_update_deg").value)

        self.max_translation_step = float(self.get_parameter("max_translation_step_m").value)
        self.max_rotation_step_deg = float(self.get_parameter("max_rotation_step_deg").value)
        self.min_improvement = float(self.get_parameter("min_improvement").value)
        self.max_sync_dt_sec = float(self.get_parameter("max_sync_dt_sec").value)

        self.eval_kwargs = {
            "canny_low": int(self.get_parameter("canny_low").value),
            "canny_high": int(self.get_parameter("canny_high").value),
            "blur_kernel": int(self.get_parameter("blur_kernel").value),
            "clip_px": float(self.get_parameter("clip_px").value),
            "min_projected_points": int(self.get_parameter("min_projected_points").value),
            "visibility_penalty_value": float(self.get_parameter("visibility_penalty_value").value),
        }


        self.edge_ratio_gate = float(self.get_parameter("edge_ratio_gate").value)
        self.mean_grad_gate = float(self.get_parameter("mean_grad_gate").value)

        self.translation_step_candidates = sorted(
            list({abs(float(x)) for x in self.get_parameter("translation_step_candidates").value})
        )
        self.rotation_step_candidates_deg = sorted(
            list({abs(float(x)) for x in self.get_parameter("rotation_step_candidates_deg").value})
        )

        ###############################################
        # INTERNAL STATE
        ###############################################
        self.intrinsics = None
        self.latest_camera_info_stamp_ns = None
        self.latest_scan = None
        self.latest_scan_stamp_ns = None
        self.latest_odom = None
        self.latest_odom_stamp_ns = None

        self.initial_extrinsics_received = False
        self.refinement_frozen = False

        self.current_t = None
        self.current_q = None
        self.best_t = None
        self.best_q = None
        self.best_cost = float("inf")
        self.recent_costs = []
        self.num_updates_applied = 0
        self.num_windows_evaluated = 0
        self.num_windows_rejected = 0

        ###############################################
        # TIME-SYNC BUFFERS (NEW)
        ###############################################
        self.scan_buffer = deque(maxlen=50)
        self.odom_buffer = deque(maxlen=50)

        self.pre_init_window = deque(maxlen=max(self.window_size, self.min_window_size))
        self.sample_window = deque(maxlen=max(self.window_size, self.min_window_size))
        self.window_counter = 0

        ###############################################
        # DIAGNOSTIC COUNTERS
        ###############################################
        self.total_images_seen = 0
        self.total_samples_buffered = 0

        self.skip_no_intrinsics = 0
        self.skip_no_init = 0
        self.skip_no_scan = 0
        self.skip_no_odom = 0
        self.skip_no_stamps = 0
        self.skip_sync_scan = 0
        self.skip_sync_odom = 0
        self.skip_sync_cam = 0
        self.skip_scene_gate = 0
        self.skip_small_window = 0
        self.skip_stride = 0
        self.skip_low_motion = 0

        self.last_skip_log_total = 0
        self.last_window_eval_log_count = 0

        ###############################################
        # PUB / SUB
        ###############################################
        self.pub_extrinsics = self.create_publisher(
            TransformStamped,
            self.output_extrinsics_topic,
            10,
        )

        self.sub_cam = self.create_subscription(CameraInfo, self.camera_info_topic, self._caminfo_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.sub_img = self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.sub_extr = self.create_subscription(
            TransformStamped,
            self.input_extrinsics_topic,
            self._extrinsics_cb,
            10,
        )

        self.get_logger().info(
            "online2d node started.\n"
            f"  image_topic={self.image_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  odom_topic={self.odom_topic}\n"
            f"  input_extrinsics_topic={self.input_extrinsics_topic}\n"
            f"  output_extrinsics_topic={self.output_extrinsics_topic}\n"
            f"  window_size={self.window_size}\n"
            f"  min_window_size={self.min_window_size}\n"
            f"  patience={self.patience}\n"
            f"  max_sync_dt_sec={self.max_sync_dt_sec}"
        )

    ###############################################
    # MESSAGE CONVERSION HELPERS
    ###############################################
    def _camera_info_to_intrinsics_dict(self, msg: CameraInfo):
        return utils.intrinsics_dict_from_camera_info_or_loader_dict({
            "width": int(msg.width),
            "height": int(msg.height),
            "K": list(msg.k),
        })

    def _scan_msg_to_scan_dict(self, msg: LaserScan):
        return {
            "ranges": [float(x) for x in msg.ranges],
            "intensities": [float(x) for x in msg.intensities] if len(msg.intensities) > 0 else [],
            "angle_min": float(msg.angle_min),
            "angle_max": float(msg.angle_max),
            "angle_increment": float(msg.angle_increment),
            "time_increment": float(msg.time_increment),
            "scan_time": float(msg.scan_time),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
        }

    def _odom_msg_to_dict(self, msg: Odometry):
        return {
            "position": {
                "x": float(msg.pose.pose.position.x),
                "y": float(msg.pose.pose.position.y),
                "z": float(msg.pose.pose.position.z),
            },
            "orientation": {
                "x": float(msg.pose.pose.orientation.x),
                "y": float(msg.pose.pose.orientation.y),
                "z": float(msg.pose.pose.orientation.z),
                "w": float(msg.pose.pose.orientation.w),
            },
            "linear_velocity": {
                "x": float(msg.twist.twist.linear.x),
                "y": float(msg.twist.twist.linear.y),
                "z": float(msg.twist.twist.linear.z),
            },
            "angular_velocity": {
                "x": float(msg.twist.twist.angular.x),
                "y": float(msg.twist.twist.angular.y),
                "z": float(msg.twist.twist.angular.z),
            },
            "header_frame_id": str(msg.header.frame_id),
            "child_frame_id": str(msg.child_frame_id),
        }

    def _transform_msg_to_t_q(self, msg: TransformStamped):
        t = np.array([
            float(msg.transform.translation.x),
            float(msg.transform.translation.y),
            float(msg.transform.translation.z),
        ], dtype=np.float32)
        q = np.array([
            float(msg.transform.rotation.x),
            float(msg.transform.rotation.y),
            float(msg.transform.rotation.z),
            float(msg.transform.rotation.w),
        ], dtype=np.float32)
        return t, utils.normalize_quaternion_xyzw(q)

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

    ###############################################
    # DIAGNOSTIC HELPERS
    ###############################################
    def _maybe_log_skip_summary(self):
        current_total = (
            self.skip_no_intrinsics
            + self.skip_no_init
            + self.skip_no_scan
            + self.skip_no_odom
            + self.skip_no_stamps
            + self.skip_sync_scan
            + self.skip_sync_odom
            + self.skip_sync_cam
            + self.skip_scene_gate
            + self.skip_small_window
            + self.skip_stride
            + self.skip_low_motion
        )

        # Log every 100 new skip events to avoid flooding.
        if current_total - self.last_skip_log_total < 100:
            return

        self.last_skip_log_total = current_total

        self.get_logger().info(
            "online2d skip summary:\n"
            f"  total_images_seen={self.total_images_seen}\n"
            f"  total_samples_buffered={self.total_samples_buffered}\n"
            f"  skip_no_intrinsics={self.skip_no_intrinsics}\n"
            f"  skip_no_init={self.skip_no_init}\n"
            f"  skip_no_scan={self.skip_no_scan}\n"
            f"  skip_no_odom={self.skip_no_odom}\n"
            f"  skip_no_stamps={self.skip_no_stamps}\n"
            f"  skip_sync_scan={self.skip_sync_scan}\n"
            f"  skip_sync_odom={self.skip_sync_odom}\n"
            f"  skip_sync_cam={self.skip_sync_cam}\n"
            f"  skip_scene_gate={self.skip_scene_gate}\n"
            f"  skip_small_window={self.skip_small_window}\n"
            f"  skip_stride={self.skip_stride}\n"
            f"  skip_low_motion={self.skip_low_motion}"
        )

    ###############################################
    # CALLBACKS
    ###############################################
    def _caminfo_cb(self, msg: CameraInfo):
        self.latest_camera_info_stamp_ns = self._stamp_to_ns(msg.header.stamp)

        if self.intrinsics is None:
            self.intrinsics = self._camera_info_to_intrinsics_dict(msg)
            self.get_logger().info("online2d received intrinsics.")

    def scan_cb(self, msg):
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        scan_dict = self._scan_msg_to_scan_dict(msg)

        self.latest_scan = scan_dict
        self.latest_scan_stamp_ns = stamp_ns

        self.scan_buffer.append({
            "stamp_ns": stamp_ns,
            "scan": scan_dict
        })

    def _scan_cb(self, msg):
        self.scan_cb(msg)

    def odom_cb(self, msg):
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        odom_dict = self._odom_msg_to_dict(msg)

        self.latest_odom = odom_dict
        self.latest_odom_stamp_ns = stamp_ns

        self.odom_buffer.append({
            "stamp_ns": stamp_ns,
            "odom": odom_dict
        })

    def _odom_cb(self, msg):
        self.odom_cb(msg)

    def _extrinsics_cb(self, msg: TransformStamped):
        # Capture the first incoming extrinsics as the initialization from offline2d,
        # then ignore later messages to avoid self-feedback if input and output topics
        # are intentionally the same.
        if self.initial_extrinsics_received:
            return

        self.current_t, self.current_q = self._transform_msg_to_t_q(msg)
        self.best_t = self.current_t.copy()
        self.best_q = self.current_q.copy()
        self.initial_extrinsics_received = True

        # Publish immediately so evaluators always have a valid estimated transform,
        # even if no refinement update is later accepted.
        self._publish_transform(self.current_t, self.current_q)

        if len(self.pre_init_window) > 0:
            promoted_count = len(self.pre_init_window)
            for bundle in self.pre_init_window:
                self.sample_window.append(bundle)
            self.window_counter += promoted_count
            self.total_samples_buffered += promoted_count
            self.pre_init_window.clear()

            self.get_logger().info(
                f"online2d promoted buffered pre-init samples into active window: {promoted_count}"
            )

            if len(self.sample_window) >= self.min_window_size:
                self._evaluate_window()

        self.get_logger().info(
            "online2d received initial extrinsics, published them, and is ready to refine."
        )

    def _image_cb(self, msg: Image):
        self.total_images_seen += 1

        if self.refinement_frozen:
            return

        if self.intrinsics is None:
            self.skip_no_intrinsics += 1
            self._maybe_log_skip_summary()
            return

        # ------------------------------------------------------------
        # PRE-INIT PHASE:
        # The offline dataset player already publishes image + scan + odom
        # from the SAME dataset sample using the SAME playback stamp.
        # So for dynamic 2d offline playback, we should consume the latest
        # current tuple directly instead of re-solving synchronization here.
        # ------------------------------------------------------------
        if not self.initial_extrinsics_received:
            if self.latest_scan is None:
                self.skip_no_scan += 1
                self._maybe_log_skip_summary()
                return

            if self.latest_odom is None:
                self.skip_no_odom += 1
                self._maybe_log_skip_summary()
                return

            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.pre_init_window.append({
                "image": image_bgr,
                "scan": self.latest_scan,
                "intrinsics": self.intrinsics,
                "weight": 1.0,
                "odom": self.latest_odom,
            })
            self.total_samples_buffered += 1

            if len(self.pre_init_window) == 1 or (len(self.pre_init_window) % 10) == 0:
                self.get_logger().info(
                    f"online2d buffered pre-init samples: {len(self.pre_init_window)}"
                )
            return

        # ------------------------------------------------------------
        # POST-INIT PHASE:
        # Same playback contract applies here as well.
        # Consume the latest scan + odom belonging to the current played sample.
        # ------------------------------------------------------------
        if self.latest_scan is None:
            self.skip_no_scan += 1
            self._maybe_log_skip_summary()
            return

        if self.latest_odom is None:
            self.skip_no_odom += 1
            self._maybe_log_skip_summary()
            return

        scan = self.latest_scan
        odom = self.latest_odom

        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        _, _, _, edge_ratio, mean_grad = utils.compute_edge_map_and_distance_transform(
            image_bgr,
            canny_low=self.eval_kwargs["canny_low"],
            canny_high=self.eval_kwargs["canny_high"],
            blur_kernel=self.eval_kwargs["blur_kernel"],
        )

        # Scene-quality gate: skip very weak frames.
        if edge_ratio < self.edge_ratio_gate or mean_grad < self.mean_grad_gate:
            self.skip_scene_gate += 1
            self._maybe_log_skip_summary()
            return

        bundle = {
            "image": image_bgr,
            "scan": scan,
            "intrinsics": self.intrinsics,
            "weight": 1.0,
            "odom": odom,
        }
        self.sample_window.append(bundle)
        self.window_counter += 1
        self.total_samples_buffered += 1

        if self.total_samples_buffered == 1 or (self.total_samples_buffered % 25) == 0:
            self.get_logger().info(
                f"online2d accepted samples into window: total_seen={self.total_samples_buffered}, "
                f"window_len={len(self.sample_window)}"
            )

        if len(self.sample_window) < self.min_window_size:
            self.skip_small_window += 1
            self._maybe_log_skip_summary()
            return

        if self.evaluation_stride > 1 and (self.window_counter % self.evaluation_stride) != 0:
            self.skip_stride += 1
            self._maybe_log_skip_summary()
            return

        self._evaluate_window()

    def _find_nearest(self, buffer, target_stamp_ns, max_dt_ns):
        best = None
        best_dt = None

        for item in buffer:
            dt = abs(item["stamp_ns"] - target_stamp_ns)
            if dt <= max_dt_ns:
                if best is None or dt < best_dt:
                    best = item
                    best_dt = dt

        return best

    ###############################################
    # REFINEMENT
    ###############################################
    def _evaluate_window(self):
        bundle_list = list(self.sample_window)
        odom_sequence = [b["odom"] for b in bundle_list]
        motion = utils.odom_window_motion_metrics(odom_sequence)

        if motion["translation_total"] < self.min_translation_for_window and \
           motion["rotation_total_deg"] < self.min_rotation_deg_for_window:
            self.skip_low_motion += 1
            self._maybe_log_skip_summary()
            return

        current_eval = utils.evaluate_multiframe_edge_cost(
            sample_bundle_list=bundle_list,
            translation_xyz=self.current_t,
            quaternion_xyzw=self.current_q,
            **self.eval_kwargs,
        )
        current_cost = float(current_eval["mean_total_cost"])

        # Initialize best_cost on first valid evaluation.
        if not math.isfinite(self.best_cost):
            self.best_cost = current_cost
            self.best_t = self.current_t.copy()
            self.best_q = self.current_q.copy()

        candidate_t, candidate_q, candidate_cost = self._search_local_update(bundle_list)
        self.num_windows_evaluated += 1
        self.recent_costs.append(candidate_cost)

        translation_step = utils.translation_distance(self.current_t, candidate_t)
        rotation_step_deg = utils.quaternion_angle_deg(self.current_q, candidate_q)

        accepted = utils.refinement_should_accept_update(
            previous_cost=current_cost,
            candidate_cost=candidate_cost,
            translation_step=translation_step,
            rotation_step_deg=rotation_step_deg,
            max_translation_step=self.max_translation_step,
            max_rotation_step_deg=self.max_rotation_step_deg,
            min_improvement=self.min_improvement,
        )

        # Periodic diagnostic logging for window evaluation.
        if self.num_windows_evaluated == 1 or (self.num_windows_evaluated % 10) == 0:
            self.get_logger().info(
                "online2d window evaluation:\n"
                f"  num_windows_evaluated={self.num_windows_evaluated}\n"
                f"  current_cost={current_cost:.6f}\n"
                f"  candidate_cost={candidate_cost:.6f}\n"
                f"  improvement={current_cost - candidate_cost:.6f}\n"
                f"  translation_step={translation_step:.6f} m\n"
                f"  rotation_step={rotation_step_deg:.6f} deg\n"
                f"  motion_translation_total={motion['translation_total']:.6f} m\n"
                f"  motion_rotation_total_deg={motion['rotation_total_deg']:.6f} deg\n"
                f"  accepted={accepted}\n"
                f"  best_cost={self.best_cost:.6f}\n"
                f"  num_updates_applied={self.num_updates_applied}\n"
                f"  num_windows_rejected={self.num_windows_rejected}"
            )

        if accepted:
            self.current_t = candidate_t.copy()
            self.current_q = candidate_q.copy()
            self.num_updates_applied += 1

            if candidate_cost < self.best_cost:
                self.best_cost = float(candidate_cost)
                self.best_t = candidate_t.copy()
                self.best_q = candidate_q.copy()
                self._publish_transform(self.best_t, self.best_q)
                self.get_logger().info(
                    f"online2d accepted update. best_cost={self.best_cost:.4f}, "
                    f"translation_step={translation_step:.4f} m, rotation_step={rotation_step_deg:.3f} deg"
                )
            else:
                self.get_logger().info(
                    "online2d accepted candidate for current state, "
                    "but it did not beat historical best_cost, so no publish occurred.\n"
                    f"  candidate_cost={candidate_cost:.6f}\n"
                    f"  best_cost={self.best_cost:.6f}"
                )
        else:
            self.num_windows_rejected += 1

            rejection_reasons = []
            if candidate_cost >= current_cost - self.min_improvement:
                rejection_reasons.append("insufficient_improvement")
            if translation_step > self.max_translation_step:
                rejection_reasons.append("translation_step_too_large")
            if rotation_step_deg > self.max_rotation_step_deg:
                rejection_reasons.append("rotation_step_too_large")
            if not rejection_reasons:
                rejection_reasons.append("rejected_by_refinement_should_accept_update")

            if self.num_windows_rejected == 1 or (self.num_windows_rejected % 10) == 0:
                self.get_logger().info(
                    "online2d rejected candidate update.\n"
                    f"  reasons={','.join(rejection_reasons)}\n"
                    f"  current_cost={current_cost:.6f}\n"
                    f"  candidate_cost={candidate_cost:.6f}\n"
                    f"  translation_step={translation_step:.6f} m\n"
                    f"  rotation_step={rotation_step_deg:.6f} deg"
                )

        if utils.deterioration_detected(
            best_cost=self.best_cost,
            recent_costs=self.recent_costs,
            patience=self.patience,
            tolerance=self.deterioration_tolerance,
        ):
            self.refinement_frozen = True
            self._publish_transform(self.best_t, self.best_q)
            self.get_logger().warn(
                "online2d detected sustained deterioration. Refinement frozen at best-so-far transform."
            )

    def _search_local_update(self, bundle_list):
        base_param = utils.make_parameter_vector_from_translation_quaternion(
            self.current_t,
            self.current_q,
        )

        trans_steps = []
        for v in self.translation_step_candidates:
            trans_steps.extend([-v, 0.0, v])
        trans_steps = sorted(list(set(trans_steps)))

        rot_steps = []
        for deg in self.rotation_step_candidates_deg:
            rad = math.radians(deg)
            rot_steps.extend([-rad, 0.0, rad])
        rot_steps = sorted(list(set(rot_steps)))

        candidates = utils.build_coarse_candidate_grid(
            base_param_vec=base_param,
            translation_steps_xyz=trans_steps,
            rotation_steps_rpy=rot_steps,
            stage_mask=None,
        )

        best_t = self.current_t.copy()
        best_q = self.current_q.copy()
        best_eval = utils.evaluate_multiframe_edge_cost(
            sample_bundle_list=bundle_list,
            translation_xyz=best_t,
            quaternion_xyzw=best_q,
            **self.eval_kwargs,
        )
        best_cost = float(best_eval["mean_total_cost"])

        for cand in candidates:
            cand = np.asarray(cand, dtype=np.float32).reshape(6,)
            cand_t, cand_q = utils.make_translation_quaternion_from_parameter_vector(cand)
            ev = utils.evaluate_multiframe_edge_cost(
                sample_bundle_list=bundle_list,
                translation_xyz=cand_t,
                quaternion_xyzw=cand_q,
                **self.eval_kwargs,
            )
            cost = float(ev["mean_total_cost"])
            if cost < best_cost:
                best_cost = cost
                best_t = cand_t.copy()
                best_q = cand_q.copy()

        return best_t, best_q, best_cost

    ###############################################
    # OUTPUT
    ###############################################
    def _publish_transform(self, translation_xyz, quaternion_xyzw):
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


def main(args=None):
    rclpy.init(args=args)
    node = Online2DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()