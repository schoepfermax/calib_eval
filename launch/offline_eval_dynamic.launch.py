"""
Offline launch (DYNAMIC 2D rig).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

pkg_share = get_package_share_directory("calib_eval")
evaluation_yaml = os.path.join(pkg_share, "config", "evaluation.yaml")


def generate_launch_description():
    # Environment-specific dataset root (laptop vs workstation)
    default_dataset_root = os.environ.get("CALIB_EVAL_DATASET_ROOT", "")

    dataset_root_cfg = LaunchConfiguration("dataset_root")
    split_cfg = LaunchConfiguration("split")
    rig_config_id_cfg = LaunchConfiguration("rig_config_id")
    use_dynamic_rig_cfg = LaunchConfiguration("use_dynamic_rig")

    # Online2D tuning knobs exposed as launch arguments
    online_window_size_cfg = LaunchConfiguration("online_window_size")
    online_min_window_size_cfg = LaunchConfiguration("online_min_window_size")
    online_evaluation_stride_cfg = LaunchConfiguration("online_evaluation_stride")
    online_min_improvement_cfg = LaunchConfiguration("online_min_improvement")
    online_deterioration_patience_cfg = LaunchConfiguration("online_deterioration_patience")
    online_deterioration_tolerance_cfg = LaunchConfiguration("online_deterioration_tolerance")
    online_rotation_step_candidates_deg_cfg = LaunchConfiguration("online_rotation_step_candidates_deg")
    online_translation_step_candidates_cfg = LaunchConfiguration("online_translation_step_candidates")

    return LaunchDescription([
        DeclareLaunchArgument(
            "dataset_root",
            default_value=default_dataset_root,
            description="Offline dataset root folder (environment-specific)."
        ),
        DeclareLaunchArgument(
            "split",
            default_value="train",
            description="Dataset split name (e.g., train/val/test)."
        ),
        DeclareLaunchArgument(
            "rig_config_id",
            default_value="dynamic",
            description="Rig config ID used to select reference YAML (e.g., dynamic)."
        ),
        DeclareLaunchArgument(
            "use_dynamic_rig",
            default_value="true",
            description="Dynamic rig mode (LaserScan-based dataset conventions)."
        ),

        DeclareLaunchArgument(
            "online_window_size",
            default_value="10",
            description="Online2D rolling window size."
        ),
        DeclareLaunchArgument(
            "online_min_window_size",
            default_value="3",
            description="Minimum valid Online2D window size."
        ),
        DeclareLaunchArgument(
            "online_evaluation_stride",
            default_value="1",
            description="Online2D evaluation stride."
        ),
        DeclareLaunchArgument(
            "online_min_improvement",
            default_value="0.0",
            description="Minimum required cost improvement for accepting an Online2D update."
        ),
        DeclareLaunchArgument(
            "online_deterioration_patience",
            default_value="20",
            description="How many worsening Online2D windows to tolerate before freezing."
        ),
        DeclareLaunchArgument(
            "online_deterioration_tolerance",
            default_value="0.0001",
            description="Tolerance used by Online2D deterioration detection."
        ),
        DeclareLaunchArgument(
            "online_rotation_step_candidates_deg",
            default_value="[0.0, 0.25]",
            description="Yaw-step magnitudes in degrees for Online2D local search, e.g. [0.0, 0.10] or [0.0, 0.05, 0.10]."
        ),
        DeclareLaunchArgument(
            "online_translation_step_candidates",
            default_value="[0.0, 0.005]",
            description="Translation-step candidates for Online2D."
        ),

        # 1) Dataset playback / preprocessing
        Node(
            package="calib_eval",
            executable="data_preprocessor_node",
            name="data_preprocessor_node",
            output="screen",
            parameters=[{
                "mode": "offline",
                "dataset_root": dataset_root_cfg,
                "split": split_cfg,
                "publish_rate_hz": 2.0,
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "camera_frame_id": "camera",

                # Choose which stored dynamic-rig representation to publish:
                #   "raw_scan"      -> for the new 2D pipeline
                #   "pseudo_points" -> only when intentionally feeding the dynamic rig into the 3D pipeline
                "dynamic_representation": "raw_scan",

                "out_image_topic": "/eval/clean/image",
                "out_points_topic": "/eval/clean/points",
                "out_scan_topic": "/eval/clean/scan",
                "out_camera_info_topic": "/eval/camera_info",
                "out_odom_topic": "/eval/odom",
            }],
        ),

        # 2) Publish reference/baseline extrinsics
        Node(
            package="calib_eval",
            executable="reference_extrinsics_publisher",
            name="reference_extrinsics_publisher",
            output="screen",
            parameters=[{
                "reference_yaml_path": "",
                "evaluation_config_path": evaluation_yaml,
                "rig_config_id": rig_config_id_cfg,
                "publish_rate_hz": 5.0,
            }],
        ),

        # 3) Offline 2D calibration
        Node(
            package="calib_eval",
            executable="offline2d",
            name="offline2d",
            output="screen",
            parameters=[{
                "image_topic": "/eval/clean/image",
                "scan_topic": "/eval/clean/scan",
                "camera_info_topic": "/eval/camera_info",
                "reference_extrinsics_topic": "/eval/ref_extrinsics",
                "output_extrinsics_topic": "/eval/offline2d_extrinsics",

                # Stronger dynamic-2d diagnostic settings after fixing
                # reference-centered rotation bounds in offline2d.
                "max_frames": 20,
                "top_k": 10,

                # Tighter search around the now-correct reference neighborhood.
                "rotation_bound_roll_deg": 6.0,
                "rotation_bound_pitch_deg": 6.0,
                "rotation_bound_yaw_deg": 8.0,

                # Slightly denser explicit search for the dynamic rig.
                "stage_a_translation_steps_x_m": [-0.06, -0.03, 0.0, 0.03, 0.06],
                "stage_a_translation_steps_y_m": [-0.06, -0.03, 0.0, 0.03, 0.06],
                "stage_a_translation_steps_z_m": [-0.03, 0.0, 0.03],
                "stage_a_rotation_steps_roll_deg": [-1.0, 0.0, 1.0],
                "stage_a_rotation_steps_pitch_deg": [-1.0, 0.0, 1.0],
                "stage_a_rotation_steps_yaw_deg": [-2.0, -1.0, 0.0, 1.0, 2.0],

                "stage_b_translation_steps_x_m": [-0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075],
                "stage_b_translation_steps_y_m": [-0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075],
                "stage_b_translation_steps_z_m": [-0.005, -0.0025, 0.0, 0.0025, 0.005],
                "stage_b_rotation_steps_roll_deg": [-0.25, -0.125, 0.0, 0.125, 0.25],
                "stage_b_rotation_steps_pitch_deg": [-0.25, -0.125, 0.0, 0.125, 0.25],
                "stage_b_rotation_steps_yaw_deg": [-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5],
            }],
        ),

        # 4) Online 2D refinement
        Node(
            package="calib_eval",
            executable="online2d",
            name="online2d",
            output="screen",
            parameters=[{
                "image_topic": "/eval/clean/image",
                "scan_topic": "/eval/clean/scan",
                "camera_info_topic": "/eval/camera_info",
                "odom_topic": "/eval/odom",
                "input_extrinsics_topic": "/eval/offline2d_extrinsics",
                "output_extrinsics_topic": "/eval/estimated_extrinsics",

                # Online2D tuning parameters exposed through launch
                "window_size": online_window_size_cfg,
                "min_window_size": online_min_window_size_cfg,
                "evaluation_stride": online_evaluation_stride_cfg,

                "min_translation_for_update_m": 0.0,
                "min_yaw_for_update_deg": 0.0,

                "max_translation_step_m": 0.10,
                "max_rotation_step_deg": 5.0,
                "min_improvement": online_min_improvement_cfg,
                "deterioration_patience": online_deterioration_patience_cfg,
                "deterioration_tolerance": online_deterioration_tolerance_cfg,

                "edge_ratio_gate": 0.0,
                "mean_grad_gate": 0.0,

                "translation_step_candidates": online_translation_step_candidates_cfg,
                "rotation_step_candidates_deg": online_rotation_step_candidates_deg_cfg,
            }],
        ),

        # 5) Atomic evaluators
        Node(
            package="calib_eval",
            executable="reprojection_evaluator_node",
            name="reprojection_evaluator_node",
            output="screen",
            parameters=[{
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "image_topic": "/eval/clean/image",

                # 2D pipeline evaluators consume raw scan.
                "lidar_topic": "/eval/clean/scan",
                "extrinsics_topic": "/eval/estimated_extrinsics",
                "evaluation_config_path": evaluation_yaml,
            }],
        ),

        Node(
            package="calib_eval",
            executable="edge_alignment_evaluator_node",
            name="edge_alignment_evaluator_node",
            output="screen",
            parameters=[{
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "image_topic": "/eval/clean/image",
                "lidar_topic": "/eval/clean/scan",
                "extrinsics_topic": "/eval/estimated_extrinsics",
                "evaluation_config_path": evaluation_yaml,
            }],
        ),

        Node(
            package="calib_eval",
            executable="baseline_deviation_evaluator_node",
            name="baseline_deviation_evaluator_node",
            output="screen",
            parameters=[{
                "estimated_extrinsics_topic": "/eval/estimated_extrinsics",
                "reference_extrinsics_topic": "/eval/ref_extrinsics",
            }],
        ),

        # 6) Aggregator
        Node(
            package="calib_eval",
            executable="evaluation_pipeline_node",
            name="evaluation_pipeline_node",
            output="screen",
            parameters=[{
                "evaluation_config_path": evaluation_yaml,
                "output_path_override": "~/evaluation_results/dynamic/",
            }],
        ),
    ])
