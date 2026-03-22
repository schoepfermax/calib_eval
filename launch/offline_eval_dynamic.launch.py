"""
Offline evaluation launch (DYNAMIC rig).
Offline pipeline:
  DatasetPlayer -> ReferenceExtrinsicsPublisher -> Model Node -> Evaluators -> EvaluationPipeline
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
    # Environment-specific dataset root (laptop vs workstation vs HPC)
    default_dataset_root = os.environ.get("CALIB_EVAL_DATASET_ROOT", "")

    dataset_root_cfg = LaunchConfiguration("dataset_root")
    split_cfg = LaunchConfiguration("split")
    rig_config_id_cfg = LaunchConfiguration("rig_config_id")
    use_dynamic_rig_cfg = LaunchConfiguration("use_dynamic_rig")

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

                # Lighter sanity/integration settings for the laptop-side 2D path.
                "max_frames": 40,
                "top_k": 20,

                # Keep the search explicit and thesis-defensible, but cheaper.
                "stage_a_translation_steps_x_m": [-0.05, 0.0, 0.05],
                "stage_a_translation_steps_y_m": [-0.05, 0.0, 0.05],
                "stage_a_translation_steps_z_m": [0.0],
                "stage_a_rotation_steps_roll_deg": [0.0],
                "stage_a_rotation_steps_pitch_deg": [0.0],
                "stage_a_rotation_steps_yaw_deg": [-2.0, 0.0, 2.0],

                "stage_b_translation_steps_x_m": [-0.01, 0.0, 0.01],
                "stage_b_translation_steps_y_m": [-0.01, 0.0, 0.01],
                "stage_b_translation_steps_z_m": [0.0],
                "stage_b_rotation_steps_roll_deg": [0.0],
                "stage_b_rotation_steps_pitch_deg": [0.0],
                "stage_b_rotation_steps_yaw_deg": [-0.5, 0.0, 0.5],
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
                "window_size": 20,
                "min_translation_for_update_m": 0.02,
                "min_yaw_for_update_deg": 2.0,
                "max_translation_step_m": 0.05,
                "max_rotation_step_deg": 2.0,
                "deterioration_patience": 5,
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
