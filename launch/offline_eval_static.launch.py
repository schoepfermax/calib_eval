"""
Offline launch (STATIC 3D rig).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

pkg_share = get_package_share_directory("calib_eval")
reference_yaml = os.path.join(pkg_share, "config", "reference", "static_mount_h1_reference.yaml")
evaluation_yaml = os.path.join(pkg_share, "config", "evaluation.yaml")


def generate_launch_description():
    # --- Adjust these per run/model ---
    default_dataset_root = os.environ.get("CALIB_EVAL_DATASET_ROOT", "")

    dataset_root_cfg = LaunchConfiguration("dataset_root")
    split_cfg = LaunchConfiguration("split")
    rig_config_id_cfg = LaunchConfiguration("rig_config_id")
    use_dynamic_rig_cfg = LaunchConfiguration("use_dynamic_rig")
    dynamic_representation_cfg = LaunchConfiguration("dynamic_representation")
    model_node_executable_cfg = LaunchConfiguration("model_node_executable")

    # Topics (parametrized to support benchmarking + multi-env without code edits)
    image_topic_cfg = LaunchConfiguration("image_topic")
    lidar_topic_cfg = LaunchConfiguration("lidar_topic")
    scan_topic_cfg = LaunchConfiguration("scan_topic")
    camera_info_topic_cfg = LaunchConfiguration("camera_info_topic")
    estimated_extrinsics_topic_cfg = LaunchConfiguration("estimated_extrinsics_topic")
    reference_extrinsics_topic_cfg = LaunchConfiguration("reference_extrinsics_topic")
    save_final_transform_output_path_cfg = LaunchConfiguration("save_final_transform_output_path")

    # Publish rates (parametrized for quick sanity tests vs slower full runs)
    dataset_publish_rate_hz_cfg = LaunchConfiguration("dataset_publish_rate_hz")
    reference_publish_rate_hz_cfg = LaunchConfiguration("reference_publish_rate_hz")

    # Checkpoint_path passed at launch-time so model node does not crash before ros2 param set.
    checkpoint_path_cfg = LaunchConfiguration("checkpoint_path")
    device_cfg = LaunchConfiguration("device")

    # LCCNet runtime mode forwarding
    lccnet_mode_cfg = LaunchConfiguration("lccnet_mode")
    lccnet_iterative_steps_cfg = LaunchConfiguration("lccnet_iterative_steps")

    # Evaluation_config_path passed at launch-time to all relevant nodes.
    evaluation_config_path_cfg = LaunchConfiguration("evaluation_config_path")

    # Optional override (keep empty by default; node resolves using rig_config_id)
    reference_yaml_path_cfg = LaunchConfiguration("reference_yaml_path")

    # Reference extrinsics
    # NOTE:
    # ReferenceExtrinsicsPublisher resolves reference YAML from rig_config_id + package share by default.

    # Choose which wrapper to start:
    #   supervised_model_node.py  (LCCNet)
    #   calib_model_node.py       (BEVCalib)

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
            default_value="static_mount_h1",
            description="Rig config ID used to select reference YAML (e.g., static_mount_h1/static_mount_h2/dynamic)."
        ),
        DeclareLaunchArgument(
            "use_dynamic_rig",
            default_value="false",
            description="If true, pipeline expects dynamic rig dataset conventions."
        ),
        DeclareLaunchArgument(
            "dynamic_representation",
            default_value="pseudo_points",
            description="Dynamic rig representation for the data_preprocessor_node: raw_scan or pseudo_points."
        ),
        DeclareLaunchArgument(
            "model_node_executable",
            default_value="supervised_model_node",
            description="Model wrapper executable: supervised_model_node (LCCNet/CalibNet later) or calib_model_node (BEVCalib)."
        ),

        DeclareLaunchArgument(
            "image_topic",
            default_value="/eval/clean/image",
            description="Image topic published by the data_preprocessor_node."
        ),
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/eval/clean/points",
            description="PointCloud2 topic published by the data_preprocessor_node."
        ),
        DeclareLaunchArgument(
            "scan_topic",
            default_value="/eval/clean/scan",
            description="LaserScan topic published by the data_preprocessor_node (if used)."
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/eval/camera_info",
            description="CameraInfo topic published by the data_preprocessor_node."
        ),
        DeclareLaunchArgument(
            "estimated_extrinsics_topic",
            default_value="/eval/estimated_extrinsics",
            description="Extrinsics output topic from the model node."
        ),
        DeclareLaunchArgument(
            "reference_extrinsics_topic",
            default_value="/eval/ref_extrinsics",
            description="Reference extrinsics topic published by reference_extrinsics_publisher."
        ),

        DeclareLaunchArgument(
            "dataset_publish_rate_hz",
            default_value="2.0",
            description="Offline dataset playback rate (Hz)."
        ),
        DeclareLaunchArgument(
            "reference_publish_rate_hz",
            default_value="5.0",
            description="Reference extrinsics publish rate (Hz)."
        ),

        DeclareLaunchArgument(
            "checkpoint_path",
            default_value="",
            description="Optional. Model checkpoint path (e.g., LCCNet kitti_iterX.tar). If empty, node uses its own default."
        ),

        DeclareLaunchArgument(
            "device",
            default_value="cpu",
            description="Inference device for the model node: cpu or cuda."
        ),

        DeclareLaunchArgument(
            "lccnet_mode",
            default_value="single_pass",
            description="LCCNet runtime mode: single_pass or iterative."
        ),

        DeclareLaunchArgument(
            "lccnet_iterative_steps",
            default_value="5",
            description="Number of iterative LCCNet refinement stages to run when lccnet_mode=iterative."
        ),

        DeclareLaunchArgument(
            "evaluation_config_path",
            default_value=evaluation_yaml,
            description="Evaluation YAML used by evaluators/pipeline and reference publisher."
        ),

        DeclareLaunchArgument(
            "reference_yaml_path",
            default_value="",
            description="Optional. Override reference YAML path. If empty, node resolves via rig_config_id."
        ),

        DeclareLaunchArgument(
            "save_final_transform_output_path",
            default_value="/home/hs-coburg.de/rav4243s/evaluation_results/estimated_extrinsics.yaml",
            description="Rolling YAML path that is continuously updated with the latest /eval/estimated_extrinsics during the run."
        ),

        # 1) Dataset playback
        Node(
            package="calib_eval",
            executable="data_preprocessor_node",
            name="data_preprocessor_node",
            output="screen",
            parameters=[{
                "mode": "offline",
                "dataset_root": dataset_root_cfg,
                "split": split_cfg,
                "publish_rate_hz": dataset_publish_rate_hz_cfg,
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "dynamic_representation": dynamic_representation_cfg,
                "camera_frame_id": "camera",
                "out_image_topic": image_topic_cfg,
                "out_points_topic": lidar_topic_cfg,
                "out_scan_topic": scan_topic_cfg,
                "out_camera_info_topic": camera_info_topic_cfg,
            }],
        ),

        # 2) Publish reference/baseline extrinsics
        Node(
            package="calib_eval",
            executable="reference_extrinsics_publisher",
            name="reference_extrinsics_publisher",
            output="screen",
            parameters=[{
                # Keep empty by default; node will resolve from rig_config_id + package share.
                "reference_yaml_path": reference_yaml_path_cfg,
                "evaluation_config_path": evaluation_config_path_cfg,
                "rig_config_id": rig_config_id_cfg,
                "publish_rate_hz": reference_publish_rate_hz_cfg,
            }],
        ),

        # 3) Model wrapper node
        Node(
            package="calib_eval",
            executable=model_node_executable_cfg,
            name="model_node",
            output="screen",
            parameters=[{
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "image_topic": image_topic_cfg,
                "lidar_topic": lidar_topic_cfg,
                "camera_info_topic": camera_info_topic_cfg,
                "estimated_extrinsics_topic": estimated_extrinsics_topic_cfg,
                "reference_yaml_path": reference_yaml_path_cfg,
                "rig_config_id": rig_config_id_cfg,
                "checkpoint_path": checkpoint_path_cfg,
                "device": device_cfg,
                "lccnet_mode": lccnet_mode_cfg,
                "lccnet_iterative_steps": lccnet_iterative_steps_cfg,
            }],
        ),

        # 4) Atomic evaluators
        Node(
            package="calib_eval",
            executable="reprojection_evaluator_node",
            name="reprojection_evaluator_node",
            output="screen",
            parameters=[{
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "image_topic": image_topic_cfg,
                "lidar_topic": lidar_topic_cfg,
                "extrinsics_topic": estimated_extrinsics_topic_cfg,
                "evaluation_config_path": evaluation_config_path_cfg,
                "camera_info_topic": camera_info_topic_cfg,
            }],
        ),

        Node(
            package="calib_eval",
            executable="edge_alignment_evaluator_node",
            name="edge_alignment_evaluator_node",
            output="screen",
            parameters=[{
                "use_dynamic_rig": use_dynamic_rig_cfg,
                "image_topic": image_topic_cfg,
                "lidar_topic": lidar_topic_cfg,
                "extrinsics_topic": estimated_extrinsics_topic_cfg,
                "evaluation_config_path": evaluation_config_path_cfg,
                "camera_info_topic": camera_info_topic_cfg,
            }],
        ),

        Node(
            package="calib_eval",
            executable="baseline_deviation_evaluator_node",
            name="baseline_deviation_evaluator_node",
            output="screen",
            parameters=[{
                "estimated_extrinsics_topic": estimated_extrinsics_topic_cfg,
                "reference_extrinsics_topic": reference_extrinsics_topic_cfg,
            }],
        ),

        # 5) Rolling final-transform saver
        Node(
            package="calib_eval",
            executable="save_final_transform",
            name="save_final_transform",
            output="screen",
            parameters=[{
                "extrinsics_topic": estimated_extrinsics_topic_cfg,
                "output_path": save_final_transform_output_path_cfg,
            }],
        ),

        # 6) System-level aggregator
        Node(
            package="calib_eval",
            executable="evaluation_pipeline_node",
            name="evaluation_pipeline_node",
            output="screen",
            parameters=[{
                "evaluation_config_path": evaluation_config_path_cfg,
            }],
        ),
    ])