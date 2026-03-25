from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Core mode / source selection
    input_mode_cfg = LaunchConfiguration("input_mode")

    # Archive reconstruction inputs
    archive_dir_cfg = LaunchConfiguration("archive_dir")
    dataset_root_cfg = LaunchConfiguration("dataset_root")
    run_name_cfg = LaunchConfiguration("run_name")
    sample_id_cfg = LaunchConfiguration("sample_id")
    reference_yaml_path_cfg = LaunchConfiguration("reference_yaml_path")
    device_cfg = LaunchConfiguration("device")

    # General pipeline-style switches
    use_dynamic_rig_cfg = LaunchConfiguration("use_dynamic_rig")
    dynamic_representation_cfg = LaunchConfiguration("dynamic_representation")

    # Live topic inputs
    image_topic_cfg = LaunchConfiguration("image_topic")
    points_topic_cfg = LaunchConfiguration("points_topic")
    camera_info_topic_cfg = LaunchConfiguration("camera_info_topic")
    reference_extrinsics_topic_cfg = LaunchConfiguration("reference_extrinsics_topic")
    estimated_extrinsics_topic_cfg = LaunchConfiguration("estimated_extrinsics_topic")

    # Display / output topics
    metadata_text_cfg = LaunchConfiguration("metadata_text")
    bottom_banner_height_px_cfg = LaunchConfiguration("bottom_banner_height_px")
    banner_background_bgr_cfg = LaunchConfiguration("banner_background_bgr")
    banner_text_bgr_cfg = LaunchConfiguration("banner_text_bgr")
    banner_font_scale_cfg = LaunchConfiguration("banner_font_scale")
    banner_text_thickness_cfg = LaunchConfiguration("banner_text_thickness")
    banner_outline_thickness_cfg = LaunchConfiguration("banner_outline_thickness")
    banner_side_margin_px_cfg = LaunchConfiguration("banner_side_margin_px")
    banner_top_bottom_margin_px_cfg = LaunchConfiguration("banner_top_bottom_margin_px")
    banner_line_gap_px_cfg = LaunchConfiguration("banner_line_gap_px")
    banner_force_uppercase_cfg = LaunchConfiguration("banner_force_uppercase")
    reference_overlay_topic_cfg = LaunchConfiguration("reference_overlay_topic")
    estimated_overlay_topic_cfg = LaunchConfiguration("estimated_overlay_topic")

    return LaunchDescription([
        ###############################################
        # Arguments
        ###############################################
        DeclareLaunchArgument(
            "input_mode",
            default_value="archive_reconstruct",
            description="Visualizer input mode: live | archive_reconstruct",
        ),

        # Archive reconstruction parameters
        DeclareLaunchArgument(
            "archive_dir",
            default_value="",
            description="Path to archived experiment folder containing run_info.txt",
        ),
        DeclareLaunchArgument(
            "dataset_root",
            default_value="",
            description="Optional dataset root override for workstation/laptop portability",
        ),
        DeclareLaunchArgument(
            "run_name",
            default_value="",
            description="Dataset run directory name, e.g. run_001",
        ),
        DeclareLaunchArgument(
            "sample_id",
            default_value="",
            description="Chosen sample id for deterministic figure reconstruction, e.g. 000123",
        ),
        DeclareLaunchArgument(
            "reference_yaml_path",
            default_value="",
            description="Optional absolute reference YAML override",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="auto",
            description="Inference device for archive reconstruction: auto | cpu | cuda",
        ),

        # Static/dynamic control
        DeclareLaunchArgument(
            "use_dynamic_rig",
            default_value="false",
            description="Whether the selected experiment belongs to the dynamic rig",
        ),
        DeclareLaunchArgument(
            "dynamic_representation",
            default_value="pseudo_points",
            description="Dynamic representation: pseudo_points | raw_scan",
        ),

        # Live topic mode defaults
        DeclareLaunchArgument(
            "image_topic",
            default_value="/eval/clean/image",
            description="Live input image topic",
        ),
        DeclareLaunchArgument(
            "points_topic",
            default_value="/eval/clean/points",
            description="Live input point cloud topic",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/eval/camera_info",
            description="Live input camera info topic",
        ),
        DeclareLaunchArgument(
            "reference_extrinsics_topic",
            default_value="/eval/ref_extrinsics",
            description="Live input reference extrinsics topic",
        ),
        DeclareLaunchArgument(
            "estimated_extrinsics_topic",
            default_value="/eval/estimated_extrinsics",
            description="Live input estimated extrinsics topic",
        ),

        # Visual output / metadata
        DeclareLaunchArgument(
            "metadata_text",
            default_value="",
            description="Optional explicit metadata banner text override",
        ),
        DeclareLaunchArgument(
            "bottom_banner_height_px",
            default_value="150",
            description="Bottom metadata banner height in pixels",
        ),
        DeclareLaunchArgument(
            "banner_background_bgr",
            default_value="[232, 232, 232]",
            description="Banner background color in BGR list form",
        ),
        DeclareLaunchArgument(
            "banner_text_bgr",
            default_value="[0, 0, 0]",
            description="Banner text color in BGR list form",
        ),
        DeclareLaunchArgument(
            "banner_font_scale",
            default_value="1.35",
            description="Banner font scale",
        ),
        DeclareLaunchArgument(
            "banner_text_thickness",
            default_value="4",
            description="Banner text thickness",
        ),
        DeclareLaunchArgument(
            "banner_outline_thickness",
            default_value="8",
            description="Banner outline thickness for readability",
        ),
        DeclareLaunchArgument(
            "banner_side_margin_px",
            default_value="28",
            description="Left/right banner margin in pixels",
        ),
        DeclareLaunchArgument(
            "banner_top_bottom_margin_px",
            default_value="18",
            description="Top/bottom banner text margin in pixels",
        ),
        DeclareLaunchArgument(
            "banner_line_gap_px",
            default_value="16",
            description="Gap in pixels between the two banner lines",
        ),
        DeclareLaunchArgument(
            "banner_force_uppercase",
            default_value="true",
            description="Render banner text in uppercase",
        ),
        DeclareLaunchArgument(
            "reference_overlay_topic",
            default_value="/viz/reference_overlay",
            description="Published image topic for reference projection view",
        ),
        DeclareLaunchArgument(
            "estimated_overlay_topic",
            default_value="/viz/estimated_overlay",
            description="Published image topic for estimated projection view",
        ),

        ###############################################
        # Visualizer node
        ###############################################


        ###############################################
        # Workstation image viewers (optional)
        ###############################################
        ExecuteProcess(
            cmd=["ros2", "run", "rqt_image_view", "rqt_image_view", "/viz/reference_overlay"],
            output="screen",
        ),
        ExecuteProcess(
            cmd=["ros2", "run", "rqt_image_view", "rqt_image_view", "/viz/estimated_overlay"],
            output="screen",
        ),

        Node(
            package="calib_eval",
            executable="visualizer",
            name="visualizer",
            output="screen",
            parameters=[{
                "input_mode": input_mode_cfg,

                "archive_dir": archive_dir_cfg,
                "dataset_root": dataset_root_cfg,
                "run_name": run_name_cfg,
                "sample_id": sample_id_cfg,
                "reference_yaml_path": reference_yaml_path_cfg,
                "device": device_cfg,

                "use_dynamic_rig": use_dynamic_rig_cfg,
                "dynamic_representation": dynamic_representation_cfg,

                "image_topic": image_topic_cfg,
                "points_topic": points_topic_cfg,
                "camera_info_topic": camera_info_topic_cfg,
                "reference_extrinsics_topic": reference_extrinsics_topic_cfg,
                "estimated_extrinsics_topic": estimated_extrinsics_topic_cfg,
                "point_stride": 2,

                "metadata_text": metadata_text_cfg,
                "bottom_banner_height_px": bottom_banner_height_px_cfg,
                "banner_background_bgr": banner_background_bgr_cfg,
                "banner_text_bgr": banner_text_bgr_cfg,
                "banner_font_scale": banner_font_scale_cfg,
                "banner_text_thickness": banner_text_thickness_cfg,
                "banner_outline_thickness": banner_outline_thickness_cfg,
                "banner_side_margin_px": banner_side_margin_px_cfg,
                "banner_top_bottom_margin_px": banner_top_bottom_margin_px_cfg,
                "banner_line_gap_px": banner_line_gap_px_cfg,
                "banner_force_uppercase": banner_force_uppercase_cfg,
                "reference_overlay_topic": reference_overlay_topic_cfg,
                "estimated_overlay_topic": estimated_overlay_topic_cfg,
            }],
        ),
    ])