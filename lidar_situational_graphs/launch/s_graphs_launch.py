import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="platform/velodyne_points",
                description="Name of the lidar topic to sub",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="platform/odometry",
                description="Name of the odom topic to sub",
            ),
            DeclareLaunchArgument(
                "imu_topic",
                default_value="imu/data",
                description="Name of the imu topic to sub",
            ),
            DeclareLaunchArgument(
                "base_frame",
                default_value="base_link",
                description="Name of the base frame",
            ),
            DeclareLaunchArgument(
                "odom_frame",
                default_value="odom",
                description="Name of the base frame",
            ),
            DeclareLaunchArgument(
                "map_frame",
                default_value="map",
                description="Name of the base frame",
            ),
            DeclareLaunchArgument(
                "compute_odom",
                default_value="true",
                description="Flag to compute the odometry",
            ),
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description="Namespace for the robot",
            ),
            DeclareLaunchArgument(
                "room_segmentation",
                default_value="new",
                description="Algorithm used for room segmentation",
            ),
            DeclareLaunchArgument(
                "keyframe_delta",
                default_value="2.0",
                description="Value in [m] for recording each keyframe",
            ),
            DeclareLaunchArgument(
                "viz_dense_map",
                default_value="false",
                description="If visualize dense map in rviz",
            ),
            DeclareLaunchArgument(
                "debug_mode",
                default_value="false",
                description="Run s_graphs node in debugging mode",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation time from /clock topic (required for bag playback)",
            ),
            DeclareLaunchArgument(
                "enable_yolo",
                default_value="true",
                description="Enable YOLO object detection",
            ),
            DeclareLaunchArgument(
                "enable_clip",
                default_value="true",
                description="Enable CLIP feature extraction",
            ),
            DeclareLaunchArgument(
                "visualize_detections",
                default_value="true",
                description="Enable visualization of detections",
            ),
            DeclareLaunchArgument(
                "confidence_threshold",
                default_value="0.25",
                description="YOLO confidence threshold",
            ),
            DeclareLaunchArgument(
                "process_every_n_frames",
                default_value="5",
                description="Process semantic info every N frames",
            ),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value="/workspace/src/s_graphs/models/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.trt",
                description="Path to YOLO model",
            ),
            DeclareLaunchArgument(
                "clip_model_path",
                default_value="/workspace/src/s_graphs/models/visual.onnx",
                description="Path to CLIP model",
            ),
            DeclareLaunchArgument(
                "enable_dynatrack",
                default_value="true",
                description="Enable DynaTrack (optical flow dynamic object detection)",
            ),
            OpaqueFunction(function=launch_sgraphs),
        ]
    )


def launch_reasoning():
    reasoning_dir = get_package_share_directory("situational_graphs_reasoning")
    reasoning_launch_file = os.path.join(
        reasoning_dir, "launch", "situational_graphs_reasoning.launch.py"
    )
    reasoning_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(reasoning_launch_file)
    )

    return reasoning_launch


def launch_sgraphs(context, *args, **kwargs):
    pkg_dir = get_package_share_directory("lidar_situational_graphs")
    # For finding the installed Python script, we need the package's lib directory
    # The share directory is usually at: /workspace/install/lidar_situational_graphs/share/lidar_situational_graphs
    # The lib directory is at: /workspace/install/lidar_situational_graphs/lib/lidar_situational_graphs
    install_dir = pkg_dir.replace("/share/lidar_situational_graphs", "")
    iqa_script_path = os.path.join(
        install_dir, "lib", "lidar_situational_graphs", "iqa_ros.py")

    prefiltering_param_file = os.path.join(
        pkg_dir, "config", "prefiltering.yaml")
    scan_matching_param_file = os.path.join(
        pkg_dir, "config", "scan_matching.yaml")
    s_graphs_param_file = os.path.join(pkg_dir, "config", "s_graphs.yaml")

    lidar_topic_arg = LaunchConfiguration("lidar_topic").perform(context)
    odom_topic_arg = LaunchConfiguration("odom_topic").perform(context)
    imu_topic_arg = LaunchConfiguration("imu_topic").perform(context)

    base_frame_arg = LaunchConfiguration("base_frame").perform(context)
    odom_frame_arg = LaunchConfiguration("odom_frame").perform(context)
    map_frame_arg = LaunchConfiguration("map_frame").perform(context)

    compute_odom_arg = LaunchConfiguration("compute_odom").perform(context)
    namespace_arg = LaunchConfiguration("namespace").perform(context)
    room_segmentation_arg = LaunchConfiguration(
        "room_segmentation").perform(context)
    keyframe_delta_arg = float(LaunchConfiguration(
        "keyframe_delta").perform(context))
    viz_dense_map_arg = LaunchConfiguration("viz_dense_map").perform(context)
    debug_mode_arg = LaunchConfiguration("debug_mode").perform(context)
    use_sim_time_arg = LaunchConfiguration("use_sim_time").perform(context)
    enable_yolo_arg = LaunchConfiguration("enable_yolo").perform(context)
    enable_clip_arg = LaunchConfiguration("enable_clip").perform(context)
    visualize_detections_arg = LaunchConfiguration(
        "visualize_detections").perform(context)
    confidence_threshold_arg = float(LaunchConfiguration(
        "confidence_threshold").perform(context))
    process_every_n_frames_arg = int(LaunchConfiguration(
        "process_every_n_frames").perform(context))
    yolo_model_path_arg = LaunchConfiguration(
        "yolo_model_path").perform(context)
    clip_model_path_arg = LaunchConfiguration(
        "clip_model_path").perform(context)

    ns_prefix = str(namespace_arg) + "/" if namespace_arg else ""
    if str(ns_prefix).startswith("/"):
        ns_prefix = ns_prefix[1:]

    base_link_frame = base_frame_arg
    odom_frame = odom_frame_arg
    map_frame = map_frame_arg

    prefiltering_cmd = Node(
        package="lidar_situational_graphs",
        executable="s_graphs_prefiltering_node",
        namespace=namespace_arg,
        parameters=[{prefiltering_param_file}, {
            "base_link_frame": base_link_frame, "use_sim_time": use_sim_time_arg == "true"}],
        output="screen",
        remappings=[
            ("velodyne_points", lidar_topic_arg),
            ("imu/data", imu_topic_arg),
        ],
    )

    scan_matching_cmd = Node(
        package="lidar_situational_graphs",
        executable="s_graphs_scan_matching_odometry_node",
        namespace=namespace_arg,
        parameters=[{scan_matching_param_file}, {
            "use_sim_time": use_sim_time_arg == "true"}],
        remappings=[("odom", odom_topic_arg)],
        output="screen",
        condition=IfCondition(compute_odom_arg),
    )

    if room_segmentation_arg == "old":
        room_segmentation_cmd = Node(
            package="lidar_situational_graphs",
            executable="s_graphs_room_segmentation_node",
            namespace=namespace_arg,
            parameters=[{"vertex_neigh_thres": 2}],
            output="screen",
        )
    else:
        reasoning_launch = launch_reasoning()

    floor_plan_cmd = Node(
        package="lidar_situational_graphs",
        executable="s_graphs_floor_plan_node",
        namespace=namespace_arg,
        parameters=[
            {
                "vertex_neigh_thres": 2,
                "keyframe_delta_trans": keyframe_delta_arg,
                "use_sim_time": use_sim_time_arg == "true",
            }
        ],
        output="screen",
    )

    s_graphs_cmd = Node(
        package="lidar_situational_graphs",
        executable="s_graphs_node",
        namespace=namespace_arg,
        parameters=[
            {s_graphs_param_file},
            {
                "odom_frame_id": odom_frame,
                "map_frame_id": map_frame,
                "keyframe_delta_trans": keyframe_delta_arg,
                "keyframe_delta_angle": keyframe_delta_arg,
                "viz_dense_map": viz_dense_map_arg,
                "use_sim_time": use_sim_time_arg == "true",
                "enable_yolo": enable_yolo_arg == "true",
                "enable_clip": enable_clip_arg == "true",
                "visualize_detections": visualize_detections_arg == "true",
                "confidence_threshold": confidence_threshold_arg,
                "process_every_n_frames": process_every_n_frames_arg,
                "yolo_model_path": yolo_model_path_arg,
                "clip_model_path": clip_model_path_arg,
            },
        ],
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        prefix=["gdbserver localhost:3000"] if debug_mode_arg == "true" else None,
        remappings=[
            ("odom", odom_topic_arg),
        ],
    )

    iqa_cmd = ExecuteProcess(
        cmd=["python3", iqa_script_path],
        output="screen",
        shell=False,
    )

    # DynaTrack: BEV optical flow dynamic object detection
    dynatrack_cmd = Node(
        package="tracking_of",
        executable="of_track_node",
        namespace=namespace_arg,
        parameters=[{"use_sim_time": use_sim_time_arg == "true"}],
        remappings=[
            ("/velodyne_points", ns_prefix + "filtered_points"),
            ("/odom", odom_topic_arg),
        ],
        output="screen",
    )

    # Zone-based semantic prefiltering node
    zones_script_path = os.path.join(
        install_dir, "lib", "lidar_situational_graphs", "zones_node.py")
    zones_cmd = ExecuteProcess(
        cmd=["python3", zones_script_path],
        output="screen",
        shell=False,
    )

    # Metrics recorder — writes CSV files to /tmp/sgraphs_metrics
    metrics_script_path = os.path.join(
        install_dir, "lib", "lidar_situational_graphs", "metrics_recorder.py")
    metrics_cmd = ExecuteProcess(
        cmd=["python3", metrics_script_path,
             "--ros-args", "-p", "output_dir:=/tmp/sgraphs_metrics"],
        output="screen",
        shell=False,
    )

    return [
        prefiltering_cmd,
        scan_matching_cmd,
        room_segmentation_cmd if room_segmentation_arg == "old" else reasoning_launch,
        floor_plan_cmd,
        s_graphs_cmd,
        iqa_cmd,
        dynatrack_cmd,
        zones_cmd,
        metrics_cmd,
    ]
