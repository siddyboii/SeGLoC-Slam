"""
═══════════════════════════════════════════════════════════════════
  HILTI Dataset Launch — Static TF + S-Graphs Pipeline
═══════════════════════════════════════════════════════════════════
  Publishes the required static transforms for the HILTI-SLAM
  handheld dataset and then launches the full s_graphs pipeline.

  Sensor frames in the HILTI bag:
    LiDAR  →  PandarXT-32         topic: /hesai/pandar
    IMU    →  imu_sensor_frame    topic: /alphasense/imu
    Cam0   →  cam0_sensor_frame   topic: /alphasense/cam0/image_raw

  The HILTI handheld unit has the LiDAR roughly centered on the
  device with the IMU and cameras on the Alphasense unit. The
  approximate calibration values below are from the HILTI-SLAM
  challenge documentation.

  Usage:
    ros2 launch lidar_situational_graphs s_graphs_campus_launch.py
    # Then in another terminal:
    ros2 bag play campus/ --clock
═══════════════════════════════════════════════════════════════════
"""

import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("lidar_situational_graphs")
    main_launch = os.path.join(pkg_dir, "launch", "s_graphs_launch.py")

    # ── Static TF: base_link → PandarXT-32 (LiDAR) ──
    # The Hesai PandarXT-32 is mounted at the top of the HILTI handheld unit.
    # Approximate transform from base_link (center of device) to LiDAR.
    # x=0, y=0, z=0.05m (slightly above), no rotation (both look forward).
    # Adjust these values if you have the exact HILTI calibration file.
    lidar_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", "os_sensor",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # ── Static TF: base_link → imu_sensor_frame (IMU) ──
    # The Alphasense IMU is co-located with the LiDAR on the handheld unit.
    # Approximately identity transform (adjust if you have exact calibration).
    imu_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", "imu_link_ned",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # ── Static TF: base_link → cam0_sensor_frame (Camera 0) ──
    # Camera 0 on the Alphasense unit. The cameras are arranged around
    # the unit. cam0 is typically the forward-looking camera.
    # Approximate values — adjust based on actual HILTI calibration.
    cam0_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", "camera_left/optical_frame",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # ── Include the main s_graphs launch with HILTI-specific topic remaps ──
    s_graphs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(main_launch),
        launch_arguments={
            "lidar_topic": "/ouster/points",
            "imu_topic": "/imu/data",
            "base_frame": "base_link",
            "odom_frame": "odom",
            "map_frame": "map",
            "compute_odom": "true",
            "use_sim_time": "true",
            "keyframe_delta": "2.0",
        }.items(),
    )

    return LaunchDescription(
        [
            # Static transforms first
            lidar_tf,
            imu_tf,
            cam0_tf,
            # Then the full pipeline
            s_graphs_launch,
        ]
    )
