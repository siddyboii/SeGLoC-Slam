"""
Semantic Test Harness Launch File

Purpose: Launches the semantic_test_node for testing YOLO and CLIP integration
with camera input. Allows configuration of model paths and processing parameters.

Usage:
  ros2 launch lidar_situational_graphs semantic_test_launch.py

With custom parameters:
  ros2 launch lidar_situational_graphs semantic_test_launch.py \
      yolo_model_path:=/path/to/yolo.engine \
      clip_model_path:=/path/to/clip.onnx \
      image_topic:=/camera/color/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    # Rationale: Launch arguments allow runtime configuration without modifying code

    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value='',
        description='Path to YOLO TensorRT engine file (.engine)'
    )

    clip_model_path_arg = DeclareLaunchArgument(
        'clip_model_path',
        default_value='',
        description='Path to CLIP ONNX model file (.onnx)'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='Input image topic name'
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.25',
        description='YOLO detection confidence threshold (0.0-1.0)'
    )

    process_every_n_frames_arg = DeclareLaunchArgument(
        'process_every_n_frames',
        default_value='5',
        description='Process every N frames (for real-time performance)'
    )

    enable_yolo_arg = DeclareLaunchArgument(
        'enable_yolo',
        default_value='true',
        description='Enable YOLO object detection'
    )

    enable_clip_arg = DeclareLaunchArgument(
        'enable_clip',
        default_value='true',
        description='Enable CLIP feature extraction'
    )

    visualize_detections_arg = DeclareLaunchArgument(
        'visualize_detections',
        default_value='true',
        description='Publish visualization images with bounding boxes'
    )

    # Create the semantic test node
    semantic_test_node = Node(
        package='lidar_situational_graphs',
        executable='semantic_test_node',
        name='semantic_test_node',
        output='screen',
        emulate_tty=True,  # Enables colored output
        parameters=[{
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
            'clip_model_path': LaunchConfiguration('clip_model_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'process_every_n_frames': LaunchConfiguration('process_every_n_frames'),
            'enable_yolo': LaunchConfiguration('enable_yolo'),
            'enable_clip': LaunchConfiguration('enable_clip'),
            'visualize_detections': LaunchConfiguration('visualize_detections'),
        }]
    )

    return LaunchDescription([
        yolo_model_path_arg,
        clip_model_path_arg,
        image_topic_arg,
        confidence_threshold_arg,
        process_every_n_frames_arg,
        enable_yolo_arg,
        enable_clip_arg,
        visualize_detections_arg,
        semantic_test_node,
    ])
