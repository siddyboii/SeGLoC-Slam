#!/usr/bin/env python3
"""
TF Filter Node - Filters out specific transforms from /tf topic.

This node subscribes to /tf_raw (remapped from bag's /tf) and republishes
to /tf, filtering out unwanted transforms like map->odom that might
conflict with S-Graphs SLAM.

Usage:
    When playing bag, remap /tf to /tf_raw:
    ros2 bag play <bag> --remap /tf:=/tf_raw

    Then run this filter:
    ros2 run lidar_situational_graphs tf_filter --ros-args -p use_sim_time:=true
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class TFFilterNode(Node):
    def __init__(self):
        super().__init__('tf_filter_node')

        # Parameters for frames to filter out
        self.declare_parameter('filtered_parent_frames', ['map'])
        self.declare_parameter('filtered_child_frames', ['/odom', 'odom'])

        self.filtered_parents = self.get_parameter(
            'filtered_parent_frames').value
        self.filtered_children = self.get_parameter(
            'filtered_child_frames').value

        self.get_logger().info(
            f'Filtering transforms from parents: {self.filtered_parents}')
        self.get_logger().info(
            f'Filtering transforms to children: {self.filtered_children}')

        # Subscribe to raw TF (from bag, remapped)
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf_raw',
            self.tf_callback,
            100
        )

        # Publish filtered TF
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 100)

        self.filtered_count = 0
        self.passed_count = 0

    def tf_callback(self, msg):
        filtered_msg = TFMessage()

        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id

            # Check if this transform should be filtered
            should_filter = False
            for fp in self.filtered_parents:
                for fc in self.filtered_children:
                    if parent == fp and (child == fc or child.lstrip('/') == fc.lstrip('/')):
                        should_filter = True
                        self.filtered_count += 1
                        if self.filtered_count % 100 == 1:
                            self.get_logger().info(
                                f'Filtered {self.filtered_count} transforms: {parent} -> {child}'
                            )
                        break
                if should_filter:
                    break

            if not should_filter:
                filtered_msg.transforms.append(transform)
                self.passed_count += 1

        # Publish if there are any transforms left
        if filtered_msg.transforms:
            self.tf_pub.publish(filtered_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TFFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'TF Filter stats: Filtered={node.filtered_count}, Passed={node.passed_count}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
