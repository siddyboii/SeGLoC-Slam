#!/usr/bin/env python3
"""
sim_kf_publisher.py

Timer-based synthetic KeyframeSemantic publisher.

- Publishes at 5 Hz
- Uses odometry for pose
- Cycles semantic object sets over time to trigger zone creation
- NO dependency on room_segmentation/room_data
"""

import rclpy
from rclpy.node import Node
import time
from threading import Lock

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from lidar_situational_graphs.msg import KeyframeSemantic


class SimKFPublisher(Node):
    def __init__(self):
        super().__init__('sim_kf_publisher')

        # ===== PARAMETERS =====
        self.publish_hz = 3.0
        self.semantic_switch_period = 30.0  # seconds per semantic mode
        self.start_kf_id = 1000

        # ===== INTERNAL STATE =====
        self.last_odom_pose = None
        self.odom_lock = Lock()
        self.kf_id = self.start_kf_id
        self.start_time = time.time()

        # ===== SEMANTIC MODES =====
        # Each entry = (objects, confidences)
        self.semantic_modes = [
            (['bed', 'pillow'], [0.95, 0.85]),           # bedroom
            (['stove', 'fridge'], [0.92, 0.88]),         # kitchen
            (['sofa', 'tv'], [0.90, 0.80]),              # living room
            (['toilet', 'sink'], [0.93, 0.87])           # bathroom
        ]

        # ===== PUB / SUB =====
        self.kf_pub = self.create_publisher(
            KeyframeSemantic, 'keyframe_semantics', 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/platform/odometry', self.cb_odom, 20
        )

        # ===== TIMER =====
        self.timer = self.create_timer(
            1.0 / self.publish_hz, self.timer_cb
        )

        self.get_logger().info(
            f"SimKFPublisher started @ {self.publish_hz} Hz, "
            f"semantic switch every {self.semantic_switch_period}s"
        )

    # ------------------------------------------------------
    # Odometry callback (UNCHANGED)
    # ------------------------------------------------------
    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose
        with self.odom_lock:
            self.last_odom_pose = (
                float(p.position.x),
                float(p.position.y),
                float(p.position.z),
                float(p.orientation.x),
                float(p.orientation.y),
                float(p.orientation.z),
                float(p.orientation.w)
            )

    # ------------------------------------------------------
    # Timer callback → publish KeyframeSemantic
    # ------------------------------------------------------
    def timer_cb(self):
        now = time.time()
        elapsed = now - self.start_time

        # Determine semantic mode
        mode_idx = int(elapsed // self.semantic_switch_period) % len(self.semantic_modes)
        objects, confidences = self.semantic_modes[mode_idx]

        msg = KeyframeSemantic()

        # ----- Header -----
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # ----- IDs (keep commented as requested) -----
        # msg.keyframe_id = self.kf_id
        # msg.keyframe_seq = self.kf_id
        msg.stamp = self.get_clock().now().to_msg()
        msg.floor_id = 0
        msg.room_id = -1

        # ----- Semantics -----
        msg.objects = list(objects)
        msg.object_confidence = list(confidences)
        msg.scene_confidence = float(
            sum(confidences) / max(len(confidences), 1)
        )
        msg.model_name = f"sim_mode_{mode_idx}"

        # ----- Pose -----
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = 'map'

        with self.odom_lock:
            od = self.last_odom_pose

        if od is not None:
            pose.pose.position.x = od[0]
            pose.pose.position.y = od[1]
            pose.pose.position.z = od[2]
            pose.pose.orientation.x = od[3]
            pose.pose.orientation.y = od[4]
            pose.pose.orientation.z = od[5]
            pose.pose.orientation.w = od[6]
        else:
            pose.pose.orientation.w = 1.0

        try:
            msg.pose = pose
        except Exception:
            pass

        # ----- Publish -----
        self.kf_pub.publish(msg)
        self.get_logger().info(
            f"[KF {self.kf_id}] Published objects={objects} mode={mode_idx}"
        )

        self.kf_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = SimKFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

