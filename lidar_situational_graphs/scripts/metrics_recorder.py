#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  SEMANTIC LOOP CLOSURE — REAL-TIME METRICS RECORDER
═══════════════════════════════════════════════════════════════════════════
  Subscribes to live ROS2 topics during a bag playback and records every
  measurable quantity to CSV files inside an output directory.

  Recorded data
  ─────────────
  1. corrected_trajectory.csv   — graph-optimised poses  (from s_graphs)
  2. raw_odometry.csv           — raw scan-matching odom  (from odom topic)
  3. keyframe_semantics.csv     — per-keyframe CLIP, YOLO, IQA, dynamicity
  4. dynamic_objects.csv        — per-frame DynaTrack output
  5. zone_events.csv            — zone create / update / delete events
  6. image_quality.csv          — IQA scores over time

  Usage
  ─────
  ros2 run lidar_situational_graphs metrics_recorder.py \
      --ros-args -p output_dir:=/tmp/sgraphs_metrics
═══════════════════════════════════════════════════════════════════════════
"""

import os
import csv
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64

# Custom message imports — adjust package name if different
try:
    from situational_graphs_msgs.msg import KeyframeSemantic, Zone, DynamicObjects
    HAS_CUSTOM_MSGS = True
except ImportError:
    HAS_CUSTOM_MSGS = False


def pose_to_row(stamp_sec, stamp_nsec, pose):
    """Extract [t, x, y, z, qx, qy, qz, qw] from a Pose."""
    p = pose.position
    q = pose.orientation
    return [
        stamp_sec, stamp_nsec,
        p.x, p.y, p.z,
        q.x, q.y, q.z, q.w,
    ]


class MetricsRecorder(Node):
    def __init__(self):
        super().__init__("metrics_recorder")

        # ---------- parameters ----------
        self.declare_parameter("output_dir", "/tmp/sgraphs_metrics")
        self.output_dir = self.get_parameter("output_dir").value
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(
            f"[MetricsRecorder] Saving to: {self.output_dir}")

        # ---------- CSV writers ----------
        self._files = {}
        self._writers = {}

        self._open_csv("corrected_trajectory", [
            "stamp_sec", "stamp_nsec", "x", "y", "z", "qx", "qy", "qz", "qw"
        ])
        self._open_csv("raw_odometry", [
            "stamp_sec", "stamp_nsec", "x", "y", "z", "qx", "qy", "qz", "qw"
        ])
        self._open_csv("keyframe_semantics", [
            "stamp_sec", "stamp_nsec", "keyframe_id",
            "clip_dim", "num_objects", "object_labels",
            "scene_dynamicity", "num_dynamic_clusters",
            "image_quality",
            "x", "y", "z", "qx", "qy", "qz", "qw",
        ])
        self._open_csv("dynamic_objects", [
            "stamp_sec", "stamp_nsec",
            "scene_dynamicity", "num_clusters", "cluster_sizes",
        ])
        self._open_csv("zone_events", [
            "stamp_sec", "stamp_nsec",
            "zone_id", "action", "floor_id",
            "num_keyframes", "num_rooms",
            "top_labels", "confidence", "version",
        ])
        self._open_csv("image_quality", [
            "stamp_sec", "stamp_nsec", "score",
        ])

        # ---------- counters ----------
        self.n_corrected = 0
        self.n_odom = 0
        self.n_kf_sem = 0
        self.n_dyn = 0
        self.n_zone = 0
        self.n_iqa = 0
        self.start_time = time.time()

        # ---------- QoS for transient_local topics ----------
        tl_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ---------- subscribers ----------
        # 1. Corrected trajectory (PoseStamped)
        self.create_subscription(
            PoseStamped,
            "s_graphs/odom_pose_corrected",
            self._cb_corrected_pose,
            10,
        )

        # 2. Raw odometry (nav_msgs/Odometry)
        self.create_subscription(
            Odometry,
            "scan_matching_odometry/odom",
            self._cb_raw_odom,
            best_effort_qos,
        )

        # 3. Image quality score
        self.create_subscription(
            Float64,
            "/image_quality_score",
            self._cb_iqa,
            10,
        )

        if HAS_CUSTOM_MSGS:
            # 4. KeyframeSemantic
            self.create_subscription(
                KeyframeSemantic,
                "keyframe_semantics",
                self._cb_keyframe_semantic,
                10,
            )
            # 5. DynamicObjects
            self.create_subscription(
                DynamicObjects,
                "dynamic_objects",
                self._cb_dynamic_objects,
                10,
            )
            # 6. Zone
            self.create_subscription(
                Zone,
                "zones",
                self._cb_zone,
                tl_qos,
            )
        else:
            self.get_logger().warn(
                "[MetricsRecorder] Custom msgs not found — "
                "keyframe_semantics / dynamic_objects / zones will NOT be recorded."
            )

        # Periodic summary every 30 seconds
        self.create_timer(30.0, self._print_summary)

        self.get_logger().info(
            "[MetricsRecorder] ✅ Ready — listening for data")

    # ─── helpers ─────────────────────────────────────────────────────────
    def _open_csv(self, name, header):
        path = os.path.join(self.output_dir, f"{name}.csv")
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        self._files[name] = f
        self._writers[name] = w

    def _write(self, name, row):
        self._writers[name].writerow(row)
        self._files[name].flush()

    # ─── callbacks ───────────────────────────────────────────────────────
    def _cb_corrected_pose(self, msg: PoseStamped):
        row = pose_to_row(msg.header.stamp.sec,
                          msg.header.stamp.nanosec, msg.pose)
        self._write("corrected_trajectory", row)
        self.n_corrected += 1

    def _cb_raw_odom(self, msg: Odometry):
        row = pose_to_row(
            msg.header.stamp.sec, msg.header.stamp.nanosec,
            msg.pose.pose,
        )
        self._write("raw_odometry", row)
        self.n_odom += 1

    def _cb_iqa(self, msg: Float64):
        now = self.get_clock().now().to_msg()
        self._write("image_quality", [now.sec, now.nanosec, msg.data])
        self.n_iqa += 1

    def _cb_keyframe_semantic(self, msg):
        stamp = msg.header.stamp
        obj_labels = ";".join(
            msg.detected_objects) if msg.detected_objects else ""
        p = msg.pose.position
        q = msg.pose.orientation
        row = [
            stamp.sec, stamp.nanosec,
            msg.keyframe_id,
            len(msg.clip_embedding),
            len(msg.detected_objects),
            obj_labels,
            msg.scene_dynamicity,
            msg.num_dynamic_clusters,
            msg.image_quality,
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
        ]
        self._write("keyframe_semantics", row)
        self.n_kf_sem += 1

    def _cb_dynamic_objects(self, msg):
        stamp = msg.header.stamp
        cluster_sizes = ";".join(
            str(s) for s in msg.cluster_pixel_count) if msg.cluster_pixel_count else ""
        row = [
            stamp.sec, stamp.nanosec,
            msg.scene_dynamicity,
            msg.num_dynamic_clusters,
            cluster_sizes,
        ]
        self._write("dynamic_objects", row)
        self.n_dyn += 1

    def _cb_zone(self, msg):
        stamp = msg.header.stamp
        action_map = {0: "CREATE", 1: "UPDATE", 2: "DELETE"}
        labels = ";".join(msg.top_labels) if msg.top_labels else ""
        row = [
            stamp.sec, stamp.nanosec,
            msg.zone_id,
            action_map.get(msg.action, str(msg.action)),
            msg.floor_id,
            len(msg.keyframe_ids),
            len(msg.room_ids),
            labels,
            msg.confidence,
            msg.version,
        ]
        self._write("zone_events", row)
        self.n_zone += 1

    def _print_summary(self):
        elapsed = time.time() - self.start_time
        self.get_logger().info(
            f"[MetricsRecorder] {elapsed:.0f}s elapsed | "
            f"corrected={self.n_corrected}  odom={self.n_odom}  "
            f"kf_sem={self.n_kf_sem}  dyn={self.n_dyn}  "
            f"zones={self.n_zone}  iqa={self.n_iqa}"
        )

    # ─── cleanup ─────────────────────────────────────────────────────────
    def destroy_node(self):
        self._print_summary()
        for f in self._files.values():
            f.close()
        self.get_logger().info("[MetricsRecorder] All CSV files closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
