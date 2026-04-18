#!/usr/bin/env python3
"""
Consolidated zone clustering node.

Architecture:
- cluster by pose only (DBSCAN or HDBSCAN via parameter)
- compute semantic prototype AFTER pose clustering
- fuse CLIP + object histogram for semantic confirmation / merge
- maintain dirty/versioned zones
- publish only dirty zones
- provide RViz perimeter + zone->keyframe link markers

This is a consolidated version of the changes discussed in the conversation.
Adapt topic names / message imports to your workspace if needed.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import Header
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from sklearn.cluster import DBSCAN

try:
    from hdbscan import HDBSCAN
except Exception:
    HDBSCAN = None

# Update these imports to your actual package names
from situational_graphs_msgs.msg import KeyframeSemantic
from situational_graphs_msgs.msg import Zone as ZoneMsg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_vec(v):
    v = np.asarray(v, dtype=float).ravel()
    s = float(np.sum(v))
    if s <= 1e-12:
        return np.zeros_like(v)
    return v / s


def l2_normalize(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = max(len(a), len(b))
    aa = np.zeros(n, dtype=float)
    bb = np.zeros(n, dtype=float)
    aa[:len(a)] = a
    bb[:len(b)] = b
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence in [0, 1] approximately."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    n = max(len(p), len(q))
    pp = np.zeros(n, dtype=float)
    qq = np.zeros(n, dtype=float)
    pp[:len(p)] = p
    qq[:len(q)] = q

    pp = pp + eps
    qq = qq + eps
    pp = pp / np.sum(pp)
    qq = qq / np.sum(qq)

    m = 0.5 * (pp + qq)
    kl_pm = np.sum(pp * np.log(pp / m))
    kl_qm = np.sum(qq * np.log(qq / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(js / np.log(2.0))


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class ZoneRecord:
    zone_id: int
    floor_id: int
    centroid: Tuple[float, float]
    sem_sig: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    clip_sig: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    top_label: str = ""
    top_conf: float = 0.0
    confidence: float = 0.0
    members: set = field(default_factory=set)
    state: str = "confirmed"
    last_updated: float = 0.0
    version: int = 1
    dirty: bool = True
    last_published_version: int = 0
    action: int = 0  # 0 CREATE, 1 UPDATE, 2 DELETE


# -----------------------------------------------------------------------------
# Node
# -----------------------------------------------------------------------------

class ZonesClusterNode(Node):
    """
    Consolidated zone clustering node.

    Key ideas:
    - pose-only clustering
    - semantic confirmation after clustering
    - persistent zone merge/update by pose + semantic distribution similarity
    - dirty/versioned publish
    - RViz visualization for perimeters and zone-keyframe links
    """

    # def __init__(self):
    #     super().__init__('zones_cluster_node')

    #     # ------------------------------------------------------------------
    #     # Parameters
    #     # ------------------------------------------------------------------
    #     self.declare_parameter('publish_rate', 1.0)
    #     self.declare_parameter('clustering_period', 3.0)
    #     self.declare_parameter('window_duration_s', 200.0)

    #     self.declare_parameter('cluster_backend', 'dbscan')  # 'dbscan' or 'hdbscan'
    #     self.declare_parameter('pose_cluster_eps_m', 2.2)
    #     self.declare_parameter('pose_cluster_min_samples', 3)
    #     self.declare_parameter('hdbscan_min_cluster_size', 3)
    #     self.declare_parameter('hdbscan_min_samples', 2)

    #     self.declare_parameter('min_kfs_confirm', 3)
    #     self.declare_parameter('min_scene_cohesion', 0.55)
    #     self.declare_parameter('semantic_merge_thresh', 0.45)
    #     self.declare_parameter('obj_merge_weight', 0.7)
    #     self.declare_parameter('clip_merge_weight', 0.3)
    #     self.declare_parameter('clip_update_beta', 0.12)

    #     self.declare_parameter('merge_dist_m', 4.0)
    #     self.declare_parameter('merge_semantic_thresh', 0.45)  # kept for compatibility
    #     self.declare_parameter('temporal_confirm_s', 8.0)
    #     self.declare_parameter('zone_confidence_base', 0.75)

    #     self.declare_parameter('zone_centroid_update_thresh', 0.25)
    #     self.declare_parameter('zone_confidence_update_thresh', 0.05)

    #     self.declare_parameter('visualize', True)
    #     self.declare_parameter('enable_perimeters', True)
    #     self.declare_parameter('enable_zone_links', True)

    #     self.declare_parameter('pose_weight', 1.0)
    #     self.declare_parameter('semantic_distance_lambda', 1.0)
    #     self.declare_parameter('combined_cluster_eps', 3.0)
    #     self.declare_parameter('clip_weight', 1.0)

    #     self.publish_rate = float(self.get_parameter('publish_rate').value)
    #     self.clustering_period = float(self.get_parameter('clustering_period').value)
    #     self.window_duration_s = float(self.get_parameter('window_duration_s').value)

    #     self.cluster_backend = str(self.get_parameter('cluster_backend').value).lower()
    #     self.pose_cluster_eps_m = float(self.get_parameter('pose_cluster_eps_m').value)
    #     self.pose_cluster_min_samples = int(self.get_parameter('pose_cluster_min_samples').value)
    #     self.hdbscan_min_cluster_size = int(self.get_parameter('hdbscan_min_cluster_size').value)
    #     self.hdbscan_min_samples = int(self.get_parameter('hdbscan_min_samples').value)

    #     self.min_kfs_confirm = int(self.get_parameter('min_kfs_confirm').value)
    #     self.min_scene_cohesion = float(self.get_parameter('min_scene_cohesion').value)
    #     self.semantic_merge_thresh = float(self.get_parameter('semantic_merge_thresh').value)
    #     self.obj_merge_weight = float(self.get_parameter('obj_merge_weight').value)
    #     self.clip_merge_weight = float(self.get_parameter('clip_merge_weight').value)
    #     self.clip_update_beta = float(self.get_parameter('clip_update_beta').value)

    #     self.merge_dist_m = float(self.get_parameter('merge_dist_m').value)
    #     self.merge_semantic_thresh = float(self.get_parameter('merge_semantic_thresh').value)
    #     self.temporal_confirm_s = float(self.get_parameter('temporal_confirm_s').value)
    #     self.zone_confidence_base = float(self.get_parameter('zone_confidence_base').value)

    #     self.zone_centroid_update_thresh = float(self.get_parameter('zone_centroid_update_thresh').value)
    #     self.zone_confidence_update_thresh = float(self.get_parameter('zone_confidence_update_thresh').value)

    #     self.visualize = bool(self.get_parameter('visualize').value)
    #     self.enable_perimeters = bool(self.get_parameter('enable_perimeters').value)
    #     self.enable_zone_links = bool(self.get_parameter('enable_zone_links').value)

    #     self.pose_weight = float(self.get_parameter('pose_weight').value)
    #     self.semantic_distance_lambda = float(self.get_parameter('semantic_distance_lambda').value)
    #     self.combined_cluster_eps = float(self.get_parameter('combined_cluster_eps').value)
    #     self.clip_weight = float(self.get_parameter('clip_weight').value)

    #     # ------------------------------------------------------------------
    #     # State
    #     # ------------------------------------------------------------------
    #     self.buffer: deque = deque()
    #     self.lock = __import__('threading').Lock()
    #     self.zones: Dict[int, dict] = {}
    #     self.next_zone_id = 1
    #     self.last_cluster_time = time.time()
    #     self.clip_dim_hint = None
    #     self.vocab: Dict[str, int] = {}
    #     self.vocab_rev: List[str] = []

    #     # ROS I/O
    #     self.kf_sub = self.create_subscription(
    #         KeyframeSemantic,
    #         '/s_graphs/keyframe_semantic',
    #         self.cb_kf_semantic,
    #         10,
    #     )

    #     qos_z = QoSProfile(depth=10)
    #     qos_z.durability = DurabilityPolicy.TRANSIENT_LOCAL
    #     qos_z.reliability = ReliabilityPolicy.RELIABLE

    #     self.zone_pub = self.create_publisher(ZoneMsg, '/zones', qos_z)
    #     self.marker_pub = self.create_publisher(MarkerArray, 'zones_markers', 10)
    #     self.perimeter_pub = self.create_publisher(MarkerArray, 'zone_perimeters', 10)
    #     self.zone_links_pub = self.create_publisher(MarkerArray, 'zone_keyframe_links', 10)

    #     self.timer = self.create_timer(1.0 / max(self.publish_rate, 1e-6), self.timer_publish)

    #     self.get_logger().info('ZonesClusterNode started (pose-first clustering + semantic confirmation).')
    def __init__(self):
        super().__init__('zones_cluster_node')

        # Parameters
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('clustering_period', 3.0)
        self.declare_parameter('window_duration_s', 200.0)

        self.declare_parameter('cluster_backend', 'dbscan')
        self.declare_parameter('pose_cluster_eps_m', 2.2)
        self.declare_parameter('pose_cluster_min_samples', 3)
        self.declare_parameter('hdbscan_min_cluster_size', 3)
        self.declare_parameter('hdbscan_min_samples', 2)

        self.declare_parameter('min_kfs_confirm', 3)
        self.declare_parameter('min_scene_cohesion', 0.55)
        self.declare_parameter('semantic_merge_thresh', 0.45)
        self.declare_parameter('obj_merge_weight', 0.7)
        self.declare_parameter('clip_merge_weight', 0.3)
        self.declare_parameter('clip_update_beta', 0.12)

        self.declare_parameter('merge_dist_m', 4.0)
        self.declare_parameter('merge_semantic_thresh', 0.45)
        self.declare_parameter('temporal_confirm_s', 8.0)
        self.declare_parameter('zone_confidence_base', 0.75)

        self.declare_parameter('zone_centroid_update_thresh', 0.25)
        self.declare_parameter('zone_confidence_update_thresh', 0.05)

        self.declare_parameter('visualize', True)
        self.declare_parameter('enable_perimeters', True)
        self.declare_parameter('enable_zone_links', True)

        self.declare_parameter('pose_weight', 1.0)
        self.declare_parameter('clip_weight', 1.0)
        self.declare_parameter('semantic_distance_lambda', 1.0)
        self.declare_parameter('combined_cluster_eps', 3.0)
        self.declare_parameter('marker_ns', 'zones')

        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.clustering_period = float(self.get_parameter('clustering_period').value)
        self.window_duration_s = float(self.get_parameter('window_duration_s').value)

        self.cluster_backend = str(self.get_parameter('cluster_backend').value).lower()
        self.pose_cluster_eps_m = float(self.get_parameter('pose_cluster_eps_m').value)
        self.pose_cluster_min_samples = int(self.get_parameter('pose_cluster_min_samples').value)
        self.hdbscan_min_cluster_size = int(self.get_parameter('hdbscan_min_cluster_size').value)
        self.hdbscan_min_samples = int(self.get_parameter('hdbscan_min_samples').value)

        self.min_kfs_confirm = int(self.get_parameter('min_kfs_confirm').value)
        self.min_scene_cohesion = float(self.get_parameter('min_scene_cohesion').value)
        self.semantic_merge_thresh = float(self.get_parameter('semantic_merge_thresh').value)
        self.obj_merge_weight = float(self.get_parameter('obj_merge_weight').value)
        self.clip_merge_weight = float(self.get_parameter('clip_merge_weight').value)
        self.clip_update_beta = float(self.get_parameter('clip_update_beta').value)

        self.merge_dist_m = float(self.get_parameter('merge_dist_m').value)
        self.merge_semantic_thresh = float(self.get_parameter('merge_semantic_thresh').value)
        self.temporal_confirm_s = float(self.get_parameter('temporal_confirm_s').value)
        self.zone_confidence_base = float(self.get_parameter('zone_confidence_base').value)

        self.zone_centroid_update_thresh = float(self.get_parameter('zone_centroid_update_thresh').value)
        self.zone_confidence_update_thresh = float(self.get_parameter('zone_confidence_update_thresh').value)

        self.visualize = bool(self.get_parameter('visualize').value)
        self.enable_perimeters = bool(self.get_parameter('enable_perimeters').value)
        self.enable_zone_links = bool(self.get_parameter('enable_zone_links').value)

        self.pose_weight = float(self.get_parameter('pose_weight').value)
        self.clip_weight = float(self.get_parameter('clip_weight').value)
        self.semantic_distance_lambda = float(self.get_parameter('semantic_distance_lambda').value)
        self.combined_cluster_eps = float(self.get_parameter('combined_cluster_eps').value)
        self.marker_ns = str(self.get_parameter('marker_ns').value)

        # State
        self.buffer = deque()
        self.lock = __import__('threading').Lock()
        self.zones = {}
        self.next_zone_id = 1
        self.last_cluster_time = time.time()
        self.buffer_dirty = False
        self.clip_dim_hint = None
        self.vocab = {}
        self.vocab_rev = []

        # ROS I/O
        self.kf_sub = self.create_subscription(
            KeyframeSemantic,
            '/s_graphs/keyframe_semantic',
            self.cb_kf_semantic,
            10,
        )

        qos_z = QoSProfile(depth=10)
        qos_z.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_z.reliability = ReliabilityPolicy.RELIABLE

        self.zone_pub = self.create_publisher(ZoneMsg, '/zones', qos_z)
        self.marker_pub = self.create_publisher(MarkerArray, 'zones_markers', 10)
        self.perimeter_pub = self.create_publisher(MarkerArray, 'zone_perimeters', 10)
        self.zone_links_pub = self.create_publisher(MarkerArray, 'zone_keyframe_links', 10)

        self.timer = self.create_timer(1.0 / max(self.publish_rate, 1e-6), self.timer_publish)

        self.get_logger().info('ZonesClusterNode started.')

    # ------------------------------------------------------------------
    # Message / buffer handling
    # ------------------------------------------------------------------

    def extract_clip_vector(self, msg):
        """Extract CLIP embedding from message if present."""
        for field in ('clip_embedding', 'scene_embedding', 'embedding', 'clip_feat'):
            raw = getattr(msg, field, None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float).ravel()
            if arr.size > 0:
                arr = l2_normalize(arr)
                if self.clip_dim_hint is None:
                    self.clip_dim_hint = arr.size
                elif arr.size != self.clip_dim_hint:
                    arr = self.pad_vec_to_len(arr, self.clip_dim_hint)
                return arr
        return np.zeros(0, dtype=float)

    def pad_vec_to_len(self, vec, length):
        vec = np.asarray(vec, dtype=float).ravel()
        if len(vec) == length:
            return vec.copy()
        out = np.zeros(length, dtype=float)
        out[:min(len(vec), length)] = vec[:min(len(vec), length)]
        return out

    def cb_kf_semantic(self, msg):
        try:
            if hasattr(msg, 'pose'):
                pose = msg.pose
                x = float(pose.position.x)
                y = float(pose.position.y)
            else:
                pose = msg.pose.pose
                x = float(pose.position.x)
                y = float(pose.position.y)

            t = self.get_clock().now().nanoseconds * 1e-9
            kfid = int(getattr(msg, 'keyframe_id', getattr(msg, 'id', -1)))
            if kfid < 0:
                return

            labels = list(getattr(msg, 'objects', []))
            confs = list(getattr(msg, 'object_confidence', []))
            sem_dict = {}
            for i, lab in enumerate(labels):
                if lab is None:
                    continue
                lab_s = str(lab)
                c = float(confs[i]) if i < len(confs) else 1.0
                sem_dict[lab_s] = max(sem_dict.get(lab_s, 0.0), c)

            clip_vec = self.extract_clip_vector(msg)

            with self.lock:
                expanded = False
                for lab in list(sem_dict.keys()):
                    if lab not in self.vocab:
                        idx = len(self.vocab_rev)
                        self.vocab[lab] = idx
                        self.vocab_rev.append(lab)
                        expanded = True

                if expanded and len(self.vocab_rev) > 0:
                    for entry in self.buffer:
                        old_vec = np.asarray(entry.get('sem_vec', np.zeros(0, dtype=float)), dtype=float).ravel()
                        if len(old_vec) != len(self.vocab_rev):
                            tmp = np.zeros(len(self.vocab_rev), dtype=float)
                            tmp[:min(len(old_vec), len(tmp))] = old_vec[:min(len(old_vec), len(tmp))]
                            entry['sem_vec'] = tmp

                if len(self.vocab_rev) > 0:
                    sem_vec = np.zeros(len(self.vocab_rev), dtype=float)
                    for lab, c in sem_dict.items():
                        idx = self.vocab.get(lab, None)
                        if idx is not None:
                            sem_vec[idx] = float(c)
                    if sem_vec.size > 0:
                        sem_vec = normalize_vec(sem_vec)
                else:
                    sem_vec = np.zeros(0, dtype=float)

                self.buffer.append({
                    'id': kfid,
                    't': t,
                    'x': x,
                    'y': y,
                    'pose': msg.pose if hasattr(msg, 'pose') else None,
                    'sem': sem_dict,
                    'sem_vec': sem_vec,
                    'clip_vec': clip_vec,
                })

                cutoff = t - self.window_duration_s
                while len(self.buffer) > 0 and self.buffer[0]['t'] < cutoff:
                    self.buffer.popleft()

                self.buffer_dirty = True

            self.get_logger().info(f"Received KF semantic id={kfid} buffer_size={len(self.buffer)}")

        except Exception as e:
            self.get_logger().error(f'cb_kf_semantic failed: {e}')
            
    def cb_kf_semantic(self, msg):
        try:
            if hasattr(msg, 'pose'):
                pose = msg.pose
                x = float(pose.position.x)
                y = float(pose.position.y)
            else:
                pose = msg.pose.pose
                x = float(pose.position.x)
                y = float(pose.position.y)

            t = self.get_clock().now().nanoseconds * 1e-9
            kfid = int(getattr(msg, 'keyframe_id', getattr(msg, 'id', -1)))
            if kfid < 0:
                return

            labels = list(getattr(msg, 'objects', []))
            confs = list(getattr(msg, 'object_confidence', []))
            sem_dict = {}
            for i, lab in enumerate(labels):
                if lab is None:
                    continue
                lab_s = str(lab)
                c = float(confs[i]) if i < len(confs) else 1.0
                sem_dict[lab_s] = max(sem_dict.get(lab_s, 0.0), c)

            clip_vec = self.extract_clip_vector(msg)

            with self.lock:
                expanded = False
                for lab in list(sem_dict.keys()):
                    if lab not in self.vocab:
                        idx = len(self.vocab_rev)
                        self.vocab[lab] = idx
                        self.vocab_rev.append(lab)
                        expanded = True

                if expanded and len(self.vocab_rev) > 0:
                    for entry in self.buffer:
                        old_vec = np.asarray(entry.get('sem_vec', np.zeros(0, dtype=float)), dtype=float).ravel()
                        if len(old_vec) != len(self.vocab_rev):
                            tmp = np.zeros(len(self.vocab_rev), dtype=float)
                            tmp[:min(len(old_vec), len(tmp))] = old_vec[:min(len(old_vec), len(tmp))]
                            entry['sem_vec'] = tmp

                if len(self.vocab_rev) > 0:
                    sem_vec = np.zeros(len(self.vocab_rev), dtype=float)
                    for lab, c in sem_dict.items():
                        idx = self.vocab.get(lab, None)
                        if idx is not None:
                            sem_vec[idx] = float(c)
                    if sem_vec.size > 0:
                        sem_vec = normalize_vec(sem_vec)
                else:
                    sem_vec = np.zeros(0, dtype=float)

                self.buffer.append({
                    'id': kfid,
                    't': t,
                    'x': x,
                    'y': y,
                    'pose': msg.pose if hasattr(msg, 'pose') else None,
                    'sem': sem_dict,
                    'sem_vec': sem_vec,
                    'clip_vec': clip_vec,
                })

                cutoff = t - self.window_duration_s
                while len(self.buffer) > 0 and self.buffer[0]['t'] < cutoff:
                    self.buffer.popleft()

                self.buffer_dirty = True

            self.get_logger().info(f"Received KF semantic id={kfid} buffer_size={len(self.buffer)}")

        except Exception as e:
            self.get_logger().error(f'cb_kf_semantic failed: {e}')

    # ------------------------------------------------------------------
    # Clustering / semantic scoring
    # ------------------------------------------------------------------

    def semantic_distribution_distance(self, obj_a, clip_a, obj_b, clip_b):
        """
        Distribution distance between two semantic descriptions.
        Uses Jensen-Shannon on object histograms and cosine distance on CLIP.
        Returns a scalar in [0, ~2].
        """
        obj_a = np.asarray(obj_a, dtype=float).ravel()
        obj_b = np.asarray(obj_b, dtype=float).ravel()
        clip_a = np.asarray(clip_a, dtype=float).ravel()
        clip_b = np.asarray(clip_b, dtype=float).ravel()

        obj_dist = 0.0
        if obj_a.size > 0 and obj_b.size > 0:
            L = max(len(obj_a), len(obj_b))
            obj_dist = js_divergence(self.pad_vec_to_len(obj_a, L), self.pad_vec_to_len(obj_b, L))

        clip_dist = 0.0
        if clip_a.size > 0 and clip_b.size > 0:
            clip_dist = 1.0 - max(-1.0, min(1.0, cosine_similarity(clip_a, clip_b)))
            if clip_dist < 0.0:
                clip_dist = 0.0

        return float(self.obj_merge_weight * obj_dist + self.clip_merge_weight * clip_dist)

    def combined_pair_distance(self, entry_a, entry_b):
        """
        d = ||p1 - p2|| * (1 + lambda * semantic_distance)
        This is the metric the user requested.
        """
        pa = np.array([float(entry_a['x']), float(entry_a['y'])], dtype=float)
        pb = np.array([float(entry_b['x']), float(entry_b['y'])], dtype=float)
        spatial = float(np.linalg.norm(pa - pb))

        sem_dist = self.semantic_distribution_distance(
            entry_a.get('sem_vec', np.zeros(0, dtype=float)),
            entry_a.get('clip_vec', np.zeros(0, dtype=float)),
            entry_b.get('sem_vec', np.zeros(0, dtype=float)),
            entry_b.get('clip_vec', np.zeros(0, dtype=float)),
        )
        return spatial * (1.0 + self.semantic_distance_lambda * sem_dist)

    def pairwise_distance_matrix(self, entries):
        n = len(entries)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.combined_pair_distance(entries[i], entries[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def cluster_distance_matrix(self, D):
        backend = self.cluster_backend
        if backend == 'hdbscan':
            if HDBSCAN is None:
                raise RuntimeError('cluster_backend=hdbscan but hdbscan is not installed')
            model = HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric='precomputed'
            )
            return model.fit_predict(D)

        model = DBSCAN(
            eps=self.combined_cluster_eps,
            min_samples=self.pose_cluster_min_samples,
            metric='precomputed'
        )
        return model.fit_predict(D)

    def build_feature_matrix(self):
        """Return a snapshot of buffered keyframes for pairwise distance clustering."""
        with self.lock:
            if len(self.buffer) == 0:
                return None, None
            entries = list(self.buffer)
            return entries, list(range(len(entries)))

    def build_semantic_prototype(self, members):
        """
        Build semantic prototype AFTER pose clustering.
        Returns:
          obj_sig, clip_sig, top_label, top_conf, cohesion
        """
        obj_vecs = []
        clip_vecs = []
        label_scores = defaultdict(float)

        for m in members:
            sem_vec = np.asarray(m.get('sem_vec', np.zeros(0, dtype=float)), dtype=float).ravel()
            if sem_vec.size > 0:
                obj_vecs.append(sem_vec)

            clip_vec = np.asarray(m.get('clip_vec', np.zeros(0, dtype=float)), dtype=float).ravel()
            if clip_vec.size > 0:
                clip_vecs.append(l2_normalize(clip_vec))

            sem_dict = m.get('sem', {})
            for lab, c in sem_dict.items():
                label_scores[lab] = max(label_scores.get(lab, 0.0), float(c))

        if len(obj_vecs) > 0:
            obj_mat = np.asarray(obj_vecs, dtype=float)
            obj_sig = normalize_vec(np.mean(obj_mat, axis=0))
        else:
            obj_sig = np.zeros(0, dtype=float)

        if len(clip_vecs) > 0:
            clip_mat = np.asarray(clip_vecs, dtype=float)
            clip_sig = l2_normalize(np.mean(clip_mat, axis=0))
        else:
            clip_sig = np.zeros(0, dtype=float)

        if len(label_scores) > 0:
            top_label, top_conf = max(label_scores.items(), key=lambda kv: kv[1])
        else:
            top_label, top_conf = '', 0.0

        cohesion_terms = []

        if obj_sig.size > 0 and len(obj_vecs) > 0:
            obj_sims = []
            for v in obj_vecs:
                if v.size == 0:
                    continue
                obj_sims.append(1.0 - js_divergence(v, obj_sig))
            if len(obj_sims) > 0:
                cohesion_terms.append(float(np.mean(obj_sims)))

        if clip_sig.size > 0 and len(clip_vecs) > 0:
            clip_sims = []
            for v in clip_vecs:
                if v.size == 0:
                    continue
                clip_sims.append(cosine_similarity(v, clip_sig))
            if len(clip_sims) > 0:
                cohesion_terms.append(float(np.mean(clip_sims)))

        cohesion = float(np.mean(cohesion_terms)) if len(cohesion_terms) > 0 else 0.0
        return obj_sig, clip_sig, top_label, top_conf, cohesion

    def run_clustering(self):
        entries, idxs = self.build_feature_matrix()
        if entries is None or len(entries) < self.pose_cluster_min_samples:
            return

        try:
            D = self.pairwise_distance_matrix(entries)
            labels = self.cluster_distance_matrix(D)
        except Exception as e:
            self.get_logger().error(f'Clustering failed: {e}')
            return

        unique_labels = set(labels.tolist())
        if -1 in unique_labels:
            unique_labels.remove(-1)

        clusters = []
        for lbl in unique_labels:
            members_idx = [i for (i, lab) in enumerate(labels) if lab == lbl]
            members = [entries[i] for i in members_idx]
            clusters.append((lbl, members_idx, members))

        for (lbl, members_idx, members) in clusters:
            kf_ids = [int(m['id']) for m in members]
            xs = [float(m['x']) for m in members]
            ys = [float(m['y']) for m in members]
            centroid = (float(np.mean(xs)), float(np.mean(ys)))

            times = [float(m['t']) for m in members]
            duration = max(times) - min(times) if len(times) >= 2 else 0.0

            obj_sig, clip_sig, top_label, top_conf, cohesion = self.build_semantic_prototype(members)

            membership_score = min(1.0, len(kf_ids) / max(1.0, float(self.min_kfs_confirm)))
            zone_conf = float(
                self.zone_confidence_base * 0.4 +
                0.6 * (0.5 * cohesion + 0.5 * membership_score)
            )

            confirmed = False
            if len(kf_ids) >= self.min_kfs_confirm and cohesion >= self.min_scene_cohesion:
                confirmed = True
            elif duration >= self.temporal_confirm_s and cohesion >= 0.30:
                confirmed = True

            if confirmed:
                self.create_or_update_zone(
                    kf_ids,
                    centroid,
                    obj_sig,
                    clip_sig,
                    top_label,
                    top_conf,
                    zone_conf,
                )

    def create_or_update_zone(self, kf_ids, centroid_xy, obj_sig, clip_sig, top_label, top_conf, conf):
        now = time.time()

        obj_sig = np.asarray(obj_sig, dtype=float).ravel()
        clip_sig = np.asarray(clip_sig, dtype=float).ravel()

        merged_zone_id = None
        best_score = -1.0

        for zid, z in self.zones.items():
            zx, zy = z.centroid
            d = math.hypot(zx - centroid_xy[0], zy - centroid_xy[1])
            if d > self.merge_dist_m:
                continue

            z_obj = np.asarray(z.sem_sig, dtype=float).ravel()
            z_clip = np.asarray(z.clip_sig, dtype=float).ravel()

            obj_sim = 0.0
            if z_obj.size > 0 and obj_sig.size > 0:
                L = max(len(z_obj), len(obj_sig))
                z_obj_p = self.pad_vec_to_len(z_obj, L)
                obj_sig_p = self.pad_vec_to_len(obj_sig, L)
                obj_sim = 1.0 - js_divergence(z_obj_p, obj_sig_p)

            clip_sim = 0.0
            if z_clip.size > 0 and clip_sig.size > 0:
                clip_sim = cosine_similarity(z_clip, clip_sig)

            sem_score = self.obj_merge_weight * obj_sim
            if z_clip.size > 0 and clip_sig.size > 0:
                sem_score += self.clip_merge_weight * max(0.0, clip_sim)

            if sem_score >= self.semantic_merge_thresh and sem_score > best_score:
                merged_zone_id = zid
                best_score = sem_score

        if merged_zone_id is None:
            zid = self.next_zone_id
            self.next_zone_id += 1
            zone = ZoneRecord(
                zone_id=zid,
                floor_id=0,
                centroid=(float(centroid_xy[0]), float(centroid_xy[1])),
                sem_sig=obj_sig.copy(),
                clip_sig=clip_sig.copy(),
                top_label=top_label,
                top_conf=float(top_conf),
                confidence=float(conf),
                members=set(kf_ids),
                state='confirmed',
                last_updated=now,
                version=1,
                dirty=True,
                last_published_version=0,
                action=0,
            )
            self.zones[zid] = zone
            self.get_logger().info(
                f'Created zone {zid} center=({zone.centroid[0]:.2f},{zone.centroid[1]:.2f}) '
                f'size={len(kf_ids)} label={top_label} conf={conf:.2f}'
            )
            return zone

        z = self.zones[merged_zone_id]
        old_members = set(z.members)
        new_members = set(kf_ids) - old_members

        if len(new_members) == 0:
            # Same support set seen again. Do not keep re-updating the zone.
            return z

        old_centroid = z.centroid
        old_obj = np.asarray(z.sem_sig, dtype=float).ravel()
        old_clip = np.asarray(z.clip_sig, dtype=float).ravel()
        old_top_label = z.top_label
        old_top_conf = float(z.top_conf)
        old_conf = float(z.confidence)

        z.members.update(kf_ids)

        with self.lock:
            member_positions = []
            member_times = []
            for entry in self.buffer:
                if entry['id'] in z.members:
                    member_positions.append((entry['x'], entry['y']))
                    member_times.append(float(entry['t']))

        if len(member_positions) > 0:
            xs = [p[0] for p in member_positions]
            ys = [p[1] for p in member_positions]
            new_centroid = (float(np.mean(xs)), float(np.mean(ys)))
        else:
            new_centroid = old_centroid

        if old_obj.size == 0:
            new_obj = obj_sig.copy()
        elif obj_sig.size == 0:
            new_obj = old_obj.copy()
        else:
            L = max(len(old_obj), len(obj_sig))
            old_obj_p = self.pad_vec_to_len(old_obj, L)
            obj_sig_p = self.pad_vec_to_len(obj_sig, L)
            new_obj = normalize_vec((1.0 - self.clip_update_beta) * old_obj_p + self.clip_update_beta * obj_sig_p)

        if old_clip.size == 0:
            new_clip = clip_sig.copy()
        elif clip_sig.size == 0:
            new_clip = old_clip.copy()
        else:
            L = max(len(old_clip), len(clip_sig))
            old_clip_p = self.pad_vec_to_len(old_clip, L)
            clip_sig_p = self.pad_vec_to_len(clip_sig, L)
            new_clip = l2_normalize((1.0 - self.clip_update_beta) * old_clip_p + self.clip_update_beta * clip_sig_p)

        new_top_label = top_label if top_conf >= old_top_conf else old_top_label
        new_top_conf = max(top_conf, old_top_conf)
        new_conf = min(1.0, 0.9 * old_conf + 0.1 * conf)

        centroid_shift = math.hypot(new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
        members_changed = (z.members != old_members)
        obj_changed = np.linalg.norm(
            self.pad_vec_to_len(old_obj, max(len(old_obj), len(new_obj))) -
            self.pad_vec_to_len(new_obj, max(len(old_obj), len(new_obj)))
        ) > 1e-6
        clip_changed = np.linalg.norm(
            self.pad_vec_to_len(old_clip, max(len(old_clip), len(new_clip))) -
            self.pad_vec_to_len(new_clip, max(len(old_clip), len(new_clip)))
        ) > 1e-6
        label_changed = (new_top_label != old_top_label)
        conf_changed = abs(new_conf - old_conf) > self.zone_confidence_update_thresh
        centroid_changed = centroid_shift > self.zone_centroid_update_thresh

        z.centroid = new_centroid
        z.sem_sig = new_obj
        z.clip_sig = new_clip
        z.top_label = new_top_label
        z.top_conf = float(new_top_conf)
        z.confidence = float(new_conf)
        z.last_updated = max(member_times) if len(member_times) > 0 else now
        z.action = 1

        changed = members_changed or obj_changed or clip_changed or label_changed or conf_changed or centroid_changed
        if changed:
            z.version += 1
            z.dirty = True
            self.get_logger().info(
                f'Merged into zone {merged_zone_id} => new size={len(z.members)} '
                f'centroid=({z.centroid[0]:.2f},{z.centroid[1]:.2f})'
            )
        else:
            z.dirty = False

        return z

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_zones(self):
        """Publish only dirty zones."""
        now = self.get_clock().now().to_msg()
        with self.lock:
            dirty_zones = [(zid, z) for zid, z in self.zones.items() if z.dirty]

        for zid, z in dirty_zones:
            try:
                zm = ZoneMsg()
                try:
                    zm.header = Header()
                    zm.header.stamp = now
                    zm.header.frame_id = 'map'
                except Exception:
                    pass

                if hasattr(zm, 'zone_id'):
                    zm.zone_id = int(zid)
                if hasattr(zm, 'floor_id'):
                    zm.floor_id = int(z.floor_id)
                if hasattr(zm, 'keyframe_ids'):
                    zm.keyframe_ids = list(sorted(z.members))
                if hasattr(zm, 'room_ids'):
                    zm.room_ids = []
                if hasattr(zm, 'top_labels'):
                    zm.top_labels = [z.top_label]
                if hasattr(zm, 'top_label_confidences'):
                    zm.top_label_confidences = [float(z.top_conf)]
                if hasattr(zm, 'confidence'):
                    zm.confidence = float(z.confidence)
                if hasattr(zm, 'version'):
                    zm.version = int(z.version)
                if hasattr(zm, 'action'):
                    zm.action = int(z.action)

                if hasattr(zm, 'centroid'):
                    zm.centroid.position.x = float(z.centroid[0])
                    zm.centroid.position.y = float(z.centroid[1])
                    zm.centroid.position.z = 0.0
                    zm.centroid.orientation.w = 1.0

                # optional fields
                if hasattr(zm, 'polygon'):
                    pass
                if hasattr(zm, 'supporting_planes'):
                    pass
                if hasattr(zm, 'supporting_wall_ids'):
                    pass

                self.zone_pub.publish(zm)

                with self.lock:
                    if zid in self.zones:
                        self.zones[zid].dirty = False
                        self.zones[zid].last_published_version = z.version

            except Exception as e:
                self.get_logger().error(f'Error publishing zone {zid}: {e}')

    def publish_zone_perimeters(self):
        if not self.enable_perimeters:
            return
        ma = MarkerArray()
        marker_id = 0

        with self.lock:
            zones_snapshot = list(self.zones.items())
            buffer_snapshot = list(self.buffer)

        for zid, z in zones_snapshot:
            member_points = []
            member_ids = set(z.members)

            for entry in buffer_snapshot:
                if int(entry['id']) in member_ids:
                    member_points.append((float(entry['x']), float(entry['y'])))

            marker = self.make_cluster_perimeter_marker(
                member_points,
                marker_id=marker_id,
                zone_id=zid,
                frame_id='map',
            )
            ma.markers.append(marker)
            marker_id += 1

        self.perimeter_pub.publish(ma)

    def publish_zone_keyframe_links(self):
        if not self.enable_zone_links:
            return
        ma = MarkerArray()
        marker_id = 0

        with self.lock:
            zones_snapshot = list(self.zones.items())
            buffer_snapshot = list(self.buffer)

        kf_pos = {}
        for entry in buffer_snapshot:
            try:
                kf_pos[int(entry['id'])] = (float(entry['x']), float(entry['y']))
            except Exception:
                continue

        for zid, z in zones_snapshot:
            cx, cy = z.centroid
            member_ids = set(z.members)

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f'zone_links_{zid}'
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.03

            cid = (zid * 37) % 255
            marker.color.r = ((cid * 97) % 255) / 255.0
            marker.color.g = ((cid * 71) % 255) / 255.0
            marker.color.b = ((cid * 53) % 255) / 255.0
            marker.color.a = 0.8

            for kf_id in member_ids:
                if kf_id not in kf_pos:
                    continue
                kx, ky = kf_pos[kf_id]
                p1 = Point(x=float(cx), y=float(cy), z=0.08)
                p2 = Point(x=float(kx), y=float(ky), z=0.08)
                marker.points.append(p1)
                marker.points.append(p2)

            if len(marker.points) > 0:
                ma.markers.append(marker)
                marker_id += 1

        self.zone_links_pub.publish(ma)

    def make_cluster_perimeter_marker(self, cluster_points, marker_id, zone_id, frame_id='map'):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'zone_perimeter_{zone_id}'
        marker.id = int(marker_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.06

        cid = (zone_id * 37) % 255
        marker.color.r = ((cid * 97) % 255) / 255.0
        marker.color.g = ((cid * 71) % 255) / 255.0
        marker.color.b = ((cid * 53) % 255) / 255.0
        marker.color.a = 0.95

        if len(cluster_points) == 0:
            marker.action = Marker.DELETE
            return marker

        if len(cluster_points) == 1:
            x, y = cluster_points[0]
            s = 0.2
            pts = [(x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s), (x - s, y - s)]
        elif len(cluster_points) == 2:
            xs = [p[0] for p in cluster_points]
            ys = [p[1] for p in cluster_points]
            pad = 0.15
            min_x, max_x = min(xs) - pad, max(xs) + pad
            min_y, max_y = min(ys) - pad, max(ys) + pad
            pts = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
        else:
            hull = self.convex_hull_2d(cluster_points)
            if len(hull) < 3:
                xs = [p[0] for p in cluster_points]
                ys = [p[1] for p in cluster_points]
                pad = 0.15
                min_x, max_x = min(xs) - pad, max(xs) + pad
                min_y, max_y = min(ys) - pad, max(ys) + pad
                pts = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
            else:
                pts = hull + [hull[0]]

        for x, y in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        return marker

    def convex_hull_2d(self, points):
        points = sorted(set(points))
        if len(points) <= 1:
            return points

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def purge_deleted_zones(self):
        with self.lock:
            to_delete = [zid for zid, z in self.zones.items() if z.action == 2 and not z.dirty]
            for zid in to_delete:
                del self.zones[zid]

    def timer_publish(self):
        now = time.time()

        if self.buffer_dirty and (now - self.last_cluster_time >= self.clustering_period):
            try:
                self.run_clustering()
                self.buffer_dirty = False
                self.last_cluster_time = now
            except Exception as e:
                self.get_logger().error(f'Exception during clustering: {e}')
                # keep buffer_dirty True so we can retry

        self.publish_zones()
        if self.visualize:
            self.publish_perimeter_and_links()
            self.publish_markers()

        self.purge_deleted_zones()

    def publish_perimeter_and_links(self):
        self.publish_zone_perimeters()
        self.publish_zone_keyframe_links()

    # ------------------------------------------------------------------
    # RViz markers (keep / adapt to your existing implementation)
    # ------------------------------------------------------------------

    # def publish_markers(self):
    #     """Publish centroid markers / labels. Adapt this to your existing marker style."""
    #     ma = MarkerArray()
    #     marker_id = 0

    #     with self.lock:
    #         zones_snapshot = list(self.zones.items())

    #     for zid, z in zones_snapshot:
    #         m = Marker()
    #         m.header.frame_id = 'map'
    #         m.header.stamp = self.get_clock().now().to_msg()
    #         m.ns = 'zone_centroids'
    #         m.id = marker_id
    #         m.type = Marker.SPHERE
    #         m.action = Marker.ADD
    #         m.pose.position.x = float(z.centroid[0])
    #         m.pose.position.y = float(z.centroid[1])
    #         m.pose.position.z = 0.1
    #         m.pose.orientation.w = 1.0
    #         m.scale.x = 0.25
    #         m.scale.y = 0.25
    #         m.scale.z = 0.25
    #         cid = (zid * 37) % 255
    #         m.color.r = ((cid * 97) % 255) / 255.0
    #         m.color.g = ((cid * 71) % 255) / 255.0
    #         m.color.b = ((cid * 53) % 255) / 255.0
    #         m.color.a = 0.95
    #         ma.markers.append(m)
    #         marker_id += 1

    #     self.marker_pub.publish(ma)
    def publish_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        i = 0
        for zid, z in self.zones.items():
            # centroid marker (sphere)
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'map'
            m.ns = self.marker_ns
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(z.centroid[0])
            m.pose.position.y = float(z.centroid[1])
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.4
            # color by zone id (deterministic)
            cid = (zid * 37) % 255
            m.color.r = ((cid * 97) % 255) / 255.0
            m.color.g = ((cid * 71) % 255) / 255.0
            m.color.b = ((cid * 53) % 255) / 255.0
            m.color.a = 0.9
            ma.markers.append(m)
            i += 1

            # text label marker
            mt = Marker()
            mt.header = m.header
            mt.ns = self.marker_ns
            mt.id = i
            mt.type = Marker.TEXT_VIEW_FACING
            mt.action = Marker.ADD
            mt.pose.position.x = float(z.centroid[0])
            mt.pose.position.y = float(z.centroid[1])
            mt.pose.position.z = 0.6
            mt.scale.z = 0.25
            mt.color.a = 1.0
            mt.color.r = 1.0
            mt.color.g = 1.0
            mt.color.b = 1.0
            mt.text = f"Z{zid}:{z.top_label}({len(z.members)})"
            ma.markers.append(mt)
            i += 1

        # publish
        if self.marker_pub is not None:
            self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = ZonesClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
