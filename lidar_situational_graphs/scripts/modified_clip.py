#!/usr/bin/env python3
"""
zones_cluster_node.py

DBSCAN-based hybrid clustering node:
 - consumes KeyframeSemantic messages (pose + objects + confidences)
 - maintains sliding window buffer of recent keyframes
 - constructs features: [x, y, w_s * semantic_vector]
 - runs DBSCAN periodically to find spatial+semantic clusters
 - promotes clusters to tentative zones and confirms them when stable
 - merges close & semantically similar zones
 - publishes Zone messages (lidar_situational_graphs/msg/Zone)
 - publishes RViz markers (visualization_msgs/MarkerArray) for debugging

Tunable params (ROS2 params):
 - publish_rate (Hz): how often to publish zones and visualize
 - clustering_period (s): run DBSCAN every this many seconds
 - window_duration_s: sliding window of last N seconds to cluster on
 - min_samples: DBSCAN min_samples
 - eps_m: DBSCAN eps in meters (on scaled feature space)
 - w_sem: semantic weight factor for concatenated feature vector
 - min_kfs_confirm: how many keyframes needed to confirm a zone
 - purity_thresh: semantic purity to help confirm zone
 - temporal_confirm_s: alternative confirmation by temporal span
 - merge_dist_m: centroid distance for merging zones
 - merge_semantic_thresh: cosine similarity threshold for merging

This node is written to be robust to unknown vocabulary (it expands semantic vector dynamically).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import numpy as np
import math
import time
from collections import deque, defaultdict
from threading import Lock

# clustering
try:
    from sklearn.cluster import DBSCAN
except Exception as e:
    raise RuntimeError("scikit-learn is required. Install with `pip install scikit-learn`") from e

# ROS messages (adjust package name if different)
from situational_graphs_msgs.msg import KeyframeSemantic, Zone
from situational_graphs_msgs.msg import Zone as ZoneMsg  # expected to exist
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header, ColorRGBA

# helper utils
# def cosine_similarity(a, b):
#     a = np.asarray(a, dtype=float)
#     b = np.asarray(b, dtype=float)
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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
    

def normalize_vec(v):
    v = np.asarray(v, dtype=float)
    s = np.sum(v)
    if s <= 0:
        return np.zeros_like(v)
    return v / s

class ZonesClusterNode(Node):
    def __init__(self):
        super().__init__('zones_cluster_node')

        # ---- parameters ----
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('clustering_period', 2.0) #3.0
        self.declare_parameter('window_duration_s', 150.0)# 200 #120
        self.declare_parameter('min_samples', 3)
        self.declare_parameter('eps_m', 2.2)  #3.0 # 1.8 in meters (but will be applied after feature scaling)
        self.declare_parameter('w_sem', 2.0)  # semantic weight
        self.declare_parameter('min_kfs_confirm', 3)
        self.declare_parameter('purity_thresh', 0.6)
        self.declare_parameter('temporal_confirm_s', 8.0)
        self.declare_parameter('merge_dist_m', 3.0) #4.0 #2.0
        self.declare_parameter('merge_semantic_thresh', 0.7)
        self.declare_parameter('zone_confidence_base', 0.5)
        self.declare_parameter('visualize', True)
        self.declare_parameter('marker_ns', 'zones')

        #CLIP
        self.declare_parameter('use_clip_embedding', True)
        self.declare_parameter('pose_weight', 1.0)
        self.declare_parameter('clip_weight', 2.0)
        self.declare_parameter('min_clip_cohesion', 0.55)
        self.declare_parameter('clip_merge_thresh', 0.60)
        self.declare_parameter('clip_update_beta', 0.12)

        # load params
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.clustering_period = float(self.get_parameter('clustering_period').value)
        self.window_duration_s = float(self.get_parameter('window_duration_s').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.eps_m = float(self.get_parameter('eps_m').value)
        self.w_sem = float(self.get_parameter('w_sem').value)
        self.min_kfs_confirm = int(self.get_parameter('min_kfs_confirm').value)
        self.purity_thresh = float(self.get_parameter('purity_thresh').value)
        self.temporal_confirm_s = float(self.get_parameter('temporal_confirm_s').value)
        self.merge_dist_m = float(self.get_parameter('merge_dist_m').value)
        self.merge_semantic_thresh = float(self.get_parameter('merge_semantic_thresh').value)
        self.zone_confidence_base = float(self.get_parameter('zone_confidence_base').value)
        self.visualize = bool(self.get_parameter('visualize').value)
        self.marker_ns = str(self.get_parameter('marker_ns').value)
        self.zone_centroid_update_thresh = 0.25
        self.zone_confidence_update_thresh = 0.05

        #CLIP
        self.use_clip_embedding = bool(self.get_parameter('use_clip_embedding').value)
        self.pose_weight = float(self.get_parameter('pose_weight').value)
        self.clip_weight = float(self.get_parameter('clip_weight').value)
        self.min_clip_cohesion = float(self.get_parameter('min_clip_cohesion').value)
        self.clip_merge_thresh = float(self.get_parameter('clip_merge_thresh').value)
        self.clip_update_beta = float(self.get_parameter('clip_update_beta').value)
        self.clip_dim_hint = None

        # ---- runtime state ----
        self.lock = Lock()
        self.buffer = deque()  # each entry: dict with keys id, t, x, y, pose (PoseStamped or None), sem (dict label->conf), sem_vec (np.array)
        self.vocab = {}  # label -> index
        self.vocab_rev = []  # index -> label
        self.next_zone_id = 1
        self.zones = {}  # zone_id -> zone dict

        # zone dict fields:
        # { 'id', 'members': set(kf_id), 'centroid': (x,y), 'sem_sig': np.array, 'confidence', 'state', 'last_updated', 'version' }

        # publishers
        qos_z = QoSProfile(depth=10)
        qos_z.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_z.reliability = ReliabilityPolicy.RELIABLE
        self.zone_pub = self.create_publisher(ZoneMsg, 'zones', qos_z)

        if self.visualize:
            self.marker_pub = self.create_publisher(MarkerArray, 'zones_markers', 10)
        else:
            self.marker_pub = None
        
        self.perimeter_pub = self.create_publisher(MarkerArray, 'zone_perimeters', 10)
        self.zone_links_pub = self.create_publisher(MarkerArray, 'zone_keyframe_links', 10)

        # subscriber to keyframe semantic
        self.kf_sub = self.create_subscription(KeyframeSemantic, '/s_graphs/keyframe_semantic', self.cb_kf_semantic, 10)

        # timers: clustering and publish/visualize
        self.last_cluster_time = 0.0
        self.create_timer(1.0 / max(0.1, self.publish_rate), self.timer_publish)
        self.get_logger().info("ZonesClusterNode started")

    # ---------------- incoming keyframe semantics ----------------
    def cb_kf_semantic(self, msg: KeyframeSemantic):
        """Callback for KeyframeSemantic messages.
        We expect the message to have at least:
          - a pose field (PoseStamped) or header+stamp and optionally pose
          - objects: list[str]
          - object_confidence: list[float] (same length)
          - possibly keyframe_id (we'll try to read it)
        """
        print("Inside call back")
        self.get_logger().info(f"POSE MSG: {msg.pose}")
        try:
            kfid = int(getattr(msg, 'keyframe_id', -1))
        except Exception:
            kfid = -1
        if kfid < 0:
            # If your message does not have a keyframe_id, we create a synthetic incremental id
            kfid = int(time.time() * 1000)  # fallback unique-ish id
        t = self.get_clock().now().nanoseconds / 1e9

        # Pose extraction — try multiple common field names
        pose_stamped = None
        try:
            if hasattr(msg, 'pose') and msg.pose is not None:
                pose_stamped = msg.pose
            elif hasattr(msg, 'header') and hasattr(msg, 'pose'):  # unlikely
                pose_stamped = msg.pose
        except Exception:
            pose_stamped = None

        # fallback to pose in header or stamp not useful
        x = y = 0.0
        if pose_stamped is not None:
            try:
                x = float(pose_stamped.position.x)
                y = float(pose_stamped.position.y)

                print("OOYYYEEEEEEEEEE x: %d y: %d", x, y)
            except Exception:
                x = y = 0.0

        # semantic labels + confidences
        labels = []
        confs = []
        try:
            labels = list(getattr(msg, 'objects', []))
            confs = list(getattr(msg, 'object_confidence', []))
        except Exception:
            labels = []
            confs = []

        # Build semantics dict label->conf
        sem_dict = {}
        for i, lab in enumerate(labels):
            try:
                c = float(confs[i]) if i < len(confs) else 1.0
            except Exception:
                c = 1.0
            if lab is None:
                continue
            lab_s = str(lab)
            sem_dict[lab_s] = max(sem_dict.get(lab_s, 0.0), float(c))

        # ensure vocab includes these labels (dynamic expansion)
        with self.lock:
            expanded = False
            for lab in list(sem_dict.keys()):
                if lab not in self.vocab:
                    idx = len(self.vocab_rev)
                    self.vocab[lab] = idx
                    self.vocab_rev.append(lab)
                    expanded = True
            # If we expanded vocab, we need to expand sem_vec for existing buffer members
            if expanded and len(self.vocab_rev) > 0:
                for entry in self.buffer:
                    old_vec = entry.get('sem_vec', None)
                    if old_vec is None:
                        # create zero vector of new length
                        entry['sem_vec'] = np.zeros(len(self.vocab_rev), dtype=float)
                    else:
                        if len(old_vec) < len(self.vocab_rev):
                            new = np.zeros(len(self.vocab_rev), dtype=float)
                            new[:len(old_vec)] = old_vec
                            entry['sem_vec'] = new

            # # create sem_vec for this message
            # if len(self.vocab_rev) == 0:
            #     sem_vec = np.zeros(0, dtype=float)
            # else:
            #     sem_vec = np.zeros(len(self.vocab_rev), dtype=float)
            #     for lab, c in sem_dict.items():
            #         idx = self.vocab.get(lab, None)
            #         if idx is not None:
            #             sem_vec[idx] = float(c)

            # # normalize semantic vector (so magnitude not explode)
            # if sem_vec.size > 0:
            #     sem_vec = normalize_vec(sem_vec)

            # # append to buffer
            # self.buffer.append({
            #     'id': int(kfid),
            #     't': float(t),
            #     'x': float(x),
            #     'y': float(y),
            #     'pose': pose_stamped,
            #     'sem': sem_dict,
            #     'sem_vec': sem_vec
            # })
            # extract clip
            clip_vec = self.extract_clip_vector(msg)

            # build object histogram
            if len(self.vocab_rev) == 0:
                sem_vec = np.zeros(0, dtype=float)
            else:
                sem_vec = np.zeros(len(self.vocab_rev), dtype=float)
                for lab, c in sem_dict.items():
                    idx = self.vocab.get(lab, None)
                    if idx is not None:
                        sem_vec[idx] = float(c)
                if sem_vec.size > 0:
                    sem_vec = normalize_vec(sem_vec)

            self.buffer.append({
                'id': int(kfid),
                't': float(t),
                'x': float(x),
                'y': float(y),
                'pose': pose_stamped,
                'sem': sem_dict,
                'sem_vec': sem_vec,
                'clip_vec': clip_vec
            })
            #print(self.buffer)
            print(sem_vec)
            print(sem_dict)

            # remove old entries outside window_duration_s
            cutoff = t - self.window_duration_s
            print(cutoff)
            while len(self.buffer) > 0 and self.buffer[0]['t'] < cutoff:
                self.buffer.popleft()

        # debug
        self.get_logger().info(f"Received kf {kfid} @ ({x:.2f},{y:.2f}) labels={list(sem_dict.keys())}")

    # ---------------- periodic operations ----------------
    def timer_publish(self):
        now = time.time()
        if now - self.last_cluster_time >= self.clustering_period:
            try:
                self.run_clustering()
            except Exception as e:
                self.get_logger().error(f"Exception during clustering: {e}")
            self.last_cluster_time = now

        # always publish current stable zones at publish_rate
        self.publish_zones()
        if self.visualize:
            self.publish_markers()
            self.publish_zone_perimeters()
            self.publish_zone_keyframe_links()
        

    # ---------------- core: clustering ----------------
    # def build_feature_matrix(self):
    #     """Return (N, D) feature matrix and associated list of buffer indices."""
    #     print("Building feature matrix")
    #     with self.lock:
    #         n = len(self.buffer)
    #         if n == 0:
    #             return None, None

    #         pos = np.zeros((n, 2), dtype=float)
    #         # semantic vectors might be zero-length if no labels observed yet
    #         sem_dim = len(self.vocab_rev)
    #         if sem_dim > 0:
    #             sem = np.zeros((n, sem_dim), dtype=float)
    #         else:
    #             sem = np.zeros((n, 0), dtype=float)

    #         for i, entry in enumerate(self.buffer):
    #             pos[i, 0] = entry['x']
    #             pos[i, 1] = entry['y']
    #             sv = entry.get('sem_vec', None)
    #             if sv is None:
    #                 if sem_dim > 0:
    #                     sem[i, :] = 0.0
    #             else:
    #                 # sv might be smaller if vocabulary was expanded earlier (but we expand on insertion)
    #                 if len(sv) != sem_dim:
    #                     tmp = np.zeros(sem_dim, dtype=float)
    #                     tmp[:len(sv)] = sv
    #                     sem[i, :] = tmp
    #                 else:
    #                     sem[i, :] = sv

    #         # scale semantic by w_sem
    #         if sem_dim > 0:
    #             sem_scaled = self.w_sem * sem
    #             features = np.hstack([pos, sem_scaled])
    #         else:
    #             features = pos  # only spatial

    #         return features, list(range(n))
    def build_feature_matrix(self):
        """Return (N, D) feature matrix and associated list of buffer indices."""
        print("Building feature matrix")
        with self.lock:
            n = len(self.buffer)
            if n == 0:
                return None, None

            pos = np.zeros((n, 2), dtype=float)
            scene_vecs = []

            for i, entry in enumerate(self.buffer):
                pos[i, 0] = entry['x']
                pos[i, 1] = entry['y']
                scene_vecs.append(self.build_scene_vector(entry))

            # pad all scene vectors to same length
            max_dim = max((len(v) for v in scene_vecs), default=0)
            if max_dim == 0:
                scene_mat = np.zeros((n, 0), dtype=float)
            else:
                scene_mat = np.zeros((n, max_dim), dtype=float)
                for i, v in enumerate(scene_vecs):
                    if len(v) > 0:
                        scene_mat[i, :len(v)] = v

            pos_scaled = self.pose_weight * pos
            scene_scaled = self.clip_weight * scene_mat

            features = np.hstack([pos_scaled, scene_scaled])
            return features, list(range(n))

    def run_clustering(self):
        print("running clustering")
        features, idxs = self.build_feature_matrix()
        if features is None or features.shape[0] < self.min_samples:
            self.get_logger().info("Not enough keyframes for clustering")
            return

        # We want eps in meters to apply to spatial part; but since sem scaled in feature vector,
        # we treat eps as a Euclidean threshold in the combined feature space. This is a heuristic.
        eps = self.eps_m

        # Run DBSCAN
        print("running dbscan")
        try:
            model = DBSCAN(eps=eps, min_samples=self.min_samples, metric='euclidean', n_jobs=1)
            labels = model.fit_predict(features)
        except Exception as e:
            # fallback: if DBSCAN fails for some reason, log and return
            self.get_logger().error(f"DBSCAN failed: {e}")
            return

        unique_labels = set(labels.tolist())
        if -1 in unique_labels:
            unique_labels.remove(-1)  # -1 is noise
        # collect clusters
        clusters = []
        with self.lock:
            for lbl in unique_labels:
                members_idx = [i for (i, lab) in enumerate(labels) if lab == lbl]
                members = [self.buffer[i] for i in members_idx]
                clusters.append((lbl, members_idx, members))
                print("Clusters appended with unique labels %d", lbl)

        # analyze clusters for potential 
        for (lbl, members_idx, members) in clusters:
            # kf_ids = [int(m['id']) for m in members]
            # nx = np.mean([m['x'] for m in members])
            # ny = np.mean([m['y'] for m in members])
            # times = [m['t'] for m in members]
            # duration = max(times) - min(times) if len(times) >= 2 else 0.0

            # # semantic signature and purity
            # sem_dim = len(self.vocab_rev)
            # sig = np.zeros(sem_dim, dtype=float) if sem_dim > 0 else np.zeros(0, dtype=float)
            # for m in members:
            #     if sem_dim > 0:
            #         v = m.get('sem_vec', np.zeros(sem_dim))
            #         if len(v) < sem_dim:
            #             tmp = np.zeros(sem_dim)
            #             tmp[:len(v)] = v
            #             v = tmp
            #         sig += v
            # if sem_dim > 0:
            #     sig = normalize_vec(sig)
            #     purity = float(np.max(sig)) if sig.size > 0 else 0.0
            #     top_idx = int(np.argmax(sig)) if sig.size > 0 else None
            #     top_label = self.vocab_rev[top_idx] if top_idx is not None and top_idx < len(self.vocab_rev) else ''
            #     top_conf = float(sig[top_idx]) if top_idx is not None else 0.0
            # else:
            #     purity = 0.0
            #     top_label = ''
            #     top_conf = 0.0
            kf_ids = [int(m['id']) for m in members]
            nx = np.mean([m['x'] for m in members])
            ny = np.mean([m['y'] for m in members])
            times = [m['t'] for m in members]
            duration = max(times) - min(times) if len(times) >= 2 else 0.0

            scene_vecs = []
            for m in members:
                scene_vec = self.get_scene_vector(m)
                if scene_vec.size > 0:
                    scene_vecs.append(scene_vec)

            if len(scene_vecs) > 0:
                scene_vecs = np.asarray(scene_vecs, dtype=float)
                if scene_vecs.ndim == 1:
                    scene_vecs = scene_vecs.reshape(-1, 1)

                scene_sig = l2_normalize(np.mean(scene_vecs, axis=0))
                cohesion = float(np.mean([cosine_similarity(v, scene_sig) for v in scene_vecs]))
            else:
                scene_sig = np.zeros(0, dtype=float)
                cohesion = 0.0

            # compute zone confidence (simple heuristic)
            # membership_score = min(1.0, len(kf_ids) / max(1.0, float(self.min_kfs_confirm)))
            # zone_conf = float(self.zone_confidence_base * 0.5 + 0.5 * (0.5 * purity + 0.5 * membership_score))
            membership_score = min(1.0, len(kf_ids) / max(1.0, float(self.min_kfs_confirm)))
            zone_conf = float(self.zone_confidence_base * 0.5 + 0.5 * (0.5 * cohesion + 0.5 * membership_score))

            confirmed = False
            if len(kf_ids) >= self.min_kfs_confirm and cohesion >= self.min_clip_cohesion:
                confirmed = True
            elif duration >= self.temporal_confirm_s and cohesion >= 0.30:
                confirmed = True

            # decide tentative/confirmed
            # confirmed = False
            # if len(kf_ids) >= self.min_kfs_confirm and purity >= self.purity_thresh:
            #     confirmed = True
            # elif duration >= self.temporal_confirm_s and purity >= 0.3:
            #     confirmed = True

            # either create new zone or update existing one (merge if necessary)
            # if confirmed:
            #     self.create_or_update_zone(kf_ids, (nx, ny), sig, top_label, top_conf, zone_conf)
            top_label = ""
            top_conf = 0.0

            # if len(sem_dict) > 0:
            #     top_label = max(sem_dict.items(), key=lambda kv: kv[1])[0]
            #     top_conf = float(max(sem_dict.values()))
            if confirmed:
                self.create_or_update_zone(kf_ids, (nx, ny), scene_sig, top_label, top_conf, zone_conf)
            else:
                # keep as candidate - maybe store for future confirmation
                # we simply log
                self.get_logger().info(f"Candidate cluster lbl={lbl} size={len(kf_ids)}duration={duration:.1f}s not confirmed")

    
    # ---------------- zone management ----------------
    # def create_or_update_zone(self, kf_ids, centroid_xy, sem_sig, top_label, top_conf, conf):
    #     """Create a new zone or merge into an existing one when close + semantically similar."""
    #     # search for mergeable zone
    #     print("Creating or updating zone")
    #     merged_zone_id = None
    #     print("Printing Zones")
    #     print(self.zones.items())
    #     for zid, z in self.zones.items():
    #         zx, zy = z['centroid']
    #         d = math.hypot(zx - centroid_xy[0], zy - centroid_xy[1])
    #         sem_sim = cosine_similarity(z['sem_sig'], sem_sig) if (z['sem_sig'].size>0 and sem_sig.size>0) else 0.0
    #         print("Printing Semantic ka samaaan", sem_sim, sem_sig)
    #         if d <= self.merge_dist_m and sem_sim >= self.merge_semantic_thresh:
    #             merged_zone_id = zid
    #             break

    #     if merged_zone_id is None:
    #         # create new zone
    #         zid = self.next_zone_id
    #         self.next_zone_id += 1
    #         zone = {
    #             'id': zid,
    #             'members': set(kf_ids),
    #             'centroid': (float(centroid_xy[0]), float(centroid_xy[1])),
    #             'sem_sig': sem_sig.copy() if hasattr(sem_sig, 'copy') else np.array(sem_sig),
    #             'top_label': top_label,
    #             'top_conf': float(top_conf),
    #             'confidence': float(conf),
    #             'state': 'confirmed',
    #             'last_updated': time.time(),
    #             'version': 1
    #         }
    #         self.zones[zid] = zone
    #         self.get_logger().info(f"Created zone {zid} center=({zone['centroid'][0]:.2f},{zone['centroid'][1]:.2f}) size={len(kf_ids)} label={top_label} conf={conf:.2f}")
    #     else:
    #         # merge into existing zone
    #         z = self.zones[merged_zone_id]
    #         old_members = set(z['members'])
    #         z['members'].update(kf_ids)
    #         # recompute centroid as mean of positions of members currently in buffer if available
    #         with self.lock:
    #             member_positions = []
    #             for entry in self.buffer:
    #                 if entry['id'] in z['members']:
    #                     member_positions.append((entry['x'], entry['y']))
    #         if len(member_positions) > 0:
    #             xs = [p[0] for p in member_positions]
    #             ys = [p[1] for p in member_positions]
    #             print(xs,ys)
    #             z['centroid'] = (float(np.mean(xs)), float(np.mean(ys)))
    #         # merge semantic signature by weighted average (old sig + new sig)
    #         try:
    #             z_old_sig = z['sem_sig']
    #             if z_old_sig.size == 0:
    #                 z['sem_sig'] = sem_sig.copy()
    #             else:
    #                 z['sem_sig'] = normalize_vec( (z_old_sig + sem_sig) / 2.0 )
    #         except Exception:
    #             z['sem_sig'] = sem_sig.copy()

    #         z['top_label'] = top_label if top_conf >= z.get('top_conf', 0.0) else z.get('top_label', top_label)
    #         z['top_conf'] = max(top_conf, z.get('top_conf', top_conf))
    #         z['confidence'] = min(1.0, 0.9 * z.get('confidence', 0.5) + 0.1 * conf)
    #         z['last_updated'] = time.time()
    #         z['version'] += 1
    #         self.get_logger().info(f"Merged into zone {merged_zone_id} => new size={len(z['members'])} centroid=({z['centroid'][0]:.2f},{z['centroid'][1]:.2f})")
    def pad_vec_to_len(self, vec, length):
        vec = np.asarray(vec, dtype=float).ravel()
        if len(vec) == length:
            return vec.copy()
        out = np.zeros(length, dtype=float)
        out[:min(len(vec), length)] = vec[:min(len(vec), length)]
        return out
    
    def extract_clip_vector(self, msg):
        """
        Extract CLIP embedding from message.
        Accepts multiple possible field names.
        """
        for field in ("clip_embedding", "scene_embedding", "embedding", "clip_feat"):
            raw = getattr(msg, field, None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float).ravel()
            if arr.size > 0:
                return l2_normalize(arr)
        return np.zeros(0, dtype=float)


    def build_scene_vector(self, entry):
        """
        Hybrid semantic vector:
        [ CLIP embedding || object histogram ]
        Both are used together, not as fallback.
        """
        clip_vec = np.asarray(entry.get("clip_vec", np.zeros(0, dtype=float)), dtype=float).ravel()
        sem_vec = np.asarray(entry.get("sem_vec", np.zeros(0, dtype=float)), dtype=float).ravel()

        # pad object histogram to current vocab size
        if len(self.vocab_rev) > 0 and sem_vec.size != len(self.vocab_rev):
            sem_vec = self.pad_vec_to_len(sem_vec, len(self.vocab_rev))

        parts = []

        if clip_vec.size > 0:
            parts.append(1.0 * l2_normalize(clip_vec))

        if sem_vec.size > 0:
            parts.append(1.0 * normalize_vec(sem_vec))

        if len(parts) == 0:
            return np.zeros(0, dtype=float)

        return np.hstack(parts)
    
    def get_scene_vector(self, entry):
        """
        Combined semantic vector used for clustering and zone updates.

        If CLIP is available:
        scene_vec = [clip_embedding || object_histogram]

        If CLIP is not available:
        scene_vec = object_histogram
        """
        sem_vec = entry.get('sem_vec', np.zeros(0, dtype=float))
        sem_vec = np.asarray(sem_vec, dtype=float).ravel()

        # keep object histogram consistent with current vocabulary size
        if len(self.vocab_rev) > 0 and sem_vec.size != len(self.vocab_rev):
            sem_vec = self.pad_vec_to_len(sem_vec, len(self.vocab_rev))

        if self.use_clip_embedding:
            clip_vec = entry.get('clip_vec', np.zeros(0, dtype=float))
            clip_vec = np.asarray(clip_vec, dtype=float).ravel()

            if clip_vec.size == 0 and self.clip_dim_hint is not None:
                clip_vec = np.zeros(self.clip_dim_hint, dtype=float)
            elif clip_vec.size > 0:
                clip_vec = l2_normalize(clip_vec)
                if self.clip_dim_hint is None:
                    self.clip_dim_hint = clip_vec.size
                elif clip_vec.size != self.clip_dim_hint:
                    clip_vec = self.pad_vec_to_len(clip_vec, self.clip_dim_hint)

            if clip_vec.size > 0:
                return np.hstack([clip_vec, sem_vec])

        return sem_vec
    
    # def create_or_update_zone(self, kf_ids, centroid_xy, sem_sig, top_label, top_conf, conf):
        """
        Create a new zone or merge into an existing one when close + semantically similar.
        Version is incremented only when the zone content actually changes.
        """
        print("Creating or updating zone")
        now = time.time()

        # make sure inputs are numpy arrays
        if not isinstance(sem_sig, np.ndarray):
            sem_sig = np.array(sem_sig, dtype=float)

        merged_zone_id = None
        merged_zone_changed = False

        print("Printing Zones")
        print(self.zones.items())

        for zid, z in self.zones.items():
            zx, zy = z['centroid']
            d = math.hypot(zx - centroid_xy[0], zy - centroid_xy[1])

            z_sig = z['sem_sig']
            if not isinstance(z_sig, np.ndarray):
                z_sig = np.array(z_sig, dtype=float)

            target_len = max(len(z_sig), len(sem_sig))
            z_sig_padded = self.pad_vec_to_len(z_sig, target_len)
            sem_sig_padded = self.pad_vec_to_len(sem_sig, target_len)
            sem_sim = cosine_similarity(z_sig_padded, sem_sig_padded) if (z_sig.size > 0 and sem_sig.size > 0) else 0.0
            print("Printing Semantic ka samaaan", sem_sim, sem_sig)

            if d <= self.merge_dist_m and sem_sim >= self.merge_semantic_thresh:
                merged_zone_id = zid
                break

        if merged_zone_id is None:
            # create new zone
            zid = self.next_zone_id
            self.next_zone_id += 1

            zone = {
                'id': zid,
                'members': set(kf_ids),
                'centroid': (float(centroid_xy[0]), float(centroid_xy[1])),
                'sem_sig': sem_sig.copy(),
                'top_label': top_label,
                'top_conf': float(top_conf),
                'confidence': float(conf),
                'state': 'confirmed',
                'last_updated': now,
                'version': 1,
                'dirty': True,
                'last_published_version': 0,
                'action': 0,  # CREATE
            }
            self.zones[zid] = zone

            self.get_logger().info(
                f"Created zone {zid} center=({zone['centroid'][0]:.2f},{zone['centroid'][1]:.2f}) "
                f"size={len(kf_ids)} label={top_label} conf={conf:.2f}"
            )
            return zone

        # merge into existing zone
        z = self.zones[merged_zone_id]
        old_members = set(z['members'])
        old_centroid = z['centroid']
        old_sig = z['sem_sig'].copy() if isinstance(z['sem_sig'], np.ndarray) else np.array(z['sem_sig'], dtype=float)
        old_top_label = z.get('top_label', '')
        old_top_conf = float(z.get('top_conf', 0.0))
        old_conf = float(z.get('confidence', 0.0))

        # update members
        z['members'].update(kf_ids)

        # recompute centroid from buffered members that belong to this zone
        with self.lock:
            member_positions = []
            for entry in self.buffer:
                if entry['id'] in z['members']:
                    member_positions.append((entry['x'], entry['y']))

        if len(member_positions) > 0:
            xs = [p[0] for p in member_positions]
            ys = [p[1] for p in member_positions]
            print(xs, ys)
            new_centroid = (float(np.mean(xs)), float(np.mean(ys)))
        else:
            new_centroid = old_centroid

        # merge semantic signature
        # if old_sig.size == 0:
        #     new_sig = sem_sig.copy()
        # elif sem_sig.size == 0:
        #     new_sig = old_sig.copy()
        # else:
        #     new_sig = normalize_vec((old_sig + sem_sig) / 2.0)
        # merge semantic signature
        old_sig = np.asarray(old_sig, dtype=float).ravel()
        sem_sig = np.asarray(sem_sig, dtype=float).ravel()

        target_len = max(len(old_sig), len(sem_sig))
        old_sig = self.pad_vec_to_len(old_sig, target_len)
        sem_sig = self.pad_vec_to_len(sem_sig, target_len)

        if old_sig.size == 0:
            new_sig = sem_sig.copy()
        elif sem_sig.size == 0:
            new_sig = old_sig.copy()
        else:
            new_sig = normalize_vec(old_sig + sem_sig)

        # update best label
        new_top_label = top_label if top_conf >= old_top_conf else old_top_label
        new_top_conf = max(top_conf, old_top_conf)

        # confidence update
        new_conf = min(1.0, 0.9 * old_conf + 0.1 * conf)

        # decide whether anything actually changed enough
        centroid_shift = math.hypot(new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
        members_changed = (z['members'] != old_members)
        sig_changed = (old_sig.size != new_sig.size) or (old_sig.size > 0 and np.linalg.norm(old_sig - new_sig) > 1e-6)
        label_changed = (new_top_label != old_top_label)
        conf_changed = abs(new_conf - old_conf) > self.zone_confidence_update_thresh
        centroid_changed = centroid_shift > self.zone_centroid_update_thresh

        merged_zone_changed = members_changed or sig_changed or label_changed or conf_changed or centroid_changed

        # apply updates
        z['centroid'] = new_centroid
        z['sem_sig'] = new_sig
        z['top_label'] = new_top_label
        z['top_conf'] = float(new_top_conf)
        z['confidence'] = float(new_conf)
        z['last_updated'] = now
        z['action'] = 1  # UPDATE

        if merged_zone_changed:
            z['version'] += 1
            z['dirty'] = True
            self.get_logger().info(
                f"Merged into zone {merged_zone_id} => new size={len(z['members'])} "
                f"centroid=({z['centroid'][0]:.2f},{z['centroid'][1]:.2f})"
            )
        else:
            z['dirty'] = False

        return z
    def create_or_update_zone(self, kf_ids, centroid_xy, sem_sig, top_label, top_conf, conf):
        """Create a new zone or merge into an existing one when close + semantically similar."""
        print("Creating or updating zone")
        print("Printing Zones")
        print(self.zones.items())

        merged_zone_id = None

        sem_sig = np.asarray(sem_sig, dtype=float).ravel()

        for zid, z in self.zones.items():
            zx, zy = z['centroid']
            d = math.hypot(zx - centroid_xy[0], zy - centroid_xy[1])

            z_sig = np.asarray(z['sem_sig'], dtype=float).ravel()

            target_len = max(len(z_sig), len(sem_sig))
            z_sig = self.pad_vec_to_len(z_sig, target_len)
            sem_sig_padded = self.pad_vec_to_len(sem_sig, target_len)

            sem_sim = cosine_similarity(z_sig, sem_sig_padded) if (z_sig.size > 0 and sem_sig_padded.size > 0) else 0.0
            print("Printing Semantic ka samaaan", sem_sim, sem_sig_padded)

            if d <= self.merge_dist_m and sem_sim >= self.clip_merge_thresh:
                merged_zone_id = zid
                break

        if merged_zone_id is None:
            zid = self.next_zone_id
            self.next_zone_id += 1
            zone = {
                'id': zid,
                'members': set(kf_ids),
                'centroid': (float(centroid_xy[0]), float(centroid_xy[1])),
                'sem_sig': sem_sig.copy(),
                'top_label': top_label,
                'top_conf': float(top_conf),
                'confidence': float(conf),
                'state': 'confirmed',
                'last_updated': time.time(),
                'version': 1,
                'dirty': True,
                'last_published_version': 0,
                'action': 0,
            }
            self.zones[zid] = zone
            self.get_logger().info(
                f"Created zone {zid} center=({zone['centroid'][0]:.2f},{zone['centroid'][1]:.2f}) "
                f"size={len(kf_ids)} label={top_label} conf={conf:.2f}"
            )
        else:
            z = self.zones[merged_zone_id]
            old_members = set(z['members'])
            old_centroid = z['centroid']
            old_sig = np.asarray(z['sem_sig'], dtype=float).ravel()
            old_top_label = z.get('top_label', '')
            old_top_conf = float(z.get('top_conf', 0.0))
            old_conf = float(z.get('confidence', 0.0))

            z['members'].update(kf_ids)

            with self.lock:
                member_positions = []
                for entry in self.buffer:
                    if entry['id'] in z['members']:
                        member_positions.append((entry['x'], entry['y']))

            if len(member_positions) > 0:
                xs = [p[0] for p in member_positions]
                ys = [p[1] for p in member_positions]
                print(xs, ys)
                new_centroid = (float(np.mean(xs)), float(np.mean(ys)))
            else:
                new_centroid = old_centroid

            new_sig = np.asarray(sem_sig, dtype=float).ravel()
            target_len = max(len(old_sig), len(new_sig))
            old_sig = self.pad_vec_to_len(old_sig, target_len)
            new_sig = self.pad_vec_to_len(new_sig, target_len)

            if old_sig.size == 0:
                merged_sig = new_sig.copy()
            elif new_sig.size == 0:
                merged_sig = old_sig.copy()
            else:
                merged_sig = l2_normalize((1.0 - self.clip_update_beta) * old_sig +
                                        self.clip_update_beta * new_sig)

            new_top_label = top_label if top_conf >= old_top_conf else old_top_label
            new_top_conf = max(top_conf, old_top_conf)
            new_conf = min(1.0, 0.9 * old_conf + 0.1 * conf)

            centroid_shift = math.hypot(new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
            members_changed = (z['members'] != old_members)
            sig_changed = (np.linalg.norm(old_sig - merged_sig) > 1e-6)
            label_changed = (new_top_label != old_top_label)
            conf_changed = abs(new_conf - old_conf) > self.zone_confidence_update_thresh
            centroid_changed = centroid_shift > self.zone_centroid_update_thresh

            z['centroid'] = new_centroid
            z['sem_sig'] = merged_sig
            z['top_label'] = new_top_label
            z['top_conf'] = float(new_top_conf)
            z['confidence'] = float(new_conf)
            z['last_updated'] = time.time()
            z['action'] = 1

            if members_changed or sig_changed or label_changed or conf_changed or centroid_changed:
                z['version'] += 1
                z['dirty'] = True
                self.get_logger().info(
                    f"Merged into zone {merged_zone_id} => new size={len(z['members'])} "
                    f"centroid=({z['centroid'][0]:.2f},{z['centroid'][1]:.2f})"
                )
            else:
                z['dirty'] = False

    def convex_hull_2d(self, points):
        """
        points: list of (x, y)
        returns: hull points in CCW order
        """
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

    def make_cluster_perimeter_marker(self, cluster_points, marker_id, zone_id, frame_id="map"):
        """
        Create a LINE_STRIP marker showing the perimeter of the cluster.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"zone_perimeter_{zone_id}"
        marker.id = int(marker_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.06  # line width

        # color depends on zone id, same pattern as your centroid color logic
        cid = (zone_id * 37) % 255
        marker.color.r = ((cid * 97) % 255) / 255.0
        marker.color.g = ((cid * 71) % 255) / 255.0
        marker.color.b = ((cid * 53) % 255) / 255.0
        marker.color.a = 0.95

        if len(cluster_points) == 0:
            marker.action = Marker.DELETE
            return marker

        if len(cluster_points) == 1:
            # draw a tiny square around the single point
            x, y = cluster_points[0]
            s = 0.2
            pts = [
                (x - s, y - s),
                (x + s, y - s),
                (x + s, y + s),
                (x - s, y + s),
                (x - s, y - s),
            ]
        elif len(cluster_points) == 2:
            # draw a small rectangle around two points
            xs = [p[0] for p in cluster_points]
            ys = [p[1] for p in cluster_points]
            pad = 0.15
            min_x, max_x = min(xs) - pad, max(xs) + pad
            min_y, max_y = min(ys) - pad, max(ys) + pad
            pts = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            ]
        else:
            hull = self.convex_hull_2d(cluster_points)
            if len(hull) < 3:
                xs = [p[0] for p in cluster_points]
                ys = [p[1] for p in cluster_points]
                pad = 0.15
                min_x, max_x = min(xs) - pad, max(xs) + pad
                min_y, max_y = min(ys) - pad, max(ys) + pad
                pts = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                    (min_x, min_y),
                ]
            else:
                pts = hull + [hull[0]]  # close loop

        for x, y in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        return marker

    def publish_zone_perimeters(self):
        """
        Publish one perimeter marker per confirmed zone.
        """
        ma = MarkerArray()
        marker_id = 0

        with self.lock:
            zones_snapshot = list(self.zones.items())
            buffer_snapshot = list(self.buffer)

        for zid, z in zones_snapshot:
            # collect member keyframe positions currently in buffer
            member_points = []
            member_ids = set(z.get('members', set()))

            for entry in buffer_snapshot:
                if int(entry['id']) in member_ids:
                    member_points.append((float(entry['x']), float(entry['y'])))

            marker = self.make_cluster_perimeter_marker(
                member_points,
                marker_id=marker_id,
                zone_id=zid,
                frame_id="map"
            )
            ma.markers.append(marker)
            marker_id += 1

        if self.perimeter_pub is not None:
            self.perimeter_pub.publish(ma)


    def publish_zone_keyframe_links(self):
        ma = MarkerArray()
        marker_id = 0

        with self.lock:
            zones_snapshot = list(self.zones.items())
            buffer_snapshot = list(self.buffer)

        # build a quick lookup from keyframe id -> (x, y)
        kf_pos = {}
        for entry in buffer_snapshot:
            try:
                kf_pos[int(entry['id'])] = (float(entry['x']), float(entry['y']))
            except Exception:
                continue

        for zid, z in zones_snapshot:
            centroid = z.get('centroid', None)
            if centroid is None:
                continue

            cx, cy = float(centroid[0]), float(centroid[1])
            member_ids = set(z.get('members', set()))

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f'zone_links_{zid}'
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.03

            # same color logic as zone id
            cid = (zid * 37) % 255
            marker.color.r = ((cid * 97) % 255) / 255.0
            marker.color.g = ((cid * 71) % 255) / 255.0
            marker.color.b = ((cid * 53) % 255) / 255.0
            marker.color.a = 0.8

            for kf_id in member_ids:
                if kf_id not in kf_pos:
                    continue

                kx, ky = kf_pos[kf_id]

                p1 = Point()
                p1.x = cx
                p1.y = cy
                p1.z = 0.08

                p2 = Point()
                p2.x = kx
                p2.y = ky
                p2.z = 0.08

                marker.points.append(p1)
                marker.points.append(p2)

            if len(marker.points) > 0:
                ma.markers.append(marker)
                marker_id += 1

        if self.zone_links_pub is not None:
            self.zone_links_pub.publish(ma)

    # ---------------- publish zones as messages ----------------
    # def publish_zones(self):
    #     """Publish ZoneMsg for each confirmed zone."""
    #     print("Publishing zones")
    #     now = self.get_clock().now().to_msg()
    #     for zid, z in list(self.zones.items()):
    #         try:
    #             zm = ZoneMsg()
    #             # put header if exists
    #             try:
    #                 zm.header = Header()
    #                 zm.header.stamp = now
    #             except Exception:
    #                 pass
    #             # set zone id
    #             if hasattr(zm, 'zone_id'):
    #                 zm.zone_id = int(zid)
    #             # keyframe ids
    #             if hasattr(zm, 'keyframe_ids'):
    #                 zm.keyframe_ids = list(z['members'])
    #             else:
    #                 try:
    #                     zm.keyframes = list(z['members'])
    #                 except Exception:
    #                     pass
    #             # top_labels / confidences
    #             if hasattr(zm, 'top_labels'):
    #                 zm.top_labels = [z.get('top_label', '')]
    #             if hasattr(zm, 'top_label_confidences'):
    #                 zm.top_label_confidences = [float(z.get('top_conf', 0.0))]
    #             if hasattr(zm, 'confidence'):
    #                 zm.confidence = float(z.get('confidence', 0.0))
    #             # supporting_planes optional - left empty
    #             if hasattr(zm, 'version'):
    #                 zm.version = int(z.get('version', 0))
    #             # centroid if present
    #             if hasattr(zm, 'centroid') or hasattr(zm, 'pose'):
    #                 try:
    #                     # some Zone.msg may have centroid or Pose; try to set if present
    #                     if hasattr(zm, 'centroid'):
    #                         zm.centroid.position.x = float(z['centroid'][0])
    #                         zm.centroid.position.y = float(z['centroid'][1])
    #                         zm.centroid.z = 0.0
    #                     if hasattr(zm, 'pose'):
    #                         p = PoseStamped()
    #                         p.header.stamp = now
    #                         p.header.frame_id = 'map'
    #                         p.pose.position.x = float(z['centroid'][0])
    #                         p.pose.position.y = float(z['centroid'][1])
    #                         p.pose.position.z = 0.0
    #                         zm.pose = p
    #                 except Exception:
    #                     pass

    #             # publish
    #             self.zone_pub.publish(zm)
    #         except Exception as e:
    #             self.get_logger().error(f"Error publishing zone {zid}: {e}")
    def publish_zones(self):
        """Publish only dirty zones."""
        print("Publishing zones")

        now = self.get_clock().now().to_msg()

        # snapshot dirty zones only
        with self.lock:
            dirty_zones = [(zid, z) for zid, z in self.zones.items() if z.get('dirty', False)]

        for zid, z in dirty_zones:
            try:
                zm = ZoneMsg()

                # header
                try:
                    zm.header = Header()
                    zm.header.stamp = now
                    zm.header.frame_id = 'map'
                except Exception:
                    pass

                # zone id
                if hasattr(zm, 'zone_id'):
                    zm.zone_id = int(zid)

                # keyframe ids
                if hasattr(zm, 'keyframe_ids'):
                    zm.keyframe_ids = list(sorted(z['members']))
                else:
                    try:
                        zm.keyframes = list(sorted(z['members']))
                    except Exception:
                        pass

                # semantic fields
                if hasattr(zm, 'top_labels'):
                    zm.top_labels = [z.get('top_label', '')]
                if hasattr(zm, 'top_label_confidences'):
                    zm.top_label_confidences = [float(z.get('top_conf', 0.0))]
                if hasattr(zm, 'confidence'):
                    zm.confidence = float(z.get('confidence', 0.0))

                # version
                if hasattr(zm, 'version'):
                    zm.version = int(z.get('version', 0))

                # centroid
                if hasattr(zm, 'centroid') or hasattr(zm, 'pose'):
                    try:
                        if hasattr(zm, 'centroid'):
                            zm.centroid.position.x = float(z['centroid'][0])
                            zm.centroid.position.y = float(z['centroid'][1])
                            zm.centroid.position.z = 0.0
                            zm.centroid.orientation.w = 1.0

                        if hasattr(zm, 'pose'):
                            p = PoseStamped()
                            p.header.stamp = now
                            p.header.frame_id = 'map'
                            p.pose.position.x = float(z['centroid'][0])
                            p.pose.position.y = float(z['centroid'][1])
                            p.pose.position.z = 0.0
                            p.pose.orientation.w = 1.0
                            zm.pose = p
                    except Exception:
                        pass

                self.zone_pub.publish(zm)

                # clear dirty after successful publish
                with self.lock:
                    if zid in self.zones:
                        self.zones[zid]['dirty'] = False
                        self.zones[zid]['last_published_version'] = z.get('version', 0)

            except Exception as e:
                self.get_logger().error(f"Error publishing zone {zid}: {e}")

    # ---------------- RViz visualization ----------------
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
            m.pose.position.x = float(z['centroid'][0])
            m.pose.position.y = float(z['centroid'][1])
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
            mt.pose.position.x = float(z['centroid'][0])
            mt.pose.position.y = float(z['centroid'][1])
            mt.pose.position.z = 0.6
            mt.scale.z = 0.25
            mt.color.a = 1.0
            mt.color.r = 1.0
            mt.color.g = 1.0
            mt.color.b = 1.0
            mt.text = f"Z{zid}:{z.get('top_label','?')}({len(z['members'])})"
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
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()