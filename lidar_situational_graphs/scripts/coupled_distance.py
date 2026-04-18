#!/usr/bin/env python3
"""
zones_cluster_node.py

Coupled Distance DBSCAN-based clustering node:
 - consumes KeyframeSemantic messages (pose + objects + confidences + CLIP)
 - maintains sliding window buffer of recent keyframes
 - computes a coupled distance metric: D_coupled = D_spatial * (1 + lambda * D_cosine(semantics))
 - runs DBSCAN periodically using the precomputed distance matrix
 - promotes clusters to tentative zones and confirms them when stable
 - merges close & semantically similar zones
 - publishes Zone messages (lidar_situational_graphs/msg/Zone)
 - publishes RViz markers (visualization_msgs/MarkerArray) for debugging

This node is robust to unknown vocabulary and strictly preserves physical topology 
by evaluating spatial distance natively in meters.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import numpy as np
import math
import time
from collections import deque
from threading import Lock

# clustering
try:
    from sklearn.cluster import DBSCAN
except Exception as e:
    raise RuntimeError("scikit-learn is required. Install with `pip install scikit-learn`") from e

try:
    from scipy.spatial.distance import pdist, squareform
except Exception as e:
    raise RuntimeError("scipy is required. Install with `pip install scipy`") from e

# ROS messages (adjust package name if different)
from situational_graphs_msgs.msg import KeyframeSemantic, Zone
from situational_graphs_msgs.msg import Zone as ZoneMsg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header, ColorRGBA


# ---------------- Global Helper Utils ----------------

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
        self.declare_parameter('clustering_period', 2.0)
        self.declare_parameter('window_duration_s', 150.0)
        self.declare_parameter('min_samples', 3)
        self.declare_parameter('eps_m', 2.2)  # Now strictly evaluated in METERS
        self.declare_parameter('w_sem', 2.0)  # Legacy semantic weight
        self.declare_parameter('min_kfs_confirm', 3)
        self.declare_parameter('purity_thresh', 0.6)
        self.declare_parameter('temporal_confirm_s', 8.0)
        self.declare_parameter('merge_dist_m', 3.0) 
        self.declare_parameter('merge_semantic_thresh', 0.7)
        self.declare_parameter('zone_confidence_base', 0.5)
        self.declare_parameter('visualize', True)
        self.declare_parameter('marker_ns', 'zones')

        # CLIP Parameters
        self.declare_parameter('use_clip_embedding', True)
        self.declare_parameter('pose_weight', 1.0)
        self.declare_parameter('clip_weight', 2.0) # Acts as lambda for coupled distance penalty
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

        # CLIP
        self.use_clip_embedding = bool(self.get_parameter('use_clip_embedding').value)
        self.pose_weight = float(self.get_parameter('pose_weight').value)
        self.clip_weight = float(self.get_parameter('clip_weight').value)
        self.min_clip_cohesion = float(self.get_parameter('min_clip_cohesion').value)
        self.clip_merge_thresh = float(self.get_parameter('clip_merge_thresh').value)
        self.clip_update_beta = float(self.get_parameter('clip_update_beta').value)
        self.clip_dim_hint = None

        # ---- runtime state ----
        self.lock = Lock()
        self.buffer = deque()  
        self.vocab = {}  
        self.vocab_rev = []  
        self.next_zone_id = 1
        self.zones = {}  

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
        try:
            kfid = int(getattr(msg, 'keyframe_id', -1))
        except Exception:
            kfid = -1
        if kfid < 0:
            kfid = int(time.time() * 1000)  
        t = self.get_clock().now().nanoseconds / 1e9

        pose_stamped = None
        try:
            if hasattr(msg, 'pose') and msg.pose is not None:
                pose_stamped = msg.pose
            elif hasattr(msg, 'header') and hasattr(msg, 'pose'): 
                pose_stamped = msg.pose
        except Exception:
            pose_stamped = None

        x = y = 0.0
        if pose_stamped is not None:
            try:
                x = float(pose_stamped.position.x)
                y = float(pose_stamped.position.y)
            except Exception:
                x = y = 0.0

        labels = []
        confs = []
        try:
            labels = list(getattr(msg, 'objects', []))
            confs = list(getattr(msg, 'object_confidence', []))
        except Exception:
            pass

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
                    old_vec = entry.get('sem_vec', None)
                    if old_vec is None:
                        entry['sem_vec'] = np.zeros(len(self.vocab_rev), dtype=float)
                    else:
                        if len(old_vec) < len(self.vocab_rev):
                            new = np.zeros(len(self.vocab_rev), dtype=float)
                            new[:len(old_vec)] = old_vec
                            entry['sem_vec'] = new

            clip_vec = self.extract_clip_vector(msg)

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

            cutoff = t - self.window_duration_s
            while len(self.buffer) > 0 and self.buffer[0]['t'] < cutoff:
                self.buffer.popleft()

    def timer_publish(self):
        now = time.time()
        if now - self.last_cluster_time >= self.clustering_period:
            try:
                self.run_clustering()
            except Exception as e:
                self.get_logger().error(f"Exception during clustering: {e}")
            self.last_cluster_time = now

        self.publish_zones()
        if self.visualize:
            self.publish_markers()
            self.publish_zone_perimeters()
            self.publish_zone_keyframe_links()

    # ---------------- Feature & Vector Utilities ----------------
    def pad_vec_to_len(self, vec, length):
        vec = np.asarray(vec, dtype=float).ravel()
        if len(vec) == length:
            return vec.copy()
        out = np.zeros(length, dtype=float)
        out[:min(len(vec), length)] = vec[:min(len(vec), length)]
        return out
    
    def extract_clip_vector(self, msg):
        for field in ("clip_embedding", "scene_embedding", "embedding", "clip_feat"):
            raw = getattr(msg, field, None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float).ravel()
            if arr.size > 0:
                return l2_normalize(arr)
        return np.zeros(0, dtype=float)

    def get_scene_vector(self, entry):
        """
        Combined semantic vector used for clustering and zone updates.
        If CLIP is available: scene_vec = [clip_embedding || object_histogram]
        If CLIP is not available: scene_vec = object_histogram
        """
        sem_vec = entry.get('sem_vec', np.zeros(0, dtype=float))
        sem_vec = np.asarray(sem_vec, dtype=float).ravel()

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

    # ---------------- Core: Information-Theoretic Coupled Clustering ----------------
    def run_clustering(self):
        print("Running coupled Information-Theoretic clustering")
        
        # 1. Extract raw data safely under lock
        with self.lock:
            n = len(self.buffer)
            if n < self.min_samples:
                return

            pos = np.zeros((n, 2), dtype=float)
            scene_vecs = []

            for i, entry in enumerate(self.buffer):
                pos[i, 0] = entry['x']
                pos[i, 1] = entry['y']
                scene_vecs.append(self.get_scene_vector(entry))

            # Pad scene vectors to uniform length for distance calculation
            max_dim = max((len(v) for v in scene_vecs), default=0)
            if max_dim == 0:
                scene_mat = np.zeros((n, 0), dtype=float)
            else:
                scene_mat = np.zeros((n, max_dim), dtype=float)
                for i, v in enumerate(scene_vecs):
                    if len(v) > 0:
                        scene_mat[i, :len(v)] = v

            # Create a shallow copy of the buffer to use outside the lock
            buffer_list = list(self.buffer)

        # 2. Compute pairwise spatial distances (Strictly in METERS)
        # Apply pose_weight only if you explicitly want to distort scale, otherwise physical distance is best
        dist_spatial = pdist(pos * self.pose_weight, metric='euclidean')

        # 3. Compute pairwise semantic/appearance distances
        if max_dim > 0:
            # Using 'cosine' distance for CLIP + Histograms. 
            # Identical = 0.0, Orthogonal = 1.0
            dist_sem = pdist(scene_mat, metric='cosine')
            # Catch zero-vectors to avoid NaNs
            dist_sem = np.nan_to_num(dist_sem, nan=0.0)
        else:
            dist_sem = np.zeros_like(dist_spatial)

        # 4. Combine using the Coupled Mathematical Metric
        # self.clip_weight acts as lambda penalty factor
        lambda_sem = self.clip_weight
        dist_coupled_condensed = dist_spatial * (1.0 + lambda_sem * dist_sem)

        # 5. Convert condensed array to N x N square matrix for DBSCAN
        dist_matrix = squareform(dist_coupled_condensed)

        # 6. Run DBSCAN using the precomputed distance matrix
        try:
            # Notice eps_m evaluates against penalized physical meters
            model = DBSCAN(eps=self.eps_m, min_samples=self.min_samples, metric='precomputed', n_jobs=1)
            labels = model.fit_predict(dist_matrix)
        except Exception as e:
            self.get_logger().error(f"DBSCAN failed: {e}")
            return

        unique_labels = set(labels.tolist())
        if -1 in unique_labels:
            unique_labels.remove(-1)  
            
        # collect clusters
        clusters = []
        for lbl in unique_labels:
            members_idx = [i for (i, lab) in enumerate(labels) if lab == lbl]
            members = [buffer_list[i] for i in members_idx]
            clusters.append((lbl, members_idx, members))

        # 7. analyze clusters for potential confirmation
        for (lbl, members_idx, members) in clusters:
            kf_ids = [int(m['id']) for m in members]
            nx = np.mean([m['x'] for m in members])
            ny = np.mean([m['y'] for m in members])
            times = [m['t'] for m in members]
            duration = max(times) - min(times) if len(times) >= 2 else 0.0

            scene_vecs_cluster = []
            sem_dict_combined = {}
            for m in members:
                # build scene vector for mean
                scene_vec = self.get_scene_vector(m)
                if scene_vec.size > 0:
                    scene_vecs_cluster.append(scene_vec)
                # merge raw semantic dict for top_label
                for k, v in m.get('sem', {}).items():
                    sem_dict_combined[k] = max(sem_dict_combined.get(k, 0.0), v)

            if len(scene_vecs_cluster) > 0:
                scene_vecs_arr = np.asarray(scene_vecs_cluster, dtype=float)
                if scene_vecs_arr.ndim == 1:
                    scene_vecs_arr = scene_vecs_arr.reshape(-1, 1)

                scene_sig = l2_normalize(np.mean(scene_vecs_arr, axis=0))
                cohesion = float(np.mean([cosine_similarity(v, scene_sig) for v in scene_vecs_arr]))
            else:
                scene_sig = np.zeros(0, dtype=float)
                cohesion = 0.0

            membership_score = min(1.0, len(kf_ids) / max(1.0, float(self.min_kfs_confirm)))
            zone_conf = float(self.zone_confidence_base * 0.5 + 0.5 * (0.5 * cohesion + 0.5 * membership_score))

            confirmed = False
            if len(kf_ids) >= self.min_kfs_confirm and cohesion >= self.min_clip_cohesion:
                confirmed = True
            elif duration >= self.temporal_confirm_s and cohesion >= 0.30:
                confirmed = True

            top_label = ""
            top_conf = 0.0
            if len(sem_dict_combined) > 0:
                top_label = max(sem_dict_combined.items(), key=lambda kv: kv[1])[0]
                top_conf = float(max(sem_dict_combined.values()))

            if confirmed:
                self.create_or_update_zone(kf_ids, (nx, ny), scene_sig, top_label, top_conf, zone_conf)

    # ---------------- Zone Management ----------------
    def create_or_update_zone(self, kf_ids, centroid_xy, sem_sig, top_label, top_conf, conf):
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
            self.get_logger().info(f"Created zone {zid} center=({zone['centroid'][0]:.2f},{zone['centroid'][1]:.2f}) size={len(kf_ids)}")
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
                self.get_logger().info(f"Merged into zone {merged_zone_id} => new size={len(z['members'])}")
            else:
                z['dirty'] = False

    # ---------------- Publishing & Visualization ----------------
    def publish_zones(self):
        now = self.get_clock().now().to_msg()
        with self.lock:
            dirty_zones = [(zid, z) for zid, z in self.zones.items() if z.get('dirty', False)]

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

                if hasattr(zm, 'keyframe_ids'):
                    zm.keyframe_ids = list(sorted(z['members']))
                else:
                    try:
                        zm.keyframes = list(sorted(z['members']))
                    except Exception:
                        pass

                if hasattr(zm, 'top_labels'):
                    zm.top_labels = [z.get('top_label', '')]
                if hasattr(zm, 'top_label_confidences'):
                    zm.top_label_confidences = [float(z.get('top_conf', 0.0))]
                if hasattr(zm, 'confidence'):
                    zm.confidence = float(z.get('confidence', 0.0))
                if hasattr(zm, 'version'):
                    zm.version = int(z.get('version', 0))

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

                with self.lock:
                    if zid in self.zones:
                        self.zones[zid]['dirty'] = False
                        self.zones[zid]['last_published_version'] = z.get('version', 0)
            except Exception as e:
                self.get_logger().error(f"Error publishing zone {zid}: {e}")

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

    def make_cluster_perimeter_marker(self, cluster_points, marker_id, zone_id, frame_id="map"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"zone_perimeter_{zone_id}"
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
            pts = [(x-s,y-s), (x+s,y-s), (x+s,y+s), (x-s,y+s), (x-s,y-s)]
        elif len(cluster_points) == 2:
            xs = [p[0] for p in cluster_points]
            ys = [p[1] for p in cluster_points]
            pad = 0.15
            min_x, max_x = min(xs) - pad, max(xs) + pad
            min_y, max_y = min(ys) - pad, max(ys) + pad
            pts = [(min_x,min_y), (max_x,min_y), (max_x,max_y), (min_x,max_y), (min_x,min_y)]
        else:
            hull = self.convex_hull_2d(cluster_points)
            if len(hull) < 3:
                xs = [p[0] for p in cluster_points]
                ys = [p[1] for p in cluster_points]
                pad = 0.15
                min_x, max_x = min(xs) - pad, max(xs) + pad
                min_y, max_y = min(ys) - pad, max(ys) + pad
                pts = [(min_x,min_y), (max_x,min_y), (max_x,max_y), (min_x,max_y), (min_x,min_y)]
            else:
                pts = hull + [hull[0]] 

        for x, y in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)
        return marker

    def publish_zone_perimeters(self):
        ma = MarkerArray()
        marker_id = 0

        with self.lock:
            zones_snapshot = list(self.zones.items())
            buffer_snapshot = list(self.buffer)

        for zid, z in zones_snapshot:
            member_points = []
            member_ids = set(z.get('members', set()))
            for entry in buffer_snapshot:
                if int(entry['id']) in member_ids:
                    member_points.append((float(entry['x']), float(entry['y'])))

            marker = self.make_cluster_perimeter_marker(member_points, marker_id, zid)
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
                p1.x, p1.y, p1.z = cx, cy, 0.08
                p2 = Point()
                p2.x, p2.y, p2.z = kx, ky, 0.08

                marker.points.append(p1)
                marker.points.append(p2)

            if len(marker.points) > 0:
                ma.markers.append(marker)
                marker_id += 1

        if self.zone_links_pub is not None:
            self.zone_links_pub.publish(ma)

    def publish_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        i = 0
        for zid, z in self.zones.items():
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
            
            cid = (zid * 37) % 255
            m.color.r = ((cid * 97) % 255) / 255.0
            m.color.g = ((cid * 71) % 255) / 255.0
            m.color.b = ((cid * 53) % 255) / 255.0
            m.color.a = 0.9
            ma.markers.append(m)
            i += 1

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