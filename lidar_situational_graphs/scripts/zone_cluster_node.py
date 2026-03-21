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
from std_msgs.msg import Header

# helper utils
# def cosine_similarity(a, b):
#     a = np.asarray(a, dtype=float)
#     b = np.asarray(b, dtype=float)
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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
        self.declare_parameter('window_duration_s', 120.0)
        self.declare_parameter('min_samples', 3)
        self.declare_parameter('eps_m', 1.8)  # in meters (but will be applied after feature scaling)
        self.declare_parameter('w_sem', 2.0)  # semantic weight
        self.declare_parameter('min_kfs_confirm', 3)
        self.declare_parameter('purity_thresh', 0.6)
        self.declare_parameter('temporal_confirm_s', 8.0)
        self.declare_parameter('merge_dist_m', 2.0)
        self.declare_parameter('merge_semantic_thresh', 0.7)
        self.declare_parameter('zone_confidence_base', 0.5)
        self.declare_parameter('visualize', True)
        self.declare_parameter('marker_ns', 'zones')

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

            # create sem_vec for this message
            if len(self.vocab_rev) == 0:
                sem_vec = np.zeros(0, dtype=float)
            else:
                sem_vec = np.zeros(len(self.vocab_rev), dtype=float)
                for lab, c in sem_dict.items():
                    idx = self.vocab.get(lab, None)
                    if idx is not None:
                        sem_vec[idx] = float(c)

            # normalize semantic vector (so magnitude not explode)
            if sem_vec.size > 0:
                sem_vec = normalize_vec(sem_vec)

            # append to buffer
            self.buffer.append({
                'id': int(kfid),
                't': float(t),
                'x': float(x),
                'y': float(y),
                'pose': pose_stamped,
                'sem': sem_dict,
                'sem_vec': sem_vec
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

    # ---------------- core: clustering ----------------
    def build_feature_matrix(self):
        """Return (N, D) feature matrix and associated list of buffer indices."""
        print("Building feature matrix")
        with self.lock:
            n = len(self.buffer)
            if n == 0:
                return None, None

            pos = np.zeros((n, 2), dtype=float)
            # semantic vectors might be zero-length if no labels observed yet
            sem_dim = len(self.vocab_rev)
            if sem_dim > 0:
                sem = np.zeros((n, sem_dim), dtype=float)
            else:
                sem = np.zeros((n, 0), dtype=float)

            for i, entry in enumerate(self.buffer):
                pos[i, 0] = entry['x']
                pos[i, 1] = entry['y']
                sv = entry.get('sem_vec', None)
                if sv is None:
                    if sem_dim > 0:
                        sem[i, :] = 0.0
                else:
                    # sv might be smaller if vocabulary was expanded earlier (but we expand on insertion)
                    if len(sv) != sem_dim:
                        tmp = np.zeros(sem_dim, dtype=float)
                        tmp[:len(sv)] = sv
                        sem[i, :] = tmp
                    else:
                        sem[i, :] = sv

            # scale semantic by w_sem
            if sem_dim > 0:
                sem_scaled = self.w_sem * sem
                features = np.hstack([pos, sem_scaled])
            else:
                features = pos  # only spatial

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
            kf_ids = [int(m['id']) for m in members]
            nx = np.mean([m['x'] for m in members])
            ny = np.mean([m['y'] for m in members])
            times = [m['t'] for m in members]
            duration = max(times) - min(times) if len(times) >= 2 else 0.0

            # semantic signature and purity
            sem_dim = len(self.vocab_rev)
            sig = np.zeros(sem_dim, dtype=float) if sem_dim > 0 else np.zeros(0, dtype=float)
            for m in members:
                if sem_dim > 0:
                    v = m.get('sem_vec', np.zeros(sem_dim))
                    if len(v) < sem_dim:
                        tmp = np.zeros(sem_dim)
                        tmp[:len(v)] = v
                        v = tmp
                    sig += v
            if sem_dim > 0:
                sig = normalize_vec(sig)
                purity = float(np.max(sig)) if sig.size > 0 else 0.0
                top_idx = int(np.argmax(sig)) if sig.size > 0 else None
                top_label = self.vocab_rev[top_idx] if top_idx is not None and top_idx < len(self.vocab_rev) else ''
                top_conf = float(sig[top_idx]) if top_idx is not None else 0.0
            else:
                purity = 0.0
                top_label = ''
                top_conf = 0.0

            # compute zone confidence (simple heuristic)
            membership_score = min(1.0, len(kf_ids) / max(1.0, float(self.min_kfs_confirm)))
            zone_conf = float(self.zone_confidence_base * 0.5 + 0.5 * (0.5 * purity + 0.5 * membership_score))

            # decide tentative/confirmed
            confirmed = False
            if len(kf_ids) >= self.min_kfs_confirm and purity >= self.purity_thresh:
                confirmed = True
            elif duration >= self.temporal_confirm_s and purity >= 0.3:
                confirmed = True

            # either create new zone or update existing one (merge if necessary)
            if confirmed:
                self.create_or_update_zone(kf_ids, (nx, ny), sig, top_label, top_conf, zone_conf)
            else:
                # keep as candidate - maybe store for future confirmation
                # we simply log
                self.get_logger().info(f"Candidate cluster lbl={lbl} size={len(kf_ids)} purity={purity:.2f} duration={duration:.1f}s not confirmed")

    
    # ---------------- zone management ----------------
    def create_or_update_zone(self, kf_ids, centroid_xy, sem_sig, top_label, top_conf, conf):
        """Create a new zone or merge into an existing one when close + semantically similar."""
        # search for mergeable zone
        print("Creating or updating zone")
        merged_zone_id = None
        print("Printing Zones")
        print(self.zones.items())
        for zid, z in self.zones.items():
            zx, zy = z['centroid']
            d = math.hypot(zx - centroid_xy[0], zy - centroid_xy[1])
            sem_sim = cosine_similarity(z['sem_sig'], sem_sig) if (z['sem_sig'].size>0 and sem_sig.size>0) else 0.0
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
                'sem_sig': sem_sig.copy() if hasattr(sem_sig, 'copy') else np.array(sem_sig),
                'top_label': top_label,
                'top_conf': float(top_conf),
                'confidence': float(conf),
                'state': 'confirmed',
                'last_updated': time.time(),
                'version': 1
            }
            self.zones[zid] = zone
            self.get_logger().info(f"Created zone {zid} center=({zone['centroid'][0]:.2f},{zone['centroid'][1]:.2f}) size={len(kf_ids)} label={top_label} conf={conf:.2f}")
        else:
            # merge into existing zone
            z = self.zones[merged_zone_id]
            old_members = set(z['members'])
            z['members'].update(kf_ids)
            # recompute centroid as mean of positions of members currently in buffer if available
            with self.lock:
                member_positions = []
                for entry in self.buffer:
                    if entry['id'] in z['members']:
                        member_positions.append((entry['x'], entry['y']))
            if len(member_positions) > 0:
                xs = [p[0] for p in member_positions]
                ys = [p[1] for p in member_positions]
                print(xs,ys)
                z['centroid'] = (float(np.mean(xs)), float(np.mean(ys)))
            # merge semantic signature by weighted average (old sig + new sig)
            try:
                z_old_sig = z['sem_sig']
                if z_old_sig.size == 0:
                    z['sem_sig'] = sem_sig.copy()
                else:
                    z['sem_sig'] = normalize_vec( (z_old_sig + sem_sig) / 2.0 )
            except Exception:
                z['sem_sig'] = sem_sig.copy()

            z['top_label'] = top_label if top_conf >= z.get('top_conf', 0.0) else z.get('top_label', top_label)
            z['top_conf'] = max(top_conf, z.get('top_conf', top_conf))
            z['confidence'] = min(1.0, 0.9 * z.get('confidence', 0.5) + 0.1 * conf)
            z['last_updated'] = time.time()
            z['version'] += 1
            self.get_logger().info(f"Merged into zone {merged_zone_id} => new size={len(z['members'])} centroid=({z['centroid'][0]:.2f},{z['centroid'][1]:.2f})")

    # ---------------- publish zones as messages ----------------
    def publish_zones(self):
        """Publish ZoneMsg for each confirmed zone."""
        print("Publishing zones")
        now = self.get_clock().now().to_msg()
        for zid, z in list(self.zones.items()):
            try:
                zm = ZoneMsg()
                # put header if exists
                try:
                    zm.header = Header()
                    zm.header.stamp = now
                except Exception:
                    pass
                # set zone id
                if hasattr(zm, 'zone_id'):
                    zm.zone_id = int(zid)
                # keyframe ids
                if hasattr(zm, 'keyframe_ids'):
                    zm.keyframe_ids = list(z['members'])
                else:
                    try:
                        zm.keyframes = list(z['members'])
                    except Exception:
                        pass
                # top_labels / confidences
                if hasattr(zm, 'top_labels'):
                    zm.top_labels = [z.get('top_label', '')]
                if hasattr(zm, 'top_label_confidences'):
                    zm.top_label_confidences = [float(z.get('top_conf', 0.0))]
                if hasattr(zm, 'confidence'):
                    zm.confidence = float(z.get('confidence', 0.0))
                # supporting_planes optional - left empty
                if hasattr(zm, 'version'):
                    zm.version = int(z.get('version', 0))
                # centroid if present
                if hasattr(zm, 'centroid') or hasattr(zm, 'pose'):
                    try:
                        # some Zone.msg may have centroid or Pose; try to set if present
                        if hasattr(zm, 'centroid'):
                            zm.centroid.position.x = float(z['centroid'][0])
                            zm.centroid.position.y = float(z['centroid'][1])
                            zm.centroid.z = 0.0
                        if hasattr(zm, 'pose'):
                            p = PoseStamped()
                            p.header.stamp = now
                            p.header.frame_id = 'map'
                            p.pose.position.x = float(z['centroid'][0])
                            p.pose.position.y = float(z['centroid'][1])
                            p.pose.position.z = 0.0
                            zm.pose = p
                    except Exception:
                        pass

                # publish
                self.zone_pub.publish(zm)
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