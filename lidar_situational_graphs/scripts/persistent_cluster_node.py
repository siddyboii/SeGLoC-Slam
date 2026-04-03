#!/usr/bin/env python3
"""
Persistent zone clustering node.

What this node does:
- subscribes to KeyframeSemantic
- keeps a short-term buffer for DBSCAN
- keeps a long-term persistent zone memory
- reactivates an old zone when a revisited keyframe is spatially + semantically similar
- merges DBSCAN clusters into existing persistent zones when possible
- creates a new zone only when no persistent zone matches
- publishes Zone messages and RViz markers

This is a stronger version of the earlier zone_cluster_node:
- old zones are NOT forgotten when they leave the sliding buffer
- revisit keyframes can be merged back into the same zone
- each zone keeps centroid, semantic signature, confidence, members, last_seen time, etc.

Adjust topic/package names if your msg package differs.
"""

import math
import time
from collections import deque
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

try:
    from sklearn.cluster import DBSCAN
except Exception as e:
    raise RuntimeError("scikit-learn is required. Install with `pip install scikit-learn`.") from e

from situational_graphs_msgs.msg import KeyframeSemantic, Zone as ZoneMsg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Polygon
from geometry_msgs.msg import Point32
from std_msgs.msg import Header


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    n = max(len(a), len(b))
    aa = np.zeros(n, dtype=float)
    bb = np.zeros(n, dtype=float)
    aa[:len(a)] = a
    bb[:len(b)] = b
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def normalize_vec(v):
    v = np.asarray(v, dtype=float)
    s = float(np.sum(v))
    if s <= 0.0:
        return np.zeros_like(v)
    return v / s


def centroid_of_points_xy(points_xy):
    if not points_xy:
        return np.array([0.0, 0.0], dtype=float), 0.0
    pts = np.asarray(points_xy, dtype=float)
    c = pts.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(pts - c, axis=1))) if len(pts) > 0 else 0.0
    return c, spread


def zone_weight(confidence, spread, support_count, base_info=1.0):
    # keep it soft by default
    support_boost = 1.0 + min(2.0, 0.25 * max(0, support_count - 1))
    spread = max(spread, 0.5)
    w = base_info * (1.0 + 10.0 * float(confidence)) * support_boost / (1.0 + spread)
    w = max(0.05, min(w, 20.0))
    return w


class ZonesClusterNode(Node):
    def __init__(self):
        super().__init__('zones_cluster_node')

        # ---------- parameters ----------
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('clustering_period', 2.0)
        self.declare_parameter('window_duration_s', 120.0)
        self.declare_parameter('min_samples', 3)
        self.declare_parameter('eps_m', 1.8)
        self.declare_parameter('w_sem', 2.0)
        self.declare_parameter('min_kfs_confirm', 3)
        self.declare_parameter('purity_thresh', 0.6)
        self.declare_parameter('temporal_confirm_s', 8.0)
        self.declare_parameter('merge_dist_m', 2.0)
        self.declare_parameter('merge_semantic_thresh', 0.7)
        self.declare_parameter('zone_confidence_base', 0.5)
        self.declare_parameter('visualize', True)
        self.declare_parameter('marker_ns', 'zones')

        # persistent-zone specific tuning
        self.declare_parameter('reactivate_dist_m', 2.5)
        self.declare_parameter('reactivate_semantic_thresh', 0.55)
        self.declare_parameter('cluster_reuse_dist_m', 2.0)
        self.declare_parameter('cluster_reuse_semantic_thresh', 0.6)
        self.declare_parameter('zone_dormant_after_s', 90.0)
        self.declare_parameter('min_zone_keyframes', 2)
        self.declare_parameter('zone_decay_alpha', 0.25)  # centroid/semantic update rate
        self.declare_parameter('max_persistent_zones', 500)

        # load parameters
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

        self.reactivate_dist_m = float(self.get_parameter('reactivate_dist_m').value)
        self.reactivate_semantic_thresh = float(self.get_parameter('reactivate_semantic_thresh').value)
        self.cluster_reuse_dist_m = float(self.get_parameter('cluster_reuse_dist_m').value)
        self.cluster_reuse_semantic_thresh = float(self.get_parameter('cluster_reuse_semantic_thresh').value)
        self.zone_dormant_after_s = float(self.get_parameter('zone_dormant_after_s').value)
        self.min_zone_keyframes = int(self.get_parameter('min_zone_keyframes').value)
        self.zone_decay_alpha = float(self.get_parameter('zone_decay_alpha').value)
        self.max_persistent_zones = int(self.get_parameter('max_persistent_zones').value)

        # ---------- runtime state ----------
        self.lock = Lock()
        self.buffer = deque()  # recent keyframes for DBSCAN

        # dynamic semantic vocab
        self.vocab = {}      # label -> idx
        self.vocab_rev = []  # idx -> label

        # persistent zone memory
        # zone dict fields:
        #   id, centroid(np.array[2]), sem_sig(np.array), keyframe_ids(set), top_label, top_conf,
        #   confidence, state, last_updated, last_seen_kf_id, last_seen_time, version, active, spatial_std
        self.zones = {}
        self.next_zone_id = 1

        # bookkeeping to prevent duplicate attachments inside this runtime
        self.kf_to_zone = {}  # kf_id -> zone_id
        self.kf_seen = set()  # keyframes already processed
        self.zone_edge_keys = set()  # (zone_id, kf_id) attachments already absorbed

        # zone update bookkeeping
        self.zones_dirty = False
        self.last_cluster_time = 0.0

        # ---------- publishers/subscribers ----------
        qos_z = QoSProfile(depth=10)
        qos_z.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_z.reliability = ReliabilityPolicy.RELIABLE
        self.zone_pub = self.create_publisher(ZoneMsg, 'zones', qos_z)
        self.marker_pub = self.create_publisher(MarkerArray, 'zones_markers', 10) if self.visualize else None

        self.kf_sub = self.create_subscription(
            KeyframeSemantic,
            '/s_graphs/keyframe_semantic',
            self.cb_kf_semantic,
            10
        )

        self.create_timer(1.0 / max(0.1, self.publish_rate), self.timer_publish)
        self.get_logger().info("Persistent ZonesClusterNode started")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def cb_kf_semantic(self, msg: KeyframeSemantic):
        """Store keyframe and try immediate persistent-zone reactivation."""
        kfid = int(getattr(msg, 'keyframe_id', -1))
        if kfid < 0:
            kfid = int(time.time() * 1000)

        t = self.get_clock().now().nanoseconds / 1e9

        # pose extraction
        pose_stamped = None
        if hasattr(msg, 'pose') and msg.pose is not None:
            pose_stamped = msg.pose

        x = 0.0
        y = 0.0
        if pose_stamped is not None:
            try:
                # your current message usage implies PoseStamped:
                x = float(pose_stamped.position.x)
                y = float(pose_stamped.position.y)
            except Exception:
                x = 0.0
                y = 0.0

        # semantic labels/confidence
        labels = list(getattr(msg, 'objects', []))
        confs = list(getattr(msg, 'object_confidence', []))

        sem_dict = {}
        for i, lab in enumerate(labels):
            if lab is None:
                continue
            c = float(confs[i]) if i < len(confs) else 1.0
            sem_dict[str(lab)] = max(sem_dict.get(str(lab), 0.0), c)

        sem_vec = self._semantic_vector_from_dict(sem_dict)

        entry = {
            'id': kfid,
            't': t,
            'x': float(x),
            'y': float(y),
            'pose': pose_stamped,
            'sem': sem_dict,
            'sem_vec': sem_vec,
            'zone_id': None,
        }

        with self.lock:
            self._expand_vocab_for_labels(list(sem_dict.keys()))
            entry['sem_vec'] = self._semantic_vector_from_dict(sem_dict)
            self.buffer.append(entry)

            # remove old entries from short-term buffer only
            cutoff = t - self.window_duration_s
            while self.buffer and self.buffer[0]['t'] < cutoff:
                self.buffer.popleft()

            # immediate persistent-zone reactivation by centroid + semantic similarity
            matched_zone_id = self._match_persistent_zone(
                np.array([entry['x'], entry['y']], dtype=float),
                entry['sem_vec'],
                entry['sem']
            )
            if matched_zone_id is not None:
                self._attach_keyframe_to_zone_locked(entry, matched_zone_id, source='reactivation')

        self.get_logger().info(
            f"Received kf {kfid} @ ({x:.2f},{y:.2f}) labels={list(sem_dict.keys())}"
        )

    # ------------------------------------------------------------------
    # Clustering timer
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    def _expand_vocab_for_labels(self, labels):
        expanded = False
        for lab in labels:
            if lab not in self.vocab:
                self.vocab[lab] = len(self.vocab_rev)
                self.vocab_rev.append(lab)
                expanded = True
        if expanded:
            # expand existing buffer vectors so dimensions stay consistent
            for entry in self.buffer:
                old = entry.get('sem_vec', None)
                if old is None:
                    entry['sem_vec'] = np.zeros(len(self.vocab_rev), dtype=float)
                elif len(old) < len(self.vocab_rev):
                    new = np.zeros(len(self.vocab_rev), dtype=float)
                    new[:len(old)] = old
                    entry['sem_vec'] = new
            # expand persistent zone semantic signatures
            for z in self.zones.values():
                sig = z.get('sem_sig', np.zeros(0, dtype=float))
                if len(sig) < len(self.vocab_rev):
                    new_sig = np.zeros(len(self.vocab_rev), dtype=float)
                    new_sig[:len(sig)] = sig
                    z['sem_sig'] = new_sig

    def _semantic_vector_from_dict(self, sem_dict):
        if len(self.vocab_rev) == 0:
            return np.zeros(0, dtype=float)
        vec = np.zeros(len(self.vocab_rev), dtype=float)
        for lab, c in sem_dict.items():
            idx = self.vocab.get(lab, None)
            if idx is not None:
                vec[idx] = float(c)
        if vec.size > 0:
            vec = normalize_vec(vec)
        return vec

    def build_feature_matrix(self):
        with self.lock:
            n = len(self.buffer)
            if n == 0:
                return None, None

            pos = np.zeros((n, 2), dtype=float)
            sem_dim = len(self.vocab_rev)
            sem = np.zeros((n, sem_dim), dtype=float) if sem_dim > 0 else np.zeros((n, 0), dtype=float)

            for i, entry in enumerate(self.buffer):
                pos[i, 0] = entry['x']
                pos[i, 1] = entry['y']
                sv = entry.get('sem_vec', None)
                if sv is None:
                    continue
                if len(sv) != sem_dim:
                    tmp = np.zeros(sem_dim, dtype=float)
                    tmp[:len(sv)] = sv
                    sem[i, :] = tmp
                else:
                    sem[i, :] = sv

            if sem_dim > 0:
                features = np.hstack([pos, self.w_sem * sem])
            else:
                features = pos

            return features, list(range(n))

    # ------------------------------------------------------------------
    # Persistent zone logic
    # ------------------------------------------------------------------
    def _zone_semantic_signature_from_members(self, members):
        sem_dim = len(self.vocab_rev)
        if sem_dim == 0 or not members:
            return np.zeros(sem_dim, dtype=float)
        sig = np.zeros(sem_dim, dtype=float)
        for m in members:
            sv = m.get('sem_vec', None)
            if sv is None:
                continue
            if len(sv) < sem_dim:
                tmp = np.zeros(sem_dim, dtype=float)
                tmp[:len(sv)] = sv
                sv = tmp
            sig += sv
        return normalize_vec(sig)

    def _zone_centroid_and_spread_from_members(self, members):
        pts = [(m['x'], m['y']) for m in members]
        if not pts:
            return np.array([0.0, 0.0], dtype=float), 0.0
        return centroid_of_points_xy(pts)

    def _create_new_zone_locked(self, members, centroid_xy, sem_sig, top_label, top_conf, confidence):
        if len(self.zones) >= self.max_persistent_zones:
            # simple eviction policy: keep older higher-confidence zones; here we skip creation
            self.get_logger().warn("Persistent zone limit reached; skipping new zone creation")
            return None

        zid = self.next_zone_id
        self.next_zone_id += 1

        cxy = np.array([float(centroid_xy[0]), float(centroid_xy[1])], dtype=float)
        zone = {
            'id': zid,
            'centroid': cxy,
            'sem_sig': sem_sig.copy() if hasattr(sem_sig, 'copy') else np.array(sem_sig, dtype=float),
            'keyframe_ids': set(int(m['id']) for m in members),
            'top_label': top_label,
            'top_conf': float(top_conf),
            'confidence': float(confidence),
            'state': 'confirmed' if len(members) >= self.min_zone_keyframes else 'tentative',
            'last_updated': time.time(),
            'last_seen_time': time.time(),
            'last_seen_kf_id': int(members[-1]['id']) if members else -1,
            'version': 1,
            'active': True,
            'spatial_std': float(self._zone_centroid_and_spread_from_members(members)[1]),
            'support_count': len(members),
            'polygon_radius': max(0.75, float(self._zone_centroid_and_spread_from_members(members)[1]) * 1.5),
        }

        self.zones[zid] = zone
        for m in members:
            self.kf_to_zone[int(m['id'])] = zid
            self.zone_edge_keys.add((zid, int(m['id'])))
            m['zone_id'] = zid

        self.zones_dirty = True
        self.get_logger().info(
            f"Created zone {zid} center=({cxy[0]:.2f},{cxy[1]:.2f}) "
            f"size={len(members)} label={top_label} conf={confidence:.2f}"
        )
        return zid

    def _update_zone_locked(self, zid, members, centroid_xy, sem_sig, top_label, top_conf, confidence, source='cluster'):
        z = self.zones.get(zid, None)
        if z is None:
            return None

        new_pts = [(m['x'], m['y']) for m in members]
        old_centroid = np.asarray(z['centroid'], dtype=float)
        new_centroid, new_spread = centroid_of_points_xy(new_pts)
        alpha = self.zone_decay_alpha
        if source == 'reactivation':
            alpha = min(0.5, max(alpha, 0.35))

        # centroid EMA
        z['centroid'] = (1.0 - alpha) * old_centroid + alpha * new_centroid

        # semantic signature EMA
        old_sig = z.get('sem_sig', np.zeros(len(self.vocab_rev), dtype=float))
        if len(old_sig) < len(self.vocab_rev):
            tmp = np.zeros(len(self.vocab_rev), dtype=float)
            tmp[:len(old_sig)] = old_sig
            old_sig = tmp
        if len(sem_sig) < len(self.vocab_rev):
            tmp = np.zeros(len(self.vocab_rev), dtype=float)
            tmp[:len(sem_sig)] = sem_sig
            sem_sig = tmp
        z['sem_sig'] = normalize_vec((1.0 - alpha) * old_sig + alpha * sem_sig)

        # members
        for m in members:
            mid = int(m['id'])
            z['keyframe_ids'].add(mid)
            self.kf_to_zone[mid] = zid
            self.zone_edge_keys.add((zid, mid))
            m['zone_id'] = zid

        # confidence/state updates
        z['support_count'] = len(z['keyframe_ids'])
        z['confidence'] = min(1.0, 0.85 * float(z.get('confidence', 0.5)) + 0.15 * float(confidence))
        z['top_label'] = top_label if top_conf >= z.get('top_conf', 0.0) else z.get('top_label', top_label)
        z['top_conf'] = max(float(top_conf), float(z.get('top_conf', top_conf)))
        z['last_updated'] = time.time()
        z['last_seen_time'] = time.time()
        z['last_seen_kf_id'] = int(members[-1]['id']) if members else z.get('last_seen_kf_id', -1)
        z['version'] = int(z.get('version', 1)) + 1
        z['active'] = True
        z['spatial_std'] = float(new_spread)
        z['polygon_radius'] = max(0.75, new_spread * 1.5)

        if z['support_count'] >= self.min_zone_keyframes and z['confidence'] >= self.purity_thresh:
            z['state'] = 'confirmed'
        else:
            z['state'] = 'tentative'

        self.zones_dirty = True
        self.get_logger().info(
            f"{source.capitalize()}-updated zone {zid}: size={len(z['keyframe_ids'])} "
            f"centroid=({z['centroid'][0]:.2f},{z['centroid'][1]:.2f}) conf={z['confidence']:.2f} "
            f"spread={z['spatial_std']:.3f}"
        )
        return zid

    def _match_persistent_zone(self, centroid_xy, sem_vec, sem_dict):
        """Return best existing zone id for a revisited keyframe, else None."""
        if not self.zones:
            return None

        best = None
        best_score = -1e9
        now = time.time()

        for zid, z in self.zones.items():
            if z is None:
                continue

            # allow dormant zones to be reactivated
            age = now - float(z.get('last_seen_time', now))
            if age > self.zone_dormant_after_s:
                z['active'] = False

            zc = np.asarray(z['centroid'], dtype=float)
            dist = float(np.linalg.norm(centroid_xy - zc))

            zsig = z.get('sem_sig', np.zeros(len(self.vocab_rev), dtype=float))
            if len(zsig) < len(self.vocab_rev):
                tmp = np.zeros(len(self.vocab_rev), dtype=float)
                tmp[:len(zsig)] = zsig
                zsig = tmp

            sem_sim = cosine_similarity(zsig, sem_vec)

            # soft score: distance + semantics + confidence
            if dist > self.reactivate_dist_m:
                continue
            if sem_sim < self.reactivate_semantic_thresh:
                continue

            conf_bonus = float(z.get('confidence', 0.0))
            recency_bonus = max(0.0, 1.0 - min(1.0, age / max(self.zone_dormant_after_s, 1.0)))
            score = 2.0 * sem_sim - 0.5 * dist + 0.4 * conf_bonus + 0.2 * recency_bonus

            if score > best_score:
                best_score = score
                best = zid

        return best

    def _match_cluster_to_zone(self, centroid_xy, sem_sig):
        """Try to merge a DBSCAN cluster into an existing persistent zone."""
        best = None
        best_score = -1e9

        for zid, z in self.zones.items():
            zc = np.asarray(z['centroid'], dtype=float)
            dist = float(np.linalg.norm(centroid_xy - zc))

            zsig = z.get('sem_sig', np.zeros(len(self.vocab_rev), dtype=float))
            if len(zsig) < len(self.vocab_rev):
                tmp = np.zeros(len(self.vocab_rev), dtype=float)
                tmp[:len(zsig)] = zsig
                zsig = tmp

            sem_sim = cosine_similarity(zsig, sem_sig)
            if dist > self.cluster_reuse_dist_m:
                continue
            if sem_sim < self.cluster_reuse_semantic_thresh:
                continue

            score = 2.0 * sem_sim - 0.5 * dist + 0.3 * float(z.get('confidence', 0.0))
            if score > best_score:
                best_score = score
                best = zid

        return best

    def _attach_keyframe_to_zone_locked(self, entry, zone_id, source='reactivation'):
        """Attach a keyframe entry to an existing zone and update zone stats."""
        if zone_id is None:
            return

        kf_id = int(entry['id'])
        edge_key = (zone_id, kf_id)
        if edge_key in self.zone_edge_keys:
            # already attached in this runtime
            entry['zone_id'] = zone_id
            self.kf_to_zone[kf_id] = zone_id
            return

        z = self.zones.get(zone_id, None)
        if z is None:
            return

        # update zone with this single keyframe as support
        # but keep it soft; a reactivation alone should not fully rewrite the zone
        centroid_xy = np.array([entry['x'], entry['y']], dtype=float)
        sem_sig = entry['sem_vec']

        self.zone_edge_keys.add(edge_key)
        self.kf_to_zone[kf_id] = zone_id
        entry['zone_id'] = zone_id

        # re-attach by treating this keyframe as a support member
        self._update_zone_locked(
            zone_id,
            [entry],
            centroid_xy,
            sem_sig,
            self._top_label_from_sem_dict(entry['sem']),
            self._top_conf_from_sem_dict(entry['sem']),
            max(float(z.get('confidence', 0.5)), self.zone_confidence_base),
            source=source
        )

    def _top_label_from_sem_dict(self, sem_dict):
        if not sem_dict:
            return ''
        return max(sem_dict.items(), key=lambda kv: kv[1])[0]

    def _top_conf_from_sem_dict(self, sem_dict):
        if not sem_dict:
            return 0.0
        return float(max(sem_dict.values()))

    # ------------------------------------------------------------------
    # DBSCAN clustering over the short-term buffer
    # ------------------------------------------------------------------
    def run_clustering(self):
        features, idxs = self.build_feature_matrix()
        if features is None or features.shape[0] < self.min_samples:
            return

        try:
            model = DBSCAN(eps=self.eps_m, min_samples=self.min_samples, metric='euclidean', n_jobs=1)
            labels = model.fit_predict(features)
        except Exception as e:
            self.get_logger().error(f"DBSCAN failed: {e}")
            return

        clusters = {}
        for row_idx, lbl in enumerate(labels):
            if lbl == -1:
                continue
            clusters.setdefault(int(lbl), []).append(self.buffer[row_idx])

        if not clusters:
            return

        with self.lock:
            for lbl, members in clusters.items():
                if len(members) < self.min_samples:
                    continue

                points_xy = [(m['x'], m['y']) for m in members]
                centroid_xy, spread = centroid_of_points_xy(points_xy)
                sem_sig = self._zone_semantic_signature_from_members(members)

                # semantic purity / top label
                if len(sem_sig) > 0:
                    top_idx = int(np.argmax(sem_sig))
                    top_label = self.vocab_rev[top_idx] if top_idx < len(self.vocab_rev) else ''
                    top_conf = float(sem_sig[top_idx])
                    purity = top_conf
                else:
                    top_label = ''
                    top_conf = 0.0
                    purity = 0.0

                # temporal extent
                times = [m['t'] for m in members]
                duration = float(max(times) - min(times)) if len(times) > 1 else 0.0

                # cluster confidence
                support = len(members)
                membership_score = min(1.0, support / max(1.0, float(self.min_kfs_confirm)))
                confidence = self.zone_confidence_base * 0.5 + 0.5 * (0.5 * purity + 0.5 * membership_score)

                # Should this cluster merge with an old/persistent zone?
                reuse_zone_id = self._match_cluster_to_zone(centroid_xy, sem_sig)

                if reuse_zone_id is not None:
                    self._update_zone_locked(
                        reuse_zone_id,
                        members,
                        centroid_xy,
                        sem_sig,
                        top_label,
                        top_conf,
                        confidence,
                        source='cluster'
                    )
                    continue

                # Confirm or create new zone
                if support >= self.min_zone_keyframes and (purity >= self.purity_thresh or duration >= self.temporal_confirm_s):
                    self._create_new_zone_locked(members, centroid_xy, sem_sig, top_label, top_conf, confidence)
                else:
                    # still keep it as a latent candidate; may get merged later by reactivation
                    self.get_logger().debug(
                        f"Cluster {lbl} not confirmed yet: size={support} purity={purity:.3f} duration={duration:.2f}"
                    )

            # mark zones as dormant if not seen recently
            now = time.time()
            for z in self.zones.values():
                if now - float(z.get('last_seen_time', now)) > self.zone_dormant_after_s:
                    z['active'] = False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _zone_polygon_for_zone(self, zone):
        c = np.asarray(zone['centroid'], dtype=float)
        r = max(0.75, float(zone.get('polygon_radius', 1.0)))
        pts = [
            Point32(x=float(c[0] - r), y=float(c[1] - r), z=0.0),
            Point32(x=float(c[0] + r), y=float(c[1] - r), z=0.0),
            Point32(x=float(c[0] + r), y=float(c[1] + r), z=0.0),
            Point32(x=float(c[0] - r), y=float(c[1] + r), z=0.0),
        ]
        poly = Polygon()
        poly.points = pts
        return poly

    def publish_zones(self):
        now_msg = self.get_clock().now().to_msg()

        with self.lock:
            for zid, z in list(self.zones.items()):
                try:
                    zm = ZoneMsg()
                    zm.header = Header()
                    zm.header.stamp = now_msg
                    zm.header.frame_id = 'map'

                    if hasattr(zm, 'zone_id'):
                        zm.zone_id = int(zid)
                    if hasattr(zm, 'floor_id'):
                        zm.floor_id = int(z.get('floor_id', -1))

                    # centroid Pose
                    if hasattr(zm, 'centroid'):
                        p = Pose()
                        p.position.x = float(z['centroid'][0])
                        p.position.y = float(z['centroid'][1])
                        p.position.z = 0.0
                        p.orientation.w = 1.0
                        zm.centroid = p

                    if hasattr(zm, 'polygon'):
                        zm.polygon = self._zone_polygon_for_zone(z)

                    if hasattr(zm, 'room_ids'):
                        zm.room_ids = list(z.get('room_ids', []))
                    if hasattr(zm, 'keyframe_ids'):
                        zm.keyframe_ids = sorted(list(z.get('keyframe_ids', [])))

                    if hasattr(zm, 'top_labels'):
                        zm.top_labels = [z.get('top_label', '')] if z.get('top_label', '') else []
                    if hasattr(zm, 'top_label_confidences'):
                        zm.top_label_confidences = [float(z.get('top_conf', 0.0))] if z.get('top_label', '') else []
                    if hasattr(zm, 'confidence'):
                        zm.confidence = float(z.get('confidence', 0.0))

                    if hasattr(zm, 'supporting_planes'):
                        zm.supporting_planes = []
                    if hasattr(zm, 'supporting_wall_ids'):
                        zm.supporting_wall_ids = list(z.get('supporting_wall_ids', []))

                    if hasattr(zm, 'action'):
                        # publish CREATE on first sight, UPDATE afterwards
                        zm.action = 0 if int(z.get('version', 1)) == 1 else 1
                    if hasattr(zm, 'version'):
                        zm.version = int(z.get('version', 1))

                    self.zone_pub.publish(zm)
                except Exception as e:
                    self.get_logger().error(f"Error publishing zone {zid}: {e}")

    def publish_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        idx = 0

        with self.lock:
            # zone centroid markers + text
            for zid, z in self.zones.items():
                c = np.asarray(z['centroid'], dtype=float)

                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = 'map'
                m.ns = self.marker_ns
                m.id = idx
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(c[0])
                m.pose.position.y = float(c[1])
                m.pose.position.z = 0.2
                m.pose.orientation.w = 1.0
                m.scale.x = 0.45
                m.scale.y = 0.45
                m.scale.z = 0.45
                cid = (int(zid) * 37) % 255
                m.color.r = ((cid * 97) % 255) / 255.0
                m.color.g = ((cid * 71) % 255) / 255.0
                m.color.b = ((cid * 53) % 255) / 255.0
                m.color.a = 0.9
                ma.markers.append(m)
                idx += 1

                mt = Marker()
                mt.header = m.header
                mt.ns = self.marker_ns
                mt.id = idx
                mt.type = Marker.TEXT_VIEW_FACING
                mt.action = Marker.ADD
                mt.pose.position.x = float(c[0])
                mt.pose.position.y = float(c[1])
                mt.pose.position.z = 0.65
                mt.scale.z = 0.25
                mt.color.a = 1.0
                mt.color.r = 1.0
                mt.color.g = 1.0
                mt.color.b = 1.0
                mt.text = f"Z{zid}:{z.get('top_label','?')} ({len(z.get('keyframe_ids', []))})"
                ma.markers.append(mt)
                idx += 1

                # recent member lines to centroid (debug)
                line = Marker()
                line.header = m.header
                line.ns = self.marker_ns + "_links"
                line.id = idx
                line.type = Marker.LINE_LIST
                line.action = Marker.ADD
                line.scale.x = 0.03
                line.color.a = 0.8
                line.color.r = 0.2
                line.color.g = 0.8
                line.color.b = 0.2

                # only draw from current buffer keyframes to this zone
                for entry in self.buffer:
                    if self.kf_to_zone.get(int(entry['id']), None) != zid:
                        continue
                    p1 = Point()
                    p1.x = float(entry['x'])
                    p1.y = float(entry['y'])
                    p1.z = 0.05
                    p2 = Point()
                    p2.x = float(c[0])
                    p2.y = float(c[1])
                    p2.z = 0.2
                    line.points.append(p1)
                    line.points.append(p2)

                ma.markers.append(line)
                idx += 1

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
