#!/usr/bin/env python3
"""
zones_node.py - plane-aware Zones creator with keyframe-pose association.

This version supports KeyframeSemantic messages that MAY contain keyframe_id
(published from s_graphs.publish_graph()). If keyframe_id is present we attach
directly; otherwise we fall back to time/pose/proximity matching and caching.
"""

from std_msgs.msg import Header
from geometry_msgs.msg import Polygon as GPolygon, Point as GPoint, PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from collections import defaultdict, deque
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from shapely.ops import unary_union
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
import math
import traceback
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Message imports (robust)
RoomsData = None
RoomData = None
PlaneData = None
try:
    from situational_graphs_msgs.msg import RoomsData, RoomData, PlaneData
except Exception:
    # will warn later if missing
    pass

KeyframeSemantic = None
Zone = None
try:
    from situational_graphs_msgs.msg import KeyframeSemantic, Zone
except Exception:
    pass

GraphKeyframes = None
try:
    # inferred message package; adjust if different
    from situational_graphs_reasoning_msgs.msg import GraphKeyframes
    GraphKeyframes = GraphKeyframes
except Exception:
    pass

from geometry_msgs.msg import Polygon as GPolygon, Point as GPoint, PoseStamped
from std_msgs.msg import Header

# -------------------------------------------------------------------
# Rule-based mapping from objects -> canonical room labels (conservative)
# -------------------------------------------------------------------
#OBJECT_TO_LABEL = {
#   'bed': {'bedroom': 1.0},
#   'pillow': {'bedroom': 0.8},
#   'wardrobe': {'bedroom': 0.6},
#   'stove': {'kitchen': 1.0},
#   'oven': {'kitchen': 0.9},
#   'fridge': {'kitchen': 0.9},
#   'sink': {'kitchen': 0.7, 'bathroom': 0.6},
#   'microwave': {'kitchen': 0.8},
#   'toilet': {'bathroom': 1.0},
#   'shower': {'bathroom': 1.0},
#   'desk': {'office': 0.9},
#   'computer': {'office': 0.8},
#   'chair': {'office': 0.4, 'dining': 0.4},
#   'table': {'dining': 0.6, 'kitchen': 0.3},
#   'dining_table': {'dining': 1.0},
#   'toolbox': {'workshop': 1.0, 'storage': 0.7},
#   'shelf': {'storage': 0.8},
#   'bookshelf': {'office': 0.6, 'storage': 0.6},
#   'washing_machine': {'laundry': 1.0},
#}

OBJECT_TO_LABEL = {
    # ── Exact YOLO class names (lowercased): "Persons","Windows","Lights","Door" ──
    'persons': {'hallway': 0.5, 'living_room': 0.4, 'office': 0.3},
    'person':  {'hallway': 0.5, 'living_room': 0.4, 'office': 0.3},
    'people':  {'hallway': 0.5, 'living_room': 0.4, 'office': 0.3},
    'crowd':   {'hallway': 0.6, 'lobby': 0.7},

    # Windows-related objects
    'windows':      {'living_room': 0.9, 'bedroom': 0.8, 'office': 0.7, 'kitchen': 0.6},
    'window':       {'living_room': 0.9, 'bedroom': 0.8, 'office': 0.7, 'kitchen': 0.6},
    'window_frame': {'living_room': 0.8, 'bedroom': 0.7, 'office': 0.6},
    'curtain':      {'bedroom': 0.7, 'living_room': 0.6},
    'blind':        {'bedroom': 0.6, 'office': 0.7},
    'glass':        {'living_room': 0.5, 'office': 0.4},

    # Lights-related objects
    'lights':        {'hallway': 0.7, 'bedroom': 0.8, 'kitchen': 0.8, 'office': 0.8, 'living_room': 0.7},
    'light':         {'hallway': 0.7, 'bedroom': 0.8, 'kitchen': 0.8, 'office': 0.8, 'living_room': 0.7},
    'lamp':          {'bedroom': 0.8, 'office': 0.9, 'living_room': 0.7},
    'ceiling_light': {'hallway': 0.8, 'kitchen': 0.9, 'bedroom': 0.8},
    'bulb':          {'hallway': 0.6, 'bedroom': 0.6, 'office': 0.6},
    'chandelier':    {'living_room': 0.9, 'dining': 0.9},
    'flashlight':    {'hallway': 0.5},

    # Door-related objects
    'door':        {'hallway': 0.9, 'bedroom': 0.8, 'bathroom': 0.8, 'office': 0.7, 'kitchen': 0.6},
    'door_frame':  {'hallway': 0.8, 'bedroom': 0.7, 'bathroom': 0.7},
    'door_handle': {'hallway': 0.6, 'bedroom': 0.5, 'bathroom': 0.5},
    'doorway':     {'hallway': 0.9, 'entrance': 0.9},
    'gate':        {'entrance': 0.9, 'hallway': 0.6},
    'exit':        {'hallway': 0.8, 'entrance': 0.9},

    # Multi-class contextual combinations
    # typically has door + light
    'entrance': {'entrance': 1.0, 'hallway': 0.7},
    'room': {'hallway': 0.3, 'living_room': 0.5, 'bedroom': 0.5, 'office': 0.5},
    'corridor': {'hallway': 1.0},
    'passage': {'hallway': 0.8},

    # Windows + Lights combinations
    'sunny_room': {'living_room': 0.8, 'bedroom': 0.7},  # window + light
    'bright_area': {'office': 0.7, 'kitchen': 0.7},

    # Door + Person combinations
    'entrance_area': {'entrance': 0.9, 'hallway': 0.7},
    'reception': {'office': 0.8, 'lobby': 0.9},

    # Generic spatial markers
    'wall': {'hallway': 0.3, 'office': 0.2, 'bedroom': 0.2},
    'floor': {'hallway': 0.3, 'office': 0.2},
    'ceiling': {'hallway': 0.4, 'office': 0.3},
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ros_time_to_sec(rostime):
    try:
        return float(rostime.sec) + float(rostime.nanosec) * 1e-9
    except Exception:
        return 0.0

# def polygon_from_planes(planes):
#     """
#     Build convex hull polygon from centroids of plane_points inside the plane structures.
#     Supports planes list that may come from x_planes or y_planes entries.
#     """
#     pts = []
#     for p in planes:
#         points = None
#         # plane_points is used in your room data
#         if hasattr(p, 'plane_points') and getattr(p, 'plane_points', None):
#             print("appending the plane points from field plane_points")
#             points = p.plane_points
#         elif hasattr(p, 'points') and getattr(p, 'points', None):
#             points = p.points
#         else:
#             # some plane types might store a 'plane_center' directly
#             if hasattr(p, 'plane_center') and p.plane_center is not None:
#                 try:
#                     pts.append((float(p.plane_center.x), float(p.plane_center.y)))
#                 except Exception:
#                     pass
#                 continue
#         if not points:
#             continue
#         sx = 0.0; sy = 0.0; n = 0
#         for q in points:
#             try:
#                 print("Calculating sx sy and n")
#                 sx += float(q.x); sy += float(q.y); n += 1
#                 print("sx sy n",sx, sy, n)
#             except Exception:
#                 continue
#         if n == 0:
#             continue
#         pts.append((sx / n, sy / n))
#     print("Length of the appended points",len(pts))
#     # if len(pts) < 1:
#     #     return None
#     print("CHALOO")
#     try:
#         print("Kuch to hora hai shapely polygon se yaar ",ShapelyPolygon(pts).convex_hull)
#         return ShapelyPolygon(pts).convex_hull
#     except Exception:
#         print("FIR SE HUI GALTI")
#         return None
def polygon_from_planes(planes):
    """
    Build convex hull polygon from ALL plane_points (preferred).
    - supports planes with .plane_points, .points or .plane_center
    - returns a shapely geometry (Polygon / buffered LineString / buffered Point) or None
    """
    from shapely.geometry import LineString  # local import so no top-level edit needed

    pts = []
    for i, p in enumerate(planes):
        points = getattr(p, 'plane_points', None) or getattr(p, 'points', None)
        if points:
            # append all plane points projected to XY
            for q in points:
                try:
                    x = float(q.x); y = float(q.y)
                    # filter invalid numbers
                    if math.isfinite(x) and math.isfinite(y):
                        pts.append((x, y))
                except Exception:
                    continue
        else:
            # fallback to plane_center if present
            if hasattr(p, 'plane_center') and p.plane_center is not None:
                try:
                    x = float(p.plane_center.x); y = float(p.plane_center.y)
                    if math.isfinite(x) and math.isfinite(y):
                        pts.append((x, y))
                except Exception:
                    pass
            # some message variants may expose plane_center as a geometry Pose/Point field
            elif hasattr(p, 'plane_center') and getattr(p, 'plane_center', None):
                try:
                    pc = p.plane_center
                    if hasattr(pc, 'position'):
                        x = float(pc.position.x); y = float(pc.position.y)
                        if math.isfinite(x) and math.isfinite(y):
                            pts.append((x, y))
                except Exception:
                    pass

    # remove duplicates while preserving order (round to reduce floating noise)
    seen = set()
    pts_unique = []
    for x, y in pts:
        key = (round(x, 4), round(y, 4))
        if key in seen:
            continue
        seen.add(key)
        pts_unique.append((x, y))

    print("Length of the appended points", len(pts_unique))
    if len(pts_unique) == 0:
        return None

    # 1 point -> small buffered area
    if len(pts_unique) == 1:
        try:
            return ShapelyPoint(pts_unique[0]).buffer(0.2).convex_hull
        except Exception:
            return None

    # 2 points -> buffered line (use a small width appropriate for your environment)
    if len(pts_unique) == 2:
        try:
            ls = LineString(pts_unique)
            # buffer size: tune this (0.15-0.5 m typical). I pick 0.25m as conservative default.
            return ls.buffer(0.25).convex_hull
        except Exception:
            return None

    # 3+ points -> convex hull of all points
    try:
        poly = ShapelyPolygon(pts_unique).convex_hull
        if poly.is_empty:
            return None
        return poly
    except Exception:
        # debug-friendly print (you can remove in production)
        print("FIR SE HUI GALTI in polygon_from_planes:", traceback.format_exc())
        return None

def objects_to_label_vec(objects, confs):
    label_scores = defaultdict(float)
    for i, obj in enumerate(objects):
        score = float(confs[i]) if i < len(confs) else 1.0
        obj_lower = obj.lower()
        if obj_lower in OBJECT_TO_LABEL:
            for lbl, w in OBJECT_TO_LABEL[obj_lower].items():
                label_scores[lbl] += w * score
    s = sum(label_scores.values()) + 1e-9
    for k in list(label_scores.keys()):
        label_scores[k] = label_scores[k] / s
    return dict(label_scores)

# -------------------------------------------------------------------
# Zones Node
# -------------------------------------------------------------------
class ZonesNode(Node):
    def __init__(self):
        super().__init__('zones_node_plane_aware')

        # parameters
        self.declare_parameter('theta_merge', 0.65)
        self.declare_parameter('semantic_window_s', 30.0)
        self.declare_parameter('min_kf_for_room', 1)
        self.declare_parameter('kf_association_radius_m', 2.0)
        self.declare_parameter('kf_association_time_s', 60.0)
        self.declare_parameter('adjacency_distance_m', 0.75)
        self.declare_parameter('min_shared_planes_for_prefilter', 1)

        # keyframe-related caches params
        self.declare_parameter('kf_pose_cache_s', 60.0)
        self.declare_parameter('kf_time_tol_s', 0.6)   # tolerance when matching semantics->keyframes by time
        self.declare_parameter('semantic_to_kf_time_tol_s', 0.4)  # time tolerance when matching new semantic to existing kfs

        self.theta_merge = float(self.get_parameter('theta_merge').value)
        self.semantic_window_s = float(self.get_parameter('semantic_window_s').value)
        self.min_kf_for_room = int(self.get_parameter('min_kf_for_room').value)
        self.kf_association_radius_m = float(self.get_parameter('kf_association_radius_m').value)
        self.kf_association_time_s = float(self.get_parameter('kf_association_time_s').value)
        self.adjacency_distance_m = float(self.get_parameter('adjacency_distance_m').value)
        self.min_shared_planes_for_prefilter = int(self.get_parameter('min_shared_planes_for_prefilter').value)

        self.kf_pose_cache_s = float(self.get_parameter('kf_pose_cache_s').value)
        self.kf_time_tol_s = float(self.get_parameter('kf_time_tol_s').value)
        self.semantic_to_kf_time_tol_s = float(self.get_parameter('semantic_to_kf_time_tol_s').value)

        # internal data structures
        # rooms and zones as in your original design
        self.rooms = {}
        self.zones = {}
        self.next_zone_id = 1

        # kf_semantics: keyed by keyframe id -> aggregated dict {objects, confs, stamp (last), pose(if semantics gave it), floor, room_id if semantics provided, count}
        self.kf_semantics = {}

        # keyframe_poses: keyed by keyframe id -> {'pose':(x,y,z), 'stamp':float}
        self.keyframe_poses = {}
        # time-ordered deque of (stamp, kf_id)
        self.keyframe_time_index = deque()

        # semantic cache: deque of (stamp, sem_entry) for semantics NOT yet matched to a keyframe
        # sem_entry: { 'objects':[], 'confs':[], 'stamp':float, 'pose':(x,y,z) or None, 'floor':int, 'room_id':int or -1, 'raw_msg': msg}
        self.semantic_cache = deque()

        # QoS and publishers/subscribers
        q = QoSProfile(depth=10)
        q.durability = DurabilityPolicy.TRANSIENT_LOCAL
        q.history = HistoryPolicy.KEEP_LAST

        self.debug_pub = self.create_publisher(Marker, "shapely_debug", 10)

        self.zone_pub = None
        if Zone is not None:
            self.zone_pub = self.create_publisher(Zone, 'zones', q)

        room_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriptions
        if RoomsData is not None:
            #self.room_sub = self.create_subscription(RoomsData, '/room_segmentation/room_data_assigned', self.cb_roomsdata, 10)
            self.room_sub = self.create_subscription(RoomsData,'/room_segmentation/room_data_assigned',self.cb_roomsdata,room_qos)
            self.get_logger().info("Subscribed to room_segmentation/room_data_assigned")
        else:
            self.get_logger().warning('RoomsData msg not available. Please build situational_graphs_msgs.')



        if KeyframeSemantic is not None:
            self.kf_sem_sub = self.create_subscription(
                KeyframeSemantic, '/s_graphs/keyframe_semantic', self.cb_kf_sem, 200)
            self.get_logger().info("Subscribed to s_graphs/keyframe_semantic")
        else:
            self.get_logger().warning('KeyframeSemantic msg not importable; ensure messages built and installed.')

        # Graph keyframes subscription (populate canonical keyframe poses)
        if GraphKeyframes is not None:
            try:
                self.kf_pose_sub = self.create_subscription(GraphKeyframes, 's_graphs/graph_keyframes', self.cb_graph_keyframes, 10)
                self.get_logger().info("Subscribed to s_graphs/graph_keyframes")
            except Exception as e:
                self.get_logger().warning("Failed to subscribe to GraphKeyframes: " + str(e))
        else:
            self.get_logger().warning("GraphKeyframes msg type not found. Please check message package.")

        self.get_logger().info("Zones Node initialized (caches ready)")


    def visualize_point(self, pt, marker_id=0):

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "shapely_points"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(pt.x)
        marker.pose.position.y = float(pt.y)
        marker.pose.position.z = 0.1

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.debug_pub.publish(marker)

    def visualize_polygon(self, poly, marker_id=100):

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "shapely_polygon"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        print(poly.exterior.coords)

        for x, y in poly.exterior.coords:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        self.debug_pub.publish(marker)

    # --------------------------
    # Semantic callback (now supports keyframe_id fast-path)
    # --------------------------
    def cb_kf_sem(self, msg):
        """
        Handle incoming semantic messages that may OR MAY NOT contain keyframe_id.
        Fast-path: if msg.keyframe_id present -> attach straight to that keyframe.
        Otherwise try to match the semantic to the nearest known keyframe:
         - time nearest within semantic_to_kf_time_tol_s
         - else spatial nearest to keyframe_poses within kf_association_radius_m
         - if matched: aggregate into self.kf_semantics[kf_id]
         - else: append to semantic_cache for later matching
        """
        try:
            print("===========ENTERED KF CALLBACK===============")
            # stamp from header preferred, fallback to msg.stamp or now
            try:
                stamp = ros_time_to_sec(msg.header.stamp)
            except Exception:
                stamp = getattr(msg, 'stamp', 0.0) or time.time()

            # quick gather of fields
            objects = list(getattr(msg, 'objects', []))
            confs = list(getattr(msg, 'object_confidence', []))
            floor = int(getattr(msg, 'floor_id', -1)) if hasattr(msg, 'floor_id') else -1
            room_id = int(getattr(msg, 'room_id', -1)) if hasattr(msg, 'room_id') else -1

            # try to find a pose in semantic message: several possible fields depending on publisher
            pose_tuple = None
            if hasattr(msg, 'pose') and getattr(msg, 'pose', None) is not None:
                print("-----------Found the pose---------")
                try:
                    ps = msg.pose
                    if hasattr(ps, 'pose'):
                        p = ps.pose
                    else:
                        p = ps
                    pose_tuple = (float(p.position.x), float(p.position.y), float(p.position.z))
                    print("----------Adding the pose----------")
                except Exception:
                    pose_tuple = None
            if pose_tuple is None and hasattr(msg, 'odom') and getattr(msg, 'odom', None) is not None:
                try:
                    od = msg.odom
                    if hasattr(od, 'pose') and hasattr(od.pose, 'pose'):
                        p = od.pose.pose
                        pose_tuple = (float(p.position.x), float(p.position.y), float(p.position.z))
                except Exception:
                    pose_tuple = None

            # Build semantic entry
            sem_entry = {
                'objects': objects,
                'confs': confs,
                'stamp': stamp,
                'pose': pose_tuple,
                'floor': floor,
                'room_id': room_id,
                'raw_msg': msg
            }

            # ----- FAST PATH: if message contains canonical keyframe id, attach directly -----
            kfid = getattr(msg, 'keyframe_id', None)
            if kfid is not None:
                print("/////////FOUND KF ID IN MESSAGE////////////")
                try:
                    kfid_int = int(kfid)
                except Exception:
                    kfid_int = -1
                print("KFID IS: ", kfid_int)
                if kfid_int >= 0:
                    # Attach directly into kf_semantics (aggregate)
                    print("______________Sending kf_id to aggregate the semantics__________________________")
                    self.keyframe_poses[kfid_int] = {'pose': pose_tuple, 'stamp': sem_entry['stamp']}
                    self.keyframe_time_index.append((sem_entry['stamp'], int(kfid)))
                    self._aggregate_semantic_to_kf(kfid_int, sem_entry)
                    # If the incoming semantic provided a pose but no keyframe pose exists yet, we keep it in kf_semantics
                    # so later room association will pick it up once graph keyframes arrive.
                    self.get_logger().debug(f"Fast-path: semantic attached to keyframe {kfid_int}")
                    return

            # ----- FALLBACK: No keyframe_id provided, try matching -----
            matched_kf = None
            best_dt = float('inf')
            tol = self.semantic_to_kf_time_tol_s

            # 1) time-based nearest keyframe (search newest first)
            for kf_stamp, kfid in reversed(self.keyframe_time_index):
                dt = abs(kf_stamp - stamp)
                if dt <= tol and dt < best_dt:
                    best_dt = dt; matched_kf = int(kfid)
                # heuristic break: if kf_stamp < stamp - tol, older ones won't match
                if kf_stamp < stamp - tol:
                    break

            # 2) spatial fallback if no time match and semantic has pose
            if matched_kf is None and sem_entry['pose'] is not None:
                px, py, pz = sem_entry['pose']
                best_dist = float('inf')
                for kfid, rec in self.keyframe_poses.items():
                    kx, ky, kz = rec['pose']
                    d = math.hypot(kx - px, ky - py)
                    if d <= self.kf_association_radius_m and d < best_dist:
                        best_dist = d; matched_kf = int(kfid)

            if matched_kf is not None:
                self._aggregate_semantic_to_kf(matched_kf, sem_entry)
                self.get_logger().debug(f"Semantic time/pose matched to kf {matched_kf} (dt={best_dt if best_dt!=float('inf') else 'n/a'})")
            else:
                # buffer it for later association
                self.semantic_cache.append((stamp, sem_entry))
                print(self.semantic_cache)
                #self._prune_semantic_cache()
                self.get_logger().debug("Semantic buffered (no keyframe match yet)")

        except Exception:
            self.get_logger().error("Error in cb_kf_sem: " + traceback.format_exc())

    def _aggregate_semantic_to_kf(self, kfid, sem_entry):
        """
        Merge a semantic entry into kf_semantics[kfid]. Simple additive aggregation:
        - append objects and confs (we can keep duplicates and let compute_room_signature handle smoothing)
        - update last stamp and pose if available
        """
        kfid = int(kfid)
        ent = self.kf_semantics.get(kfid)
        if ent is None:
            ent = {'objects': [], 'confs': [], 'stamp': 0.0, 'pose': None, 'floor': -1, 'room_id': -1, 'count': 0}
        # append objects
        if sem_entry.get('objects', None):
            ent['objects'].extend(sem_entry['objects'])
        if sem_entry.get('confs', None):
            ent['confs'].extend(sem_entry['confs'])
        # update stamp/pose/floor/room_id
        ent['stamp'] = max(ent.get('stamp', 0.0), sem_entry.get('stamp', 0.0))
        if sem_entry.get('pose') is not None:
            ent['pose'] = sem_entry.get('pose')
        if sem_entry.get('floor', -1) != -1:
            ent['floor'] = sem_entry.get('floor')
        if sem_entry.get('room_id', -1) != -1:
            ent['room_id'] = sem_entry.get('room_id')
        ent['count'] = ent.get('count', 0) + 1
        self.kf_semantics[kfid] = ent
        # prune old semantics if needed
        # Need to add a check here somewhere if the pruning is working correct or not
        print("Aggregated the semantic now sending to pruneing")
        print(self.kf_semantics)
        #self._prune_kf_semantics()

    # --------------------------
    # Graph keyframes callback
    # --------------------------
    def cb_graph_keyframes(self, msg):
        """
        Populate keyframe poses cache and attempt to attach any buffered semantics that now match.
        msg.keyframes expected to be a list of keyframe entries with .id and .pose and .header.stamp
        """
        try:
            # determine top-level msg stamp
            try:
                top_stamp = ros_time_to_sec(msg.header.stamp)
            except Exception:
                top_stamp = time.time()

            kflist = getattr(msg, 'keyframes', None)
            if kflist is None:
                # maybe msg itself is a single keyframe entry
                kflist = [msg]

            for kf in kflist:
                try:
                    kfid = int(getattr(kf, 'id', -1))
                    if kfid < 0:
                        continue
                    try:
                        kf_stamp = ros_time_to_sec(kf.header.stamp)
                        if kf_stamp == 0.0:
                            kf_stamp = top_stamp
                    except Exception:
                        kf_stamp = top_stamp
                    pose = getattr(kf, 'pose', None)
                    if pose is None:
                        continue
                    px = float(pose.position.x); py = float(pose.position.y); pz = float(pose.position.z)

                    # Remove existing index entries for this kfid (avoid duplicates)
                    if any(True for _ in self.keyframe_time_index if _[1] == kfid):
                        self.keyframe_time_index = deque([(s,i) for (s,i) in self.keyframe_time_index if i != kfid])

                    # Insert newest at right (time-ordered)
                    self.keyframe_poses[kfid] = {'pose': (px, py, pz), 'stamp': kf_stamp}
                    self.keyframe_time_index.append((kf_stamp, int(kfid)))

                except Exception:
                    continue

            # prune keyframe cache
            #self._prune_keyframe_cache()
            # try to attach any buffered semantics that now fall within tolerance for these new keyframes
            self._attach_buffered_semantics_to_keyframes()
        except Exception:
            self.get_logger().error("Error in cb_graph_keyframes: " + traceback.format_exc())

    def _attach_buffered_semantics_to_keyframes(self):
        """Try to match buffered semantic messages to the known keyframe poses/time."""
        if not self.semantic_cache:
            return
        remaining = deque()
        while self.semantic_cache:
            stamp, sem = self.semantic_cache.popleft()
            matched_kf = None
            best_dt = float('inf')
            tol = self.semantic_to_kf_time_tol_s
            # try time match first
            for kf_stamp, kfid in reversed(self.keyframe_time_index):
                dt = abs(kf_stamp - stamp)
                if dt <= tol and dt < best_dt:
                    best_dt = dt; matched_kf = int(kfid)
                if kf_stamp < stamp - tol:
                    break
            # if no time match, try spatial if sem has pose
            if matched_kf is None and sem.get('pose') is not None:
                px, py, pz = sem['pose']
                best_dist = float('inf')
                for kfid, rec in self.keyframe_poses.items():
                    kx, ky, kz = rec['pose']
                    d = math.hypot(kx - px, ky - py)
                    if d <= self.kf_association_radius_m and d < best_dist:
                        best_dist = d; matched_kf = int(kfid)
            if matched_kf is not None:
                self._aggregate_semantic_to_kf(matched_kf, sem)
            else:
                # keep for future
                remaining.append((stamp, sem))
        self.semantic_cache = remaining
        print(self.semantic_cache)
        #self._prune_semantic_cache()

    # --------------------------
    # RoomsData callback
    # --------------------------
    def cb_roomsdata(self, msg):
        try:
            print("!!!!!!!!!!!! ENTERED ROOM CALLBACK !!!!!!!!!!!!!!!!!!!!!!!!!!")
            # capture header stamp if present (used for time based matching)
            try:
                self.last_room_stamp = ros_time_to_sec(msg.header.stamp)
            except Exception:
                self.last_room_stamp = 0.0

            try:
                print("-------------------ROOM MILGAYA MITTAR---------------------")
                rooms = msg.rooms
            except Exception:
                rooms = [msg]

            print("rooms kitne hai abhi? ", rooms.__sizeof__())
            for r in rooms:
                rid = int(getattr(r, 'id', getattr(r, 'room_id', -1)))
                print("Room ID is: ", rid)
                if rid < 0:
                    continue
                floor_id = int(getattr(r, 'floor_id', 0))

                # collect plane structs and ids (we will use x_planes and y_planes)
                planes_combined = []
                plane_ids = set()
                # x_planes & y_planes typically present
                try:
                    xps = getattr(r, 'x_planes', [])
                    yps = getattr(r, 'y_planes', [])
                    print("yps is", yps)
                except Exception:
                    xps = []; yps = []

                for p in list(xps) + list(yps):
                    planes_combined.append(p)
                    print("planes combined is ",planes_combined)
                    if hasattr(p, 'id'):
                        try:
                            plane_ids.add(int(p.id))
                        except Exception:
                            pass

                # build polygon from plane centroids
                print(len(planes_combined))
                poly = polygon_from_planes(planes_combined) if len(planes_combined) > 0 else None
                print("made polygon from the planes from the message ", poly)

                # cluster center
                cluster_center = None
                try:
                    cc = getattr(r, 'cluster_center', None)
                    if cc is not None and hasattr(cc, 'x'):
                        cluster_center = (float(cc.x), float(cc.y))
                    else:
                        rc = getattr(r, 'room_center', None)
                        if rc is not None and hasattr(rc, 'position'):
                            cluster_center = (float(rc.position.x), float(rc.position.y))
                except Exception:
                    cluster_center = None

                # update rooms data structure
                if rid not in self.rooms:
                    self.rooms[rid] = {'planes': planes_combined, 'plane_ids': plane_ids, 'walls': [], 'keyframes': set(), 'room_sig': {}, 'poly': poly, 'floor_id': floor_id, 'cluster_center': cluster_center, 'confidence': float(getattr(r, 'confidence', 1.0))}
                else:
                    ent = self.rooms[rid]
                    ent['planes'] = planes_combined
                    ent['plane_ids'] = plane_ids
                    ent['poly'] = poly
                    ent['floor_id'] = floor_id
                    ent['cluster_center'] = cluster_center
                    ent['confidence'] = float(getattr(r, 'confidence', ent.get('confidence', 1.0)))

                # associate keyframes
                associated_kfs = self.associate_keyframes_to_room_direct(rid, floor_id, poly, cluster_center)
                print("Associated the keyframes to room id now will try to create the zones", associated_kfs)
                # store associated keyframes
                for kf in associated_kfs:
                    try:
                        self.rooms[rid]['keyframes'].add(int(kf))
                    except Exception:
                        pass

                # compute room signature and decide zone merge/create
                recent_kfs = [kf for kf in self.rooms[rid]['keyframes'] if kf in self.kf_semantics]
                if len(recent_kfs) >= self.min_kf_for_room:
                    print("Computing signature")
                    room_sig = self.compute_room_signature(recent_kfs)
                    self.rooms[rid]['room_sig'] = room_sig
                    print("Trying to merge room into zone")
                    merged = self.try_merge_room_into_zone(rid, room_sig)
                    if not merged:
                        print("Nahi hua merge afsos")
                        self.create_zone_from_room(rid, room_sig)
                else:
                    self.get_logger().debug(f"Room {rid}: not enough KFs with semantics ({len(recent_kfs)}) to compute signature")
        except Exception:
            self.get_logger().error("Error in cb_roomsdata: " + traceback.format_exc())

    # --------------------------
    # Association: keyframes to room
    # --------------------------
    def associate_keyframes_to_room_direct(self, rid, floor_id, poly, cluster_center):
        """
        Strategy:
         1) any kf in kf_semantics that has room_id == rid (sem fast-path) and is recent
         2) time-based: match keyframe timestamps near last_room_stamp (if present)
         3) polygon containment using kf pose (kf_semantics.pose or keyframe_poses)
         4) proximity to cluster_center
        """
        now = self.get_clock().now().nanoseconds / 1e9
        result = set()
        print("Doing association do not disturb")

        # 1) direct mapping from semantics (if semantic entries had room_id)
        for kfid, ent in self.kf_semantics.items():
            print("Something is wrong")
            if ent.get('room_id', -1) == rid:
                if now - ent.get('stamp', 0.0) > self.kf_association_time_s:
                    continue
                if ent.get('floor', -1) != -1 and ent.get('floor', -1) != floor_id:
                    continue
                result.add(int(kfid))
        if result:
            return list(result)

        # 2) time-based matching using last_room_stamp if available
        room_stamp = getattr(self, 'last_room_stamp', 0.0)
        if room_stamp and room_stamp > 1.0:
            tol = self.kf_time_tol_s
            for kf_stamp, kfid in reversed(self.keyframe_time_index):
                print("Time based matching but with wrong topic", abs(kf_stamp - room_stamp))
                if abs(kf_stamp - room_stamp) <= tol:
                    # check recency & floor if semantic info exists for this kf
                    sement = self.kf_semantics.get(int(kfid), None)
                    if sement:
                        if now - sement.get('stamp', 0.0) > self.kf_association_time_s:
                            continue
                        if sement.get('floor', -1) != -1 and sement.get('floor', -1) != floor_id:
                            continue
                    result.add(int(kfid))
            if result:
                return list(result)

        # 3) polygon containment using kf poses
        print("Do we have poly? ", poly)
        if poly is not None:
            # check semantics that have pose first
            print("Pose matching on the bases of pose stored in room and semantic keyframe")
            print("Lets see the size of kf semantics",self.kf_semantics)
            for kfid, sem in self.kf_semantics.items():
                print("kf id is this mere dost",kfid)
                pos = sem.get('pose', None)
                print(pos)
                print(self.keyframe_poses)
                if pos is None and int(kfid) in self.keyframe_poses:
                    pos = self.keyframe_poses[int(kfid)]['pose']
                    print("pos is this", pos)
                if pos is None:
                    continue
                # recency and floor
                print(now - sem.get('stamp', 0.0) - self.kf_association_time_s)
                # if now - sem.get('stamp', 0.0) > self.kf_association_time_s:
                #     continue
                # if sem.get('floor', -1) != -1 and sem.get('floor', -1) != floor_id:
                #     continue
                print("Before even trying")
                pt = ShapelyPoint(pos[0], pos[1])
                print(pt)
                print(poly.contains(pt))
                self.visualize_point(pt)
                self.visualize_polygon(poly)
                try:
                    pt = ShapelyPoint(pos[0], pos[1])
                    if poly.contains(pt) or poly.touches(pt):
                        result.add(int(kfid))
                except Exception: 
                    print("Problem in poly ni ho paya kuch")
                    continue
            # also check pose-only keyframe_poses not represented in kf_semantics
            for kfid, rec in self.keyframe_poses.items():
                if int(kfid) in result:
                    continue
                if now - rec.get('stamp', 0.0) > self.kf_association_time_s:
                    continue
                try:
                    pt = ShapelyPoint(rec['pose'][0], rec['pose'][1])
                    if poly.contains(pt) or poly.touches(pt):
                        result.add(int(kfid))
                except Exception:
                    continue
            if result:
                return list(result)

        # 4) proximity to cluster_center fallback
        print("Aagaye cluster centering pe?", cluster_center)
        if cluster_center is not None:
            print("Kya hora hai bhai?")
            cx, cy = cluster_center
            # check semantics with pose first
            for kfid, sem in self.kf_semantics.items():
                pos = sem.get('pose', None)
                if pos is None and int(kfid) in self.keyframe_poses:
                    pos = self.keyframe_poses[int(kfid)]['pose']
                if pos is None:
                    continue
                if now - sem.get('stamp', 0.0) > self.kf_association_time_s:
                    continue
                if sem.get('floor', -1) != -1 and sem.get('floor', -1) != floor_id:
                    continue
                dx = pos[0] - cx; dy = pos[1] - cy
                if math.hypot(dx, dy) <= self.kf_association_radius_m:
                    result.add(int(kfid))
            # check pose-only keyframes
            for kfid, rec in self.keyframe_poses.items():
                if int(kfid) in result:
                    continue
                if now - rec.get('stamp', 0.0) > self.kf_association_time_s:
                    continue
                dx = rec['pose'][0] - cx; dy = rec['pose'][1] - cy
                if math.hypot(dx, dy) <= self.kf_association_radius_m:
                    result.add(int(kfid))
            if result:
                return list(result)

        return []

    # --------------------------
    # Semantic fusion and similarity
    # --------------------------
    def compute_room_signature(self, kf_list):
        counts = {}
        total = 0.0
        for kf in kf_list:
            ent = self.kf_semantics.get(kf)
            if ent is None:
                continue
            objs = ent.get('objects', []); confs = ent.get('confs', [])
            for i, obj in enumerate(objs):
                key = obj.lower()
                c = float(confs[i]) if i < len(confs) else 1.0
                counts[key] = counts.get(key, 0.0) + c
                total += c
        V = max(len(counts), 1)
        alpha0 = 0.5
        denom = total + alpha0 * V
        room_prob = {}
        for k, v in counts.items():
            room_prob[k] = (v + alpha0) / denom
        return room_prob

    def semantic_similarity(self, a, b):
        keys = set(list(a.keys()) + list(b.keys()))
        if len(keys) == 0:
            return 0.0
        va = np.array([a.get(k,0.0) for k in keys])
        vb = np.array([b.get(k,0.0) for k in keys])
        na = np.linalg.norm(va); nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    # --------------------------
    # Zone merge / create logic
    # --------------------------
    def try_merge_room_into_zone(self, room_id, room_sig):
        best_z = None; best_sim = 0.0
        r = self.rooms[room_id]
        for zid, z in self.zones.items():
            if z.get('floor_id', None) != r.get('floor_id', None):
                continue
            # structural adjacency by shared plane ids
            if len(z.get('support_plane_ids', set()).intersection(r.get('plane_ids', set()))) == 0:
                # fallback to polygon adjacency
                if z.get('poly') is None or r.get('poly') is None:
                    continue
                if not (z['poly'].touches(r['poly']) or z['poly'].distance(r['poly']) < self.adjacency_distance_m):
                    continue
            sim = self.semantic_similarity(room_sig, z.get('label_vec', {}))
            if sim > best_sim:
                best_sim = sim; best_z = zid
        if best_z is not None and best_sim >= self.theta_merge:
            # merge
            self.zones[best_z]['rooms'].add(room_id)
            # recompute polygon union
            polys = [self.rooms[rid]['poly'] for rid in self.zones[best_z]['rooms'] if self.rooms[rid]['poly'] is not None]
            if polys:
                union = unary_union(polys)
                self.zones[best_z]['poly'] = union.convex_hull
            # update support plane ids
            sup = set(self.zones[best_z].get('support_plane_ids', set()))
            sup.update(self.rooms[room_id].get('plane_ids', set()))
            self.zones[best_z]['support_plane_ids'] = sup
            # recompute label vector (normalize)
            agg = {}
            for rid in self.zones[best_z]['rooms']:
                rsig = self.rooms[rid].get('room_sig', {})
                for k,v in rsig.items():
                    agg[k] = agg.get(k,0.0) + v
            s = sum(agg.values()) + 1e-9
            for k in agg: agg[k] = agg[k] / s
            self.zones[best_z]['label_vec'] = agg
            self.zones[best_z]['version'] = self.zones[best_z].get('version', 0) + 1
            self.zones[best_z]['confidence'] = max(self.zones[best_z].get('confidence', 0.0), self.rooms[room_id].get('confidence', 0.0))
            self.publish_zone(best_z, action=1)
            return True
        return False

    def create_zone_from_room(self, room_id, room_sig):
        zid = self.next_zone_id; self.next_zone_id += 1
        r = self.rooms[room_id]
        self.zones[zid] = {
            'rooms': set([room_id]),
            'label_vec': room_sig,
            'poly': r.get('poly'),
            'support_plane_ids': set(r.get('plane_ids', set())),
            'supporting_planes': r.get('planes', []),
            'supporting_wall_ids': [getattr(w, 'id', None) for w in r.get('walls', [])],
            'version': 1,
            'confidence': r.get('confidence', 1.0),
            'floor_id': r.get('floor_id', 0)
        }
        self.publish_zone(zid, action=0)

    # --------------------------
    # Publish Zone message
    # --------------------------
    def publish_zone(self, zid, action=1):
        if Zone is None:
            self.get_logger().warning("Zone msg type not available; skipping publish.")
            return
        z = self.zones[zid]
        msg = Zone()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.zone_id = int(zid)
        msg.floor_id = int(z.get('floor_id', 0))
        # centroid
        centx = 0.0; centy = 0.0; ncent = 0
        for rid in z['rooms']:
            p = self.rooms[rid].get('poly')
            if p is None: continue
            c = p.centroid
            centx += c.x; centy += c.y; ncent += 1
        if ncent > 0:
            msg.centroid.position.x = float(centx / ncent)
            msg.centroid.position.y = float(centy / ncent)
            msg.centroid.position.z = 0.0
        # polygon -> geometry_msgs.Polygon
        try:
            gpoly = GPolygon()
            if z.get('poly') is not None:
                for x, y in list(z['poly'].exterior.coords):
                    pt = GPoint(); pt.x = float(x); pt.y = float(y); pt.z = 0.0
                    gpoly.points.append(pt)
            msg.polygon = gpoly
        except Exception:
            pass
        msg.room_ids = list(z['rooms'])
        # collect keyframes
        kfs = []
        for rid in z['rooms']:
            kfs.extend(list(self.rooms[rid].get('keyframes', [])))
        msg.keyframe_ids = kfs
        labels = list(z.get('label_vec', {}).keys())
        confs = [float(z.get('label_vec', {}).get(l, 0.0)) for l in labels]
        msg.top_labels = labels
        msg.top_label_confidences = confs
        msg.confidence = float(z.get('confidence', 0.0))
        try:
            msg.supporting_planes = list(z.get('supporting_planes', []))
        except Exception:
            pass
        try:
            msg.supporting_wall_ids = list(z.get('supporting_wall_ids', []))
        except Exception:
            pass
        msg.action = int(action)
        msg.version = int(z.get('version', 1))
        self.zone_pub.publish(msg)

    # --------------------------
    # Prune caches
    # --------------------------
    def _prune_semantic_cache(self):
        cutoff = time.time() - max(self.semantic_window_s, self.kf_association_time_s)
        while self.semantic_cache and self.semantic_cache[0][0] < cutoff:
            self.semantic_cache.popleft()

    def _prune_kf_semantics(self):
        cutoff = time.time() - max(self.semantic_window_s, self.kf_association_time_s)
        to_del = []
        for kf, ent in list(self.kf_semantics.items()):
            if ent.get('stamp', 0.0) < cutoff:
                to_del.append(kf)
        for kf in to_del:
            try:
                del self.kf_semantics[kf]
            except KeyError:
                pass

    def _prune_keyframe_cache(self):
        cutoff = time.time() - self.kf_pose_cache_s
        while self.keyframe_time_index and self.keyframe_time_index[0][0] < cutoff:
            old_stamp, kfid = self.keyframe_time_index.popleft()
            rec = self.keyframe_poses.get(kfid, None)
            if rec is not None and rec.get('stamp', 0.0) <= old_stamp + 1e-6:
                try:
                    del self.keyframe_poses[kfid]
                except KeyError:
                    pass

# Node entrypoint
def main(args=None):
    rclpy.init(args=args)
    node = ZonesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
