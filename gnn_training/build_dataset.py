#!/usr/bin/env python3
"""
build_dataset.py — Convert subgraph JSONs + GT poses into a PyG training dataset.

Usage:
    python build_dataset.py \
        --subgraphs /tmp/subgraphs \
        --gt /home/subhanshu/LangLoop/s_graphs/gt/office.txt \
        --output dataset.pt \
        --pos_thresh 2.0 \
        --neg_thresh 5.0
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data


# ── Node feature dimensions ──
NODE_TYPE_DIM = 4       # one-hot: [keyframe, plane, room, object]
CLIP_DIM = 512
SCAN_PROFILE_DIM = 16   # scan range profile bins
PE_DIM = 4              # sin(x/λ), cos(x/λ), sin(y/λ), cos(y/λ)
FEATURE_DIM = 533       # CLIP(512) + floor_level(1) + scan_profile(16) + PE(4)
TOTAL_NODE_DIM = NODE_TYPE_DIM + FEATURE_DIM  # 537
PE_WAVELENGTH = 5.0     # meters — tuned for typical warehouse aisle spacing

# Edge types
EDGE_TYPES = {"observes": 0, "spatial": 1, "inside": 2, "bounded_by": 3, "sees": 4}
EDGE_FEAT_DIM = 4       # [distance, bearing, angle, plane_dist]

# Object classes
OBJECT_CLASSES = ["Table", "Tree", "Desks", "Computer Monitor", "Pillar"]
NUM_OBJ_CLASSES = len(OBJECT_CLASSES)

# Plane types
PLANE_TYPES = {"x_vert": 0, "y_vert": 1, "horizontal": 2}


def parse_gt(gt_path):
    """Parse ground truth file → list of (timestamp, x, y, z)."""
    gt_poses = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t = float(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            gt_poses.append((t, x, y, z))
    gt_poses.sort(key=lambda p: p[0])
    return gt_poses


def interpolate_gt(gt_poses, query_timestamp):
    """Interpolate GT position at query_timestamp."""
    if not gt_poses:
        return None

    # Clamp to range
    if query_timestamp <= gt_poses[0][0]:
        return gt_poses[0][1:4]
    if query_timestamp >= gt_poses[-1][0]:
        return gt_poses[-1][1:4]

    # Binary search for bracketing poses
    lo, hi = 0, len(gt_poses) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if gt_poses[mid][0] <= query_timestamp:
            lo = mid
        else:
            hi = mid

    t0, x0, y0, z0 = gt_poses[lo]
    t1, x1, y1, z1 = gt_poses[hi]

    if abs(t1 - t0) < 1e-9:
        return (x0, y0, z0)

    alpha = (query_timestamp - t0) / (t1 - t0)
    x = x0 + alpha * (x1 - x0)
    y = y0 + alpha * (y1 - y0)
    z = z0 + alpha * (z1 - z0)
    return (x, y, z)


def compute_positional_encoding(x, y, wavelength=PE_WAVELENGTH):
    """2D sinusoidal positional encoding → [sin(x/λ), cos(x/λ), sin(y/λ), cos(y/λ)]."""
    sx = x / wavelength
    sy = y / wavelength
    return [np.sin(sx), np.cos(sx), np.sin(sy), np.cos(sy)]


def encode_keyframe_features(features):
    """Encode keyframe node: type_onehot + [clip_512 + floor_level + scan_profile_16 + pe_4]."""
    feat = np.zeros(TOTAL_NODE_DIM, dtype=np.float32)
    feat[0] = 1.0  # type = keyframe

    clip = features.get("clip", [])
    for i, val in enumerate(clip[:CLIP_DIM]):
        feat[NODE_TYPE_DIM + i] = val

    feat[NODE_TYPE_DIM + CLIP_DIM] = float(features.get("floor_level", 0))

    # Scan range profile (16 bins) — heading-invariant via circular shift
    scan_profile = features.get("scan_range_profile", [])
    for i, val in enumerate(scan_profile[:SCAN_PROFILE_DIM]):
        feat[NODE_TYPE_DIM + CLIP_DIM + 1 + i] = val

    # Positional encoding (4 dims)
    pe = features.get("pe", [])
    for i, val in enumerate(pe[:PE_DIM]):
        feat[NODE_TYPE_DIM + CLIP_DIM + 1 + SCAN_PROFILE_DIM + i] = val

    return feat


def encode_plane_features(features):
    """Encode plane node: type_onehot + [nx, ny, nz, relative_d, relative_bearing,
       plane_type_onehot_3, mean_pairwise_angle, min_gap]."""
    feat = np.zeros(TOTAL_NODE_DIM, dtype=np.float32)
    feat[1] = 1.0  # type = plane

    feat[NODE_TYPE_DIM + 0] = features.get("nx", 0)
    feat[NODE_TYPE_DIM + 1] = features.get("ny", 0)
    feat[NODE_TYPE_DIM + 2] = features.get("nz", 0)
    # Use relative_d if available, fall back to absolute d
    relative_d = features.get("relative_d", features.get("d", 0))
    feat[NODE_TYPE_DIM + 3] = relative_d / 10.0  # normalize
    # Relative bearing
    feat[NODE_TYPE_DIM + 4] = features.get("relative_bearing", 0) / np.pi  # normalize [-1, 1]

    pt = features.get("plane_type", "")
    pt_idx = PLANE_TYPES.get(pt, 0)
    feat[NODE_TYPE_DIM + 5 + pt_idx] = 1.0  # 3 slots: index 5, 6, 7

    # Inter-plane aggregate features
    feat[NODE_TYPE_DIM + 8] = features.get("mean_pairwise_angle", 0) / (np.pi / 2)  # normalized
    feat[NODE_TYPE_DIM + 9] = features.get("min_gap", 0) / 10.0  # normalized

    return feat


def encode_room_features(features):
    """Encode room node: type_onehot + [cx, cy, cz, num_walls/4, pe_4]."""
    feat = np.zeros(TOTAL_NODE_DIM, dtype=np.float32)
    feat[2] = 1.0  # type = room

    cx = features.get("cx", 0)
    cy = features.get("cy", 0)
    feat[NODE_TYPE_DIM + 0] = cx / 10.0
    feat[NODE_TYPE_DIM + 1] = cy / 10.0
    feat[NODE_TYPE_DIM + 2] = features.get("cz", 0) / 10.0
    feat[NODE_TYPE_DIM + 3] = features.get("num_walls", 0) / 4.0
    # Positional encoding for room center
    pe = compute_positional_encoding(cx, cy)
    for i, val in enumerate(pe[:PE_DIM]):
        feat[NODE_TYPE_DIM + 4 + i] = val
    return feat


def encode_object_features(features):
    """Encode object node: type_onehot + [class_onehot_5 + confidence]."""
    feat = np.zeros(TOTAL_NODE_DIM, dtype=np.float32)
    feat[3] = 1.0  # type = object

    class_idx = features.get("class_idx", 0)
    if 0 <= class_idx < NUM_OBJ_CLASSES:
        feat[NODE_TYPE_DIM + class_idx] = 1.0

    feat[NODE_TYPE_DIM + NUM_OBJ_CLASSES] = features.get("confidence", 0)
    return feat


ENCODERS = {
    "keyframe": encode_keyframe_features,
    "plane": encode_plane_features,
    "room": encode_room_features,
    "object": encode_object_features,
}


def encode_edge_features(edge):
    """Encode edge features → fixed-size vector [distance, bearing, angle, plane_dist]."""
    feats = edge.get("features", {})
    return np.array([
        feats.get("distance", 0) / 10.0,         # normalize
        feats.get("bearing", 0) / np.pi,          # normalize to [-1, 1]
        feats.get("angle", 0) / (np.pi / 2),      # normalize to [0, 1]
        feats.get("plane_dist", 0) / 10.0,         # normalize
        feats.get("dist_to_center", 0) / 10.0,     # normalize
    ], dtype=np.float32)


def json_to_pyg(subgraph):
    """Convert a subgraph JSON dict → PyG Data object."""
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]

    # Encode node features
    node_features = []
    for node in nodes:
        node_type = node["type"]
        encoder = ENCODERS.get(node_type, encode_keyframe_features)
        feat = encoder(node["features"])
        node_features.append(feat)

    x = torch.tensor(np.stack(node_features), dtype=torch.float32)

    # Encode edges (bidirectional for GNN message passing)
    src_list, dst_list = [], []
    edge_attrs = []
    edge_types = []

    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        etype = EDGE_TYPES.get(edge["type"], 0)
        eattr = encode_edge_features(edge)

        # Forward direction
        src_list.append(src)
        dst_list.append(dst)
        edge_attrs.append(eattr)
        edge_types.append(etype)

        # Reverse direction (for message passing)
        src_list.append(dst)
        dst_list.append(src)
        edge_attrs.append(eattr)
        edge_types.append(etype)

    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
        edge_type = torch.zeros(0, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_type = edge_type
    data.keyframe_id = subgraph["keyframe_id"]
    data.timestamp = subgraph["timestamp_sec"]

    return data


def main():
    parser = argparse.ArgumentParser(description="Build GNN training dataset from subgraph JSONs")
    parser.add_argument("--subgraphs", default="/tmp/subgraphs", help="Directory with kf_*.json")
    parser.add_argument("--gt", default="/home/subhanshu/LangLoop/s_graphs/gt/office.txt",
                        help="Ground truth trajectory file")
    parser.add_argument("--output", default="dataset.pt", help="Output dataset file")
    parser.add_argument("--pos_thresh", type=float, default=2.0, help="Positive pair threshold (m)")
    parser.add_argument("--neg_thresh", type=float, default=5.0, help="Negative pair threshold (m)")
    args = parser.parse_args()

    # 1. Load GT
    print(f"Loading GT from {args.gt}...")
    gt_poses = parse_gt(args.gt)
    print(f"  {len(gt_poses)} GT poses loaded (t={gt_poses[0][0]:.1f} to {gt_poses[-1][0]:.1f})")

    # 2. Load subgraph JSONs
    json_files = sorted(glob.glob(os.path.join(args.subgraphs, "kf_*.json")))
    print(f"Loading {len(json_files)} subgraph JSONs...")

    graphs = []
    gt_positions = []

    for jf in json_files:
        with open(jf, 'r') as f:
            sg = json.load(f)

        # Convert to PyG
        data = json_to_pyg(sg)

        # Match timestamp to GT
        gt_pos = interpolate_gt(gt_poses, sg["timestamp_sec"])
        if gt_pos is None:
            print(f"  Warning: no GT for KF{sg['keyframe_id']} (t={sg['timestamp_sec']})")
            continue

        data.gt_x = gt_pos[0]
        data.gt_y = gt_pos[1]
        data.gt_z = gt_pos[2]

        graphs.append(data)
        gt_positions.append(gt_pos)

    print(f"  {len(graphs)} graphs with GT poses")

    # 3. Create pairs
    gt_positions = np.array(gt_positions)
    pos_pairs = []
    neg_pairs = []

    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            dist = np.linalg.norm(gt_positions[i] - gt_positions[j])
            if dist < args.pos_thresh:
                pos_pairs.append((i, j))
            elif dist > args.neg_thresh:
                neg_pairs.append((i, j))

    print(f"\n  Pairs created:")
    print(f"    Positive (< {args.pos_thresh}m): {len(pos_pairs)}")
    print(f"    Negative (> {args.neg_thresh}m): {len(neg_pairs)}")
    print(f"    Skipped (ambiguous):  {len(graphs)*(len(graphs)-1)//2 - len(pos_pairs) - len(neg_pairs)}")

    # Balance: subsample negatives to at most 3x positives
    if len(neg_pairs) > 3 * len(pos_pairs) and len(pos_pairs) > 0:
        np.random.seed(42)
        indices = np.random.choice(len(neg_pairs), size=3 * len(pos_pairs), replace=False)
        neg_pairs = [neg_pairs[i] for i in indices]
        print(f"    Negative (subsampled): {len(neg_pairs)}")

    # 4. Save dataset
    dataset = {
        "graphs": graphs,
        "pos_pairs": pos_pairs,
        "neg_pairs": neg_pairs,
        "gt_positions": gt_positions,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    torch.save(dataset, output_path)
    print(f"\nDataset saved to {output_path}")
    print(f"  Total pairs: {len(pos_pairs) + len(neg_pairs)}")


if __name__ == "__main__":
    main()
