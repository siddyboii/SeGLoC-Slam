#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SEMANTIC LOOP CLOSURE — POST-RUN METRICS ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
  Reads CSV files produced by metrics_recorder.py + loop_metrics.csv produced
  by the C++ logger in loop_mapper.cpp.  Computes:

  TRAJECTORY METRICS
    • ATE   (Absolute Trajectory Error)   — RMSE of position errors
    • RPE   (Relative Pose Error)         — frame-to-frame drift
    • Drift (% of total distance)

  LOOP CLOSURE METRICS (from loop_metrics.csv)
    • Total / Semantic / Geometry-only counts
    • CLIP similarity distribution (mean, std, histogram)
    • Semantic alpha distribution
    • ICP fitness distribution
    • Object overlap distribution
    • IQA quality gate distribution
    • Dynamicity factor distribution
    • Zone factor distribution
    • Information matrix norm distribution

  SEMANTIC COVERAGE METRICS (from keyframe_semantics.csv)
    • % keyframes with CLIP embeddings
    • % keyframes with detected objects
    • % keyframes with IQA scores
    • % keyframes with dynamicity data
    • Object class frequency distribution
    • Scene dynamicity over time

  ZONE METRICS (from zone_events.csv)
    • Number of zones created / updated / deleted
    • Average keyframes per zone
    • Average zone confidence

  GRAPH METRICS (from session_details.txt if available)
    • Total vertices, total edges
    • Optimisation computation time (from /tmp/optimization_computation_time.txt)

  OUTPUT
    • Prints a formatted table to stdout
    • Saves metrics_summary.txt
    • Generates plots (if matplotlib available) → saved as PNG

  Usage
  ─────
  python3 analyse_metrics.py /tmp/sgraphs_metrics [--loop-csv /path/to/loop_metrics.csv]
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def read_csv(path):
    """Read CSV into list of dicts."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def stamp_to_sec(row, sec_key="stamp_sec", nsec_key="stamp_nsec"):
    return float(row[sec_key]) + float(row[nsec_key]) * 1e-9


def quat_to_mat(qx, qy, qz, qw):
    """Quaternion to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def rows_to_poses(rows):
    """Convert CSV rows with x,y,z,qx,qy,qz,qw to Nx4x4 SE3 array + timestamps."""
    stamps = []
    poses = []
    for r in rows:
        t = stamp_to_sec(r)
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(
            float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))
        T[:3, 3] = [float(r["x"]), float(r["y"]), float(r["z"])]
        stamps.append(t)
        poses.append(T)
    return np.array(stamps), np.array(poses)


def associate_by_time(stamps_a, stamps_b, max_diff=0.05):
    """Find closest pairs between two timestamp arrays. Returns list of (idx_a, idx_b)."""
    pairs = []
    j_start = 0
    for i, ta in enumerate(stamps_a):
        best_j = -1
        best_dt = max_diff
        for j in range(j_start, len(stamps_b)):
            dt = abs(stamps_b[j] - ta)
            if dt < best_dt:
                best_dt = dt
                best_j = j
            if stamps_b[j] - ta > max_diff:
                break
        if best_j >= 0:
            pairs.append((i, best_j))
            j_start = best_j
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  TRAJECTORY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ate(poses_ref, poses_est, pairs):
    """
    ATE (Absolute Trajectory Error)
    First aligns est to ref using Umeyama, then computes RMSE of position errors.
    Returns: rmse, mean, median, max, std, per_pair_errors
    """
    if len(pairs) < 3:
        return None

    ref_pts = np.array([poses_ref[i][:3, 3] for i, j in pairs])
    est_pts = np.array([poses_est[j][:3, 3] for i, j in pairs])

    # Umeyama alignment (scale=False for SLAM)
    mu_ref = ref_pts.mean(axis=0)
    mu_est = est_pts.mean(axis=0)
    ref_c = ref_pts - mu_ref
    est_c = est_pts - mu_est

    H = est_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mu_ref - R @ mu_est

    # Apply alignment
    aligned = (R @ est_pts.T).T + t
    errors = np.linalg.norm(aligned - ref_pts, axis=1)

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
        "errors": errors,
    }


def compute_rpe(poses_ref, poses_est, pairs, delta=1):
    """
    RPE (Relative Pose Error)
    Measures local consistency: how well short motions are estimated.
    Returns: trans_rmse, rot_rmse (degrees), per_pair_errors
    """
    trans_errors = []
    rot_errors = []

    for k in range(len(pairs) - delta):
        i1, j1 = pairs[k]
        i2, j2 = pairs[k + delta]

        # Reference relative motion
        dT_ref = np.linalg.inv(poses_ref[i1]) @ poses_ref[i2]
        # Estimated relative motion
        dT_est = np.linalg.inv(poses_est[j1]) @ poses_est[j2]

        # Error transform
        E = np.linalg.inv(dT_ref) @ dT_est

        trans_errors.append(np.linalg.norm(E[:3, 3]))
        # Rotation error in degrees
        cos_angle = (np.trace(E[:3, :3]) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rot_errors.append(math.degrees(math.acos(cos_angle)))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_errors**2))),
        "trans_mean": float(np.mean(trans_errors)),
        "trans_max": float(np.max(trans_errors)),
        "rot_rmse": float(np.sqrt(np.mean(rot_errors**2))),
        "rot_mean": float(np.mean(rot_errors)),
        "rot_max": float(np.max(rot_errors)),
        "trans_errors": trans_errors,
        "rot_errors": rot_errors,
    }


def compute_drift(poses, stamps):
    """
    Compute total path length and end-to-end drift (% of path length).
    Works on a single trajectory (e.g., corrected - raw comparison).
    """
    if len(poses) < 2:
        return None
    positions = np.array([p[:3, 3] for p in poses])
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = float(np.sum(segment_lengths))
    end_to_end = float(np.linalg.norm(positions[-1] - positions[0]))

    return {
        "total_path_length_m": total_length,
        "end_to_end_distance_m": end_to_end,
        "total_duration_s": stamps[-1] - stamps[0],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  LOOP CLOSURE METRICS (from C++ CSV logger)
# ═══════════════════════════════════════════════════════════════════════════

def analyse_loop_metrics(rows):
    """Analyse the per-loop CSV data written by the C++ logger."""
    if not rows:
        return None

    metrics = {}
    n = len(rows)
    metrics["total_loops"] = n

    # Separate semantic vs geometry-only
    semantic_rows = [r for r in rows if r.get("has_clip", "0") == "1"]
    geom_rows = [r for r in rows if r.get("has_clip", "0") == "0"]
    metrics["semantic_loops"] = len(semantic_rows)
    metrics["geometry_only_loops"] = len(geom_rows)
    metrics["semantic_pct"] = 100.0 * len(semantic_rows) / n if n > 0 else 0

    # Distribution of each factor
    float_cols = [
        "clip_sim", "semantic_alpha", "icp_fitness",
        "object_overlap", "quality_gate", "dyn_factor", "zone_factor",
        "info_matrix_norm",
    ]
    for col in float_cols:
        vals = []
        for r in rows:
            v = r.get(col, "")
            if v and v != "N/A":
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if vals:
            arr = np.array(vals)
            metrics[f"{col}_mean"] = float(np.mean(arr))
            metrics[f"{col}_std"] = float(np.std(arr))
            metrics[f"{col}_min"] = float(np.min(arr))
            metrics[f"{col}_max"] = float(np.max(arr))
            metrics[f"{col}_median"] = float(np.median(arr))
            metrics[f"{col}_values"] = arr  # for plots

    # Accepted vs skipped (stair keyframes)
    accepted = sum(1 for r in rows if r.get("edge_added", "1") == "1")
    metrics["edges_added"] = accepted
    metrics["edges_skipped_stairs"] = n - accepted

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  SEMANTIC COVERAGE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_semantic_coverage(rows):
    """Analyse keyframe_semantics.csv."""
    if not rows:
        return None

    n = len(rows)
    m = {}
    m["total_keyframes_with_semantics"] = n

    clip_count = sum(1 for r in rows if int(r.get("clip_dim", "0")) > 0)
    obj_count = sum(1 for r in rows if int(r.get("num_objects", "0")) > 0)

    m["pct_with_clip"] = 100.0 * clip_count / n
    m["pct_with_objects"] = 100.0 * obj_count / n

    # IQA
    iqa_vals = []
    for r in rows:
        v = r.get("image_quality", "")
        if v:
            try:
                iqa_vals.append(float(v))
            except ValueError:
                pass
    iqa_count = sum(1 for v in iqa_vals if v > 0)
    m["pct_with_iqa"] = 100.0 * iqa_count / n if n else 0
    if iqa_vals:
        m["iqa_mean"] = float(np.mean(iqa_vals))
        m["iqa_std"] = float(np.std(iqa_vals))

    # Dynamicity
    dyn_vals = []
    for r in rows:
        v = r.get("scene_dynamicity", "")
        if v:
            try:
                dyn_vals.append(float(v))
            except ValueError:
                pass
    dyn_count = sum(1 for v in dyn_vals if v > 0)
    m["pct_with_dynamicity"] = 100.0 * dyn_count / n if n else 0
    if dyn_vals:
        m["dynamicity_mean"] = float(np.mean(dyn_vals))
        m["dynamicity_std"] = float(np.std(dyn_vals))
        m["dynamicity_values"] = np.array(dyn_vals)

    # Object class frequency
    all_objects = []
    for r in rows:
        labels = r.get("object_labels", "")
        if labels:
            all_objects.extend(labels.split(";"))
    m["object_class_counts"] = dict(Counter(all_objects).most_common(20))

    return m


# ═══════════════════════════════════════════════════════════════════════════
#  ZONE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_zones(rows):
    """Analyse zone_events.csv."""
    if not rows:
        return None

    m = {}
    actions = Counter(r["action"] for r in rows)
    m["zone_creates"] = actions.get("CREATE", 0)
    m["zone_updates"] = actions.get("UPDATE", 0)
    m["zone_deletes"] = actions.get("DELETE", 0)

    # Latest state of each zone
    latest = {}
    for r in rows:
        zid = int(r["zone_id"])
        if r["action"] != "DELETE":
            latest[zid] = r

    m["total_active_zones"] = len(latest)

    if latest:
        kf_counts = [int(r["num_keyframes"]) for r in latest.values()]
        confs = []
        for r in latest.values():
            try:
                confs.append(float(r["confidence"]))
            except (ValueError, KeyError):
                pass

        m["avg_keyframes_per_zone"] = float(np.mean(kf_counts))
        m["max_keyframes_per_zone"] = int(np.max(kf_counts))
        if confs:
            m["avg_zone_confidence"] = float(np.mean(confs))

    return m


# ═══════════════════════════════════════════════════════════════════════════
#  COMPUTATION TIME METRICS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_optimization_time(path="/tmp/optimization_computation_time.txt"):
    """Read the computation time file written by graph_slam.cpp."""
    if not os.path.isfile(path):
        return None
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass
    if not vals:
        return None
    arr = np.array(vals)
    return {
        "num_optimizations": len(arr),
        "mean_time_s": float(np.mean(arr)),
        "std_time_s": float(np.std(arr)),
        "min_time_s": float(np.min(arr)),
        "max_time_s": float(np.max(arr)),
        "total_time_s": float(np.sum(arr)),
        "values": arr,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH METRICS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_session_details(dump_dir):
    """Read session_details.txt from a dump directory."""
    path = os.path.join(dump_dir, "session_details.txt")
    if not os.path.isfile(path):
        return None
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                m[parts[0]] = parts[1]
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  DYNAMIC OBJECTS METRICS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_dynamic_objects(rows):
    """Analyse dynamic_objects.csv."""
    if not rows:
        return None
    m = {}
    m["total_frames"] = len(rows)
    dyns = []
    clusters = []
    for r in rows:
        try:
            dyns.append(float(r["scene_dynamicity"]))
        except (ValueError, KeyError):
            pass
        try:
            clusters.append(int(r["num_clusters"]))
        except (ValueError, KeyError):
            pass
    if dyns:
        arr = np.array(dyns)
        m["dynamicity_mean"] = float(np.mean(arr))
        m["dynamicity_std"] = float(np.std(arr))
        m["dynamicity_max"] = float(np.max(arr))
        m["pct_dynamic_frames"] = float(100.0 * np.sum(arr > 0.05) / len(arr))
    if clusters:
        arr = np.array(clusters)
        m["avg_clusters_per_frame"] = float(np.mean(arr))
        m["max_clusters"] = int(np.max(arr))
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(output_dir, trajectory_data, loop_data, sem_data, opt_data, dyn_data):
    """Generate and save all plots."""
    if not HAS_PLT:
        print("  ⚠  matplotlib not available — skipping plots")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── 1. Trajectory 2D plot ──
    if trajectory_data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if "corrected_positions" in trajectory_data:
            pos = trajectory_data["corrected_positions"]
            ax.plot(pos[:, 0], pos[:, 1], "b-",
                    label="Corrected (graph-optimised)", linewidth=1.5)
        if "raw_positions" in trajectory_data:
            pos = trajectory_data["raw_positions"]
            ax.plot(pos[:, 0], pos[:, 1], "r--",
                    label="Raw odometry", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Trajectory: Raw Odometry vs Graph-Corrected")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, "trajectory_2d.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved trajectory_2d.png")

    # ── 2. ATE over time ──
    if trajectory_data and "ate_errors" in trajectory_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(trajectory_data["ate_errors"], "b-", linewidth=0.8)
        ate_rmse = trajectory_data.get("ate_rmse", 0)
        ax.axhline(y=ate_rmse, color="r", linestyle="--",
                   label=f"RMSE = {ate_rmse:.4f} m")
        ax.set_xlabel("Pose index")
        ax.set_ylabel("Position error (m)")
        ax.set_title("ATE (Absolute Trajectory Error) per Pose")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, "ate_over_time.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved ate_over_time.png")

    # ── 3. RPE over time ──
    if trajectory_data and "rpe_trans" in trajectory_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(trajectory_data["rpe_trans"], "b-", linewidth=0.8)
        ax1.set_ylabel("Translation error (m)")
        ax1.set_title("RPE — Translation")
        ax1.grid(True, alpha=0.3)
        ax2.plot(trajectory_data["rpe_rot"], "r-", linewidth=0.8)
        ax2.set_ylabel("Rotation error (deg)")
        ax2.set_xlabel("Pair index")
        ax2.set_title("RPE — Rotation")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "rpe_over_time.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved rpe_over_time.png")

    # ── 4. Loop closure factor distributions ──
    if loop_data:
        factor_cols = ["clip_sim", "semantic_alpha", "icp_fitness",
                       "object_overlap", "quality_gate", "dyn_factor", "zone_factor"]
        available = [(c, loop_data[f"{c}_values"]) for c in factor_cols
                     if f"{c}_values" in loop_data]

        if available:
            n_plots = len(available)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
            if n_plots == 1:
                axes = [axes]
            for ax, (name, vals) in zip(axes, available):
                ax.hist(vals, bins=20, color="steelblue",
                        edgecolor="black", alpha=0.8)
                ax.axvline(np.mean(vals), color="red", linestyle="--",
                           label=f"mean={np.mean(vals):.3f}")
                ax.set_title(name.replace("_", " ").title())
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
            fig.suptitle("Loop Closure Factor Distributions", fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "loop_factor_distributions.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  📊 Saved loop_factor_distributions.png")

    # ── 5. Semantic alpha scatter (alpha vs clip_sim) ──
    if loop_data and "clip_sim_values" in loop_data and "semantic_alpha_values" in loop_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(loop_data["clip_sim_values"], loop_data["semantic_alpha_values"],
                   c="steelblue", alpha=0.6, edgecolors="black", linewidth=0.3, s=40)
        ax.axhline(y=1.0, color="gray", linestyle="--",
                   alpha=0.5, label="neutral α=1.0")
        ax.set_xlabel("CLIP Cosine Similarity")
        ax.set_ylabel("Semantic Alpha (scaling factor)")
        ax.set_title("Semantic Alpha vs CLIP Similarity per Loop Closure")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, "alpha_vs_clip.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved alpha_vs_clip.png")

    # ── 6. Optimisation time ──
    if opt_data and "values" in opt_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(opt_data["values"], "b-", linewidth=0.8)
        ax.axhline(y=opt_data["mean_time_s"], color="r", linestyle="--",
                   label=f"mean = {opt_data['mean_time_s']:.3f}s")
        ax.set_xlabel("Optimisation iteration")
        ax.set_ylabel("Time (s)")
        ax.set_title("Graph Optimisation Computation Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, "optimization_time.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved optimization_time.png")

    # ── 7. Scene dynamicity over time ──
    if dyn_data and "dynamicity_mean" in dyn_data:
        # Use raw dynamic objects data
        pass

    # ── 8. Object class bar chart ──
    if sem_data and "object_class_counts" in sem_data:
        counts = sem_data["object_class_counts"]
        if counts:
            labels = list(counts.keys())[:15]
            values = [counts[l] for l in labels]
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.barh(labels, values, color="steelblue", edgecolor="black")
            ax.set_xlabel("Detection count")
            ax.set_title("Top Detected Object Classes Across All Keyframes")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "object_class_frequency.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  📊 Saved object_class_frequency.png")

    print(f"\n  All plots saved to: {plot_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def print_section(title, items, file=None):
    """Print a formatted section."""
    sep = "─" * 60
    lines = [f"\n  ┌{sep}┐", f"  │  {title:<56}  │", f"  ├{sep}┤"]
    for k, v in items:
        if isinstance(v, np.ndarray):
            continue
        lines.append(f"  │  {k:<36} {fmt(v):>18}  │")
    lines.append(f"  └{sep}┘")
    text = "\n".join(lines)
    print(text)
    if file:
        file.write(text + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse S-Graphs semantic loop closure metrics")
    parser.add_argument("data_dir",
                        help="Directory containing CSV files from metrics_recorder.py")
    parser.add_argument("--loop-csv", default=None,
                        help="Path to loop_metrics.csv from C++ logger "
                             "(default: <data_dir>/loop_metrics.csv)")
    parser.add_argument("--dump-dir", default=None,
                        help="Path to s_graphs dump directory (for session_details.txt)")
    parser.add_argument("--opt-time-file",
                        default="/tmp/optimization_computation_time.txt",
                        help="Path to optimisation time file from graph_slam.cpp")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")

    args = parser.parse_args()
    data_dir = args.data_dir
    loop_csv_path = args.loop_csv or os.path.join(data_dir, "loop_metrics.csv")

    print("═" * 64)
    print("  SEMANTIC LOOP CLOSURE — METRICS ANALYSIS")
    print("═" * 64)
    print(f"  Data directory: {data_dir}")

    summary_path = os.path.join(data_dir, "metrics_summary.txt")
    summary_file = open(summary_path, "w")
    summary_file.write("SEMANTIC LOOP CLOSURE — METRICS SUMMARY\n")
    summary_file.write("=" * 64 + "\n")

    trajectory_plot_data = {}

    # ── 1. TRAJECTORY METRICS ──
    corrected_rows = read_csv(os.path.join(
        data_dir, "corrected_trajectory.csv"))
    raw_rows = read_csv(os.path.join(data_dir, "raw_odometry.csv"))

    if corrected_rows and raw_rows:
        stamps_corr, poses_corr = rows_to_poses(corrected_rows)
        stamps_raw, poses_raw = rows_to_poses(raw_rows)

        trajectory_plot_data["corrected_positions"] = np.array(
            [p[:3, 3] for p in poses_corr])
        trajectory_plot_data["raw_positions"] = np.array(
            [p[:3, 3] for p in poses_raw])

        pairs = associate_by_time(stamps_raw, stamps_corr)
        print(f"\n  Trajectory: {len(corrected_rows)} corrected, {len(raw_rows)} raw, "
              f"{len(pairs)} matched pairs")

        # ATE: raw odom as "reference", corrected as "estimate"
        # This shows HOW MUCH the optimiser changed the trajectory
        ate = compute_ate(poses_raw, poses_corr, pairs)
        if ate:
            trajectory_plot_data["ate_errors"] = ate["errors"]
            trajectory_plot_data["ate_rmse"] = ate["rmse"]
            print_section("ATE (Absolute Trajectory Error)", [
                ("RMSE (m)", ate["rmse"]),
                ("Mean (m)", ate["mean"]),
                ("Median (m)", ate["median"]),
                ("Max (m)", ate["max"]),
                ("Std (m)", ate["std"]),
                ("Num matched pairs", len(pairs)),
            ], summary_file)

        rpe = compute_rpe(poses_raw, poses_corr, pairs)
        if rpe:
            trajectory_plot_data["rpe_trans"] = rpe["trans_errors"]
            trajectory_plot_data["rpe_rot"] = rpe["rot_errors"]
            print_section("RPE (Relative Pose Error)", [
                ("Translation RMSE (m)", rpe["trans_rmse"]),
                ("Translation Mean (m)", rpe["trans_mean"]),
                ("Translation Max (m)", rpe["trans_max"]),
                ("Rotation RMSE (deg)", rpe["rot_rmse"]),
                ("Rotation Mean (deg)", rpe["rot_mean"]),
                ("Rotation Max (deg)", rpe["rot_max"]),
            ], summary_file)

        # Drift
        drift_corr = compute_drift(poses_corr, stamps_corr)
        drift_raw = compute_drift(poses_raw, stamps_raw)
        if drift_corr and drift_raw:
            print_section("Trajectory Statistics", [
                ("Corrected path length (m)",
                 drift_corr["total_path_length_m"]),
                ("Raw path length (m)", drift_raw["total_path_length_m"]),
                ("Duration (s)", drift_corr["total_duration_s"]),
                ("Corrected end-to-end (m)",
                 drift_corr["end_to_end_distance_m"]),
                ("Raw end-to-end (m)", drift_raw["end_to_end_distance_m"]),
            ], summary_file)
    elif corrected_rows:
        stamps_corr, poses_corr = rows_to_poses(corrected_rows)
        trajectory_plot_data["corrected_positions"] = np.array(
            [p[:3, 3] for p in poses_corr])
        drift = compute_drift(poses_corr, stamps_corr)
        if drift:
            print_section("Trajectory (corrected only)", [
                ("Poses recorded", len(corrected_rows)),
                ("Path length (m)", drift["total_path_length_m"]),
                ("Duration (s)", drift["total_duration_s"]),
            ], summary_file)
    else:
        print("\n  ⚠  No trajectory data found")

    # ── 2. LOOP CLOSURE METRICS ──
    loop_rows = read_csv(loop_csv_path)
    loop_data = analyse_loop_metrics(loop_rows)
    if loop_data:
        items = [
            ("Total loop closures", loop_data["total_loops"]),
            ("Semantically weighted", loop_data["semantic_loops"]),
            ("Geometry-only", loop_data["geometry_only_loops"]),
            ("Semantic coverage (%)", loop_data["semantic_pct"]),
            ("Edges added", loop_data["edges_added"]),
            ("Edges skipped (stairs)", loop_data["edges_skipped_stairs"]),
        ]
        for col in ["clip_sim", "semantic_alpha", "icp_fitness",
                    "object_overlap", "quality_gate", "dyn_factor", "zone_factor",
                    "info_matrix_norm"]:
            if f"{col}_mean" in loop_data:
                items.append((f"{col} mean", loop_data[f"{col}_mean"]))
                items.append((f"{col} std", loop_data[f"{col}_std"]))
                items.append((f"{col} [min, max]",
                              f"[{loop_data[f'{col}_min']:.4f}, {loop_data[f'{col}_max']:.4f}]"))
        print_section("Loop Closure Metrics", items, summary_file)
    else:
        print(f"\n  ⚠  No loop metrics CSV found at: {loop_csv_path}")
        print(f"     Add the C++ CSV logger to loop_mapper.cpp (see below)")

    # ── 3. SEMANTIC COVERAGE ──
    sem_rows = read_csv(os.path.join(data_dir, "keyframe_semantics.csv"))
    sem_data = analyse_semantic_coverage(sem_rows)
    if sem_data:
        items = [
            ("Total semantic keyframes",
             sem_data["total_keyframes_with_semantics"]),
            ("With CLIP embeddings (%)", sem_data["pct_with_clip"]),
            ("With detected objects (%)", sem_data["pct_with_objects"]),
            ("With IQA scores (%)", sem_data.get("pct_with_iqa", "N/A")),
            ("With dynamicity (%)", sem_data.get("pct_with_dynamicity", "N/A")),
        ]
        if "iqa_mean" in sem_data:
            items.append(
                ("IQA mean ± std", f"{sem_data['iqa_mean']:.4f} ± {sem_data['iqa_std']:.4f}"))
        if "dynamicity_mean" in sem_data:
            items.append(("Dynamicity mean ± std",
                          f"{sem_data['dynamicity_mean']:.4f} ± {sem_data['dynamicity_std']:.4f}"))
        if sem_data["object_class_counts"]:
            top3 = list(sem_data["object_class_counts"].items())[:3]
            items.append(("Top 3 objects", ", ".join(
                f"{k}({v})" for k, v in top3)))
        print_section("Semantic Coverage", items, summary_file)

    # ── 4. ZONE METRICS ──
    zone_rows = read_csv(os.path.join(data_dir, "zone_events.csv"))
    zone_data = analyse_zones(zone_rows)
    if zone_data:
        items = [
            ("Zone creates", zone_data.get("zone_creates", 0)),
            ("Zone updates", zone_data.get("zone_updates", 0)),
            ("Zone deletes", zone_data.get("zone_deletes", 0)),
            ("Active zones (final)", zone_data.get("total_active_zones", 0)),
            ("Avg keyframes/zone", zone_data.get("avg_keyframes_per_zone", "N/A")),
            ("Max keyframes/zone", zone_data.get("max_keyframes_per_zone", "N/A")),
            ("Avg zone confidence", zone_data.get("avg_zone_confidence", "N/A")),
        ]
        print_section("Zone Metrics", items, summary_file)

    # ── 5. DYNAMIC OBJECTS ──
    dyn_rows = read_csv(os.path.join(data_dir, "dynamic_objects.csv"))
    dyn_data = analyse_dynamic_objects(dyn_rows)
    if dyn_data:
        items = [
            ("Total DynaTrack frames", dyn_data["total_frames"]),
            ("Dynamicity mean ± std",
             f"{dyn_data.get('dynamicity_mean', 0):.4f} ± {dyn_data.get('dynamicity_std', 0):.4f}"),
            ("Max dynamicity", dyn_data.get("dynamicity_max", "N/A")),
            ("Dynamic frames (>5%)",
             f"{dyn_data.get('pct_dynamic_frames', 0):.1f}%"),
            ("Avg clusters/frame", dyn_data.get("avg_clusters_per_frame", "N/A")),
            ("Max clusters", dyn_data.get("max_clusters", "N/A")),
        ]
        print_section("DynaTrack Dynamic Objects", items, summary_file)

    # ── 6. COMPUTATION TIME ──
    opt_data = analyse_optimization_time(args.opt_time_file)
    if opt_data:
        items = [
            ("Num optimisations", opt_data["num_optimizations"]),
            ("Mean time (s)", opt_data["mean_time_s"]),
            ("Std time (s)", opt_data["std_time_s"]),
            ("Min time (s)", opt_data["min_time_s"]),
            ("Max time (s)", opt_data["max_time_s"]),
            ("Total opt time (s)", opt_data["total_time_s"]),
        ]
        print_section("Graph Optimisation Time", items, summary_file)
    else:
        print(f"\n  ⚠  No optimisation time file at {args.opt_time_file}")
        print(f"     Set `save_compute_time: true` in graph_slam.cpp params")

    # ── 7. GRAPH STRUCTURE ──
    if args.dump_dir:
        session = analyse_session_details(args.dump_dir)
        if session:
            items = [(k, v) for k, v in session.items()]
            print_section("Graph Structure (from dump)", items, summary_file)

    # ── 8. IQA OVER TIME ──
    iqa_rows = read_csv(os.path.join(data_dir, "image_quality.csv"))
    if iqa_rows:
        scores = [float(r["score"]) for r in iqa_rows]
        items = [
            ("Total IQA samples", len(scores)),
            ("Mean score", float(np.mean(scores))),
            ("Std score", float(np.std(scores))),
            ("Min / Max", f"{min(scores):.4f} / {max(scores):.4f}"),
        ]
        print_section("Image Quality (IQA)", items, summary_file)

    summary_file.close()
    print(f"\n  📄 Summary saved to: {summary_path}")

    # ── PLOTS ──
    if not args.no_plots:
        print("\n  Generating plots...")
        generate_plots(data_dir, trajectory_plot_data,
                       loop_data, sem_data, opt_data, dyn_data)

    print("\n" + "═" * 64)
    print("  ANALYSIS COMPLETE")
    print("═" * 64)


if __name__ == "__main__":
    main()
