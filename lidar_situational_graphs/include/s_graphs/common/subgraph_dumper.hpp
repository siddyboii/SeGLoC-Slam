#ifndef SUBGRAPH_DUMPER_HPP
#define SUBGRAPH_DUMPER_HPP

#include <s_graphs/common/keyframe.hpp>
#include <s_graphs/common/planes.hpp>
#include <s_graphs/common/rooms.hpp>

#include <Eigen/Eigen>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace s_graphs {

/**
 * @brief Dumps per-keyframe subgraphs as JSON for offline GNN training.
 *
 * Each JSON file contains:
 *   - Nodes: keyframe (CLIP + floor), planes (normal, distance, type),
 *            room (center, wall count), objects (class, confidence)
 *   - Edges: observes (KF→plane), spatial (plane↔plane), inside (KF→room),
 *            bounded_by (room→plane), sees (KF→object)
 */
struct SubgraphDumper {
  // YOLO classes — order determines class_idx in object nodes
  static inline const std::vector<std::string> OBJECT_CLASSES = {
      "Table", "Tree", "Desks", "Computer Monitor", "Pillar"};

  static constexpr const char* OUTPUT_DIR = "/tmp/subgraphs";

  /**
   * @brief Dump a keyframe's local subgraph as JSON.
   */
  static void dump(
      const KeyFrame::Ptr& kf,
      const std::unordered_map<int, VerticalPlanes>& x_vert_planes,
      const std::unordered_map<int, VerticalPlanes>& y_vert_planes,
      const std::unordered_map<int, HorizontalPlanes>& hort_planes,
      const std::unordered_map<int, Rooms>& rooms_vec) {
    // Create output directory if it doesn't exist
    mkdir(OUTPUT_DIR, 0755);

    // Get keyframe position
    Eigen::Vector3d kf_pos = Eigen::Vector3d::Zero();
    if (kf->node) {
      kf_pos = kf->node->estimate().translation();
    }

    // ── Collect nodes and edges ──
    std::vector<std::string> nodes_json;
    std::vector<std::string> edges_json;
    int next_id = 0;

    // --- Node 0: Keyframe ---
    {
      std::ostringstream oss;
      oss << std::setprecision(8);
      oss << "{\"id\":" << next_id << ",\"type\":\"keyframe\",\"features\":{";

      // CLIP embedding
      oss << "\"clip\":[";
      if (kf->clip_embedding) {
        const auto& emb = kf->clip_embedding.value();
        for (size_t i = 0; i < emb.size(); i++) {
          if (i > 0) oss << ",";
          oss << emb[i];
        }
      }
      oss << "],";

      // Floor level
      oss << "\"floor_level\":" << kf->floor_level << ",";

      // Scan range profile (16 bins) — heading-invariant via circular shift
      auto scan_profile = compute_scan_range_profile(kf->cloud);
      oss << "\"scan_range_profile\":[";
      for (size_t i = 0; i < scan_profile.size(); i++) {
        if (i > 0) oss << ",";
        oss << scan_profile[i];
      }
      oss << "],";

      // Positional encoding
      auto pe = compute_positional_encoding(kf_pos.x(), kf_pos.y());
      oss << "\"pe\":[" << pe[0] << "," << pe[1] << "," << pe[2] << "," << pe[3] << "]";

      oss << "}}";
      nodes_json.push_back(oss.str());
    }
    int kf_node_id = next_id++;

    // --- Plane nodes ---
    // Track plane_id → node_id mapping for plane-plane edges and room edges
    std::unordered_map<int, int> plane_to_node;  // global plane id → subgraph node id

    auto add_plane_nodes = [&](const std::vector<int>& plane_ids,
                               const auto& planes_map,
                               const std::string& plane_type) {
      for (int pid : plane_ids) {
        if (plane_to_node.count(pid)) continue;  // skip duplicates
        auto it = planes_map.find(pid);
        if (it == planes_map.end() || !it->second.plane_node) continue;

        Eigen::Vector4d coeffs = it->second.plane_node->estimate().coeffs();
        double nx = coeffs[0], ny = coeffs[1], nz = coeffs[2], d = coeffs[3];

        // Distance from keyframe to plane
        double dist = std::abs(nx * kf_pos.x() + ny * kf_pos.y() + nz * kf_pos.z() + d);

        // Bearing angle from keyframe to plane (XY projection of normal)
        double bearing = std::atan2(ny, nx);

        int node_id = next_id++;
        plane_to_node[pid] = node_id;

        // Compute KF heading for relative bearing
        Eigen::Vector3d kf_heading = Eigen::Vector3d::UnitX();
        if (kf->node) {
          kf_heading = kf->node->estimate().rotation() * Eigen::Vector3d::UnitX();
        }
        double kf_yaw = std::atan2(kf_heading.y(), kf_heading.x());
        double relative_bearing = std::atan2(ny, nx) - kf_yaw;
        // Normalize to [-pi, pi]
        while (relative_bearing > M_PI) relative_bearing -= 2.0 * M_PI;
        while (relative_bearing < -M_PI) relative_bearing += 2.0 * M_PI;

        // Plane node
        std::ostringstream oss;
        oss << std::setprecision(8);
        oss << "{\"id\":" << node_id << ",\"type\":\"plane\",\"features\":{";
        oss << "\"nx\":" << nx << ",\"ny\":" << ny << ",\"nz\":" << nz;
        oss << ",\"d\":" << d;
        oss << ",\"relative_d\":" << dist;
        oss << ",\"relative_bearing\":" << relative_bearing;
        oss << ",\"plane_type\":\"" << plane_type << "\"";
        oss << ",\"global_plane_id\":" << pid;
        oss << "}}";
        nodes_json.push_back(oss.str());

        // Edge: KF → plane (observes)
        std::ostringstream edge_oss;
        edge_oss << std::setprecision(8);
        edge_oss << "{\"src\":" << kf_node_id << ",\"dst\":" << node_id;
        edge_oss << ",\"type\":\"observes\",\"features\":{";
        edge_oss << "\"distance\":" << dist << ",\"bearing\":" << bearing;
        edge_oss << "}}";
        edges_json.push_back(edge_oss.str());
      }
    };

    add_plane_nodes(kf->x_plane_ids, x_vert_planes, "x_vert");
    add_plane_nodes(kf->y_plane_ids, y_vert_planes, "y_vert");

    // Horizontal planes (different container type)
    for (int pid : kf->hort_plane_ids) {
      if (plane_to_node.count(pid)) continue;  // skip duplicates
      auto it = hort_planes.find(pid);
      if (it == hort_planes.end() || !it->second.plane_node) continue;

      Eigen::Vector4d coeffs = it->second.plane_node->estimate().coeffs();
      double nx = coeffs[0], ny = coeffs[1], nz = coeffs[2], d = coeffs[3];
      double dist = std::abs(nx * kf_pos.x() + ny * kf_pos.y() + nz * kf_pos.z() + d);
      double bearing = std::atan2(ny, nx);

      int node_id = next_id++;
      plane_to_node[pid] = node_id;

      // Compute relative bearing for horizontal planes
      Eigen::Vector3d kf_heading = Eigen::Vector3d::UnitX();
      if (kf->node) {
        kf_heading = kf->node->estimate().rotation() * Eigen::Vector3d::UnitX();
      }
      double kf_yaw = std::atan2(kf_heading.y(), kf_heading.x());
      double relative_bearing = std::atan2(ny, nx) - kf_yaw;
      while (relative_bearing > M_PI) relative_bearing -= 2.0 * M_PI;
      while (relative_bearing < -M_PI) relative_bearing += 2.0 * M_PI;

      std::ostringstream oss;
      oss << std::setprecision(8);
      oss << "{\"id\":" << node_id << ",\"type\":\"plane\",\"features\":{";
      oss << "\"nx\":" << nx << ",\"ny\":" << ny << ",\"nz\":" << nz;
      oss << ",\"d\":" << d;
      oss << ",\"relative_d\":" << dist;
      oss << ",\"relative_bearing\":" << relative_bearing;
      oss << ",\"plane_type\":\"horizontal\"";
      oss << ",\"global_plane_id\":" << pid;
      oss << "}}";
      nodes_json.push_back(oss.str());

      std::ostringstream edge_oss;
      edge_oss << std::setprecision(8);
      edge_oss << "{\"src\":" << kf_node_id << ",\"dst\":" << node_id;
      edge_oss << ",\"type\":\"observes\",\"features\":{";
      edge_oss << "\"distance\":" << dist << ",\"bearing\":" << bearing;
      edge_oss << "}}";
      edges_json.push_back(edge_oss.str());
    }

    // --- Plane ↔ Plane edges (spatial relations) ---
    // For every pair of planes in the subgraph, compute angle between normals
    std::vector<std::pair<int, int>> plane_pairs;  // (global_plane_id, node_id)
    for (const auto& [gpid, nid] : plane_to_node) {
      plane_pairs.push_back({gpid, nid});
    }
    for (size_t i = 0; i < plane_pairs.size(); i++) {
      for (size_t j = i + 1; j < plane_pairs.size(); j++) {
        int gpid_i = plane_pairs[i].first;
        int gpid_j = plane_pairs[j].first;
        int nid_i = plane_pairs[i].second;
        int nid_j = plane_pairs[j].second;

        // Get normals for both planes
        Eigen::Vector3d n_i = get_plane_normal(gpid_i, x_vert_planes, y_vert_planes, hort_planes);
        Eigen::Vector3d n_j = get_plane_normal(gpid_j, x_vert_planes, y_vert_planes, hort_planes);

        double dot = n_i.dot(n_j);
        dot = std::max(-1.0, std::min(1.0, dot));  // clamp for acos safety
        double angle = std::acos(std::abs(dot));  // [0, π/2]: 0=parallel, π/2=perpendicular

        // Distance between planes (for parallel planes: |d1 - d2|)
        double d_i = get_plane_d(gpid_i, x_vert_planes, y_vert_planes, hort_planes);
        double d_j = get_plane_d(gpid_j, x_vert_planes, y_vert_planes, hort_planes);
        double plane_dist = std::abs(d_i - d_j);

        std::ostringstream edge_oss;
        edge_oss << std::setprecision(8);
        edge_oss << "{\"src\":" << nid_i << ",\"dst\":" << nid_j;
        edge_oss << ",\"type\":\"spatial\",\"features\":{";
        edge_oss << "\"angle\":" << angle << ",\"plane_dist\":" << plane_dist;
        edge_oss << "}}";
        edges_json.push_back(edge_oss.str());
      }
    }

    // --- Room node ---
    for (const auto& [rid, room] : rooms_vec) {
      if (room.room_keyframes.count(kf->id()) == 0) continue;

      int room_node_id = next_id++;

      // Room center from node estimate
      double cx = 0, cy = 0, cz = 0;
      int num_walls = 0;
      if (room.node) {
        Eigen::Isometry3d room_pose = room.node->estimate();
        cx = room_pose.translation().x();
        cy = room_pose.translation().y();
        cz = room_pose.translation().z();
      }
      // Count bounding planes
      if (room.plane_x1_id >= 0) num_walls++;
      if (room.plane_x2_id >= 0) num_walls++;
      if (room.plane_y1_id >= 0) num_walls++;
      if (room.plane_y2_id >= 0) num_walls++;

      std::ostringstream oss;
      oss << std::setprecision(8);
      oss << "{\"id\":" << room_node_id << ",\"type\":\"room\",\"features\":{";
      oss << "\"cx\":" << cx << ",\"cy\":" << cy << ",\"cz\":" << cz;
      oss << ",\"num_walls\":" << num_walls;
      oss << ",\"global_room_id\":" << rid;
      oss << "}}";
      nodes_json.push_back(oss.str());

      // Edge: KF → room (inside)
      double dist_to_center = (kf_pos - Eigen::Vector3d(cx, cy, cz)).norm();
      {
        std::ostringstream edge_oss;
        edge_oss << std::setprecision(8);
        edge_oss << "{\"src\":" << kf_node_id << ",\"dst\":" << room_node_id;
        edge_oss << ",\"type\":\"inside\",\"features\":{";
        edge_oss << "\"dist_to_center\":" << dist_to_center;
        edge_oss << "}}";
        edges_json.push_back(edge_oss.str());
      }

      // Edges: room → bounding planes
      auto add_bounded_edge = [&](int plane_id) {
        if (plane_id < 0) return;
        auto pit = plane_to_node.find(plane_id);
        if (pit != plane_to_node.end()) {
          std::ostringstream edge_oss;
          edge_oss << "{\"src\":" << room_node_id << ",\"dst\":" << pit->second;
          edge_oss << ",\"type\":\"bounded_by\",\"features\":{}}";
          edges_json.push_back(edge_oss.str());
        }
      };
      add_bounded_edge(room.plane_x1_id);
      add_bounded_edge(room.plane_x2_id);
      add_bounded_edge(room.plane_y1_id);
      add_bounded_edge(room.plane_y2_id);

      break;  // keyframe can only be in one room
    }

    // --- Object nodes (from YOLO detections) ---
    if (kf->detected_objects) {
      const auto& objects = kf->detected_objects.value();
      for (size_t i = 0; i < objects.size(); i++) {
        const std::string& obj_name = objects[i];
        if (obj_name.empty() || obj_name == " ") continue;  // skip empty class

        int class_idx = -1;
        for (size_t c = 0; c < OBJECT_CLASSES.size(); c++) {
          if (obj_name == OBJECT_CLASSES[c]) {
            class_idx = static_cast<int>(c);
            break;
          }
        }
        if (class_idx < 0) continue;  // unknown class

        // Get confidence if available
        float confidence = 0.0f;
        if (kf->object_confidences) {
          auto cit = kf->object_confidences.value().find(obj_name);
          if (cit != kf->object_confidences.value().end()) {
            confidence = cit->second;
          }
        }

        int obj_node_id = next_id++;
        std::ostringstream oss;
        oss << std::setprecision(8);
        oss << "{\"id\":" << obj_node_id << ",\"type\":\"object\",\"features\":{";
        oss << "\"class\":\"" << obj_name << "\"";
        oss << ",\"class_idx\":" << class_idx;
        oss << ",\"confidence\":" << confidence;
        oss << "}}";
        nodes_json.push_back(oss.str());

        // Edge: KF → object (sees)
        std::ostringstream edge_oss;
        edge_oss << "{\"src\":" << kf_node_id << ",\"dst\":" << obj_node_id;
        edge_oss << ",\"type\":\"sees\",\"features\":{}}";
        edges_json.push_back(edge_oss.str());
      }
    }

    // ── Write JSON file ──
    std::ostringstream filename;
    filename << OUTPUT_DIR << "/kf_" << std::setw(5) << std::setfill('0')
             << kf->id() << ".json";

    std::ofstream ofs(filename.str());
    if (!ofs.is_open()) {
      std::cerr << "[SUBGRAPH_DUMP] Failed to open " << filename.str() << std::endl;
      return;
    }

    ofs << std::setprecision(10);
    ofs << "{\n";
    ofs << "  \"keyframe_id\": " << kf->id() << ",\n";
    ofs << "  \"timestamp_sec\": " << kf->stamp.seconds() << ",\n";
    ofs << "  \"floor_level\": " << kf->floor_level << ",\n";

    // Nodes array
    ofs << "  \"nodes\": [\n";
    for (size_t i = 0; i < nodes_json.size(); i++) {
      ofs << "    " << nodes_json[i];
      if (i < nodes_json.size() - 1) ofs << ",";
      ofs << "\n";
    }
    ofs << "  ],\n";

    // Edges array
    ofs << "  \"edges\": [\n";
    for (size_t i = 0; i < edges_json.size(); i++) {
      ofs << "    " << edges_json[i];
      if (i < edges_json.size() - 1) ofs << ",";
      ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "[SUBGRAPH_DUMP] KF" << kf->id()
              << ": " << nodes_json.size() << " nodes, "
              << edges_json.size() << " edges → " << filename.str()
              << std::endl;
  }

 private:
  // Helper: get plane normal from any of the 3 plane maps
  static Eigen::Vector3d get_plane_normal(
      int pid,
      const std::unordered_map<int, VerticalPlanes>& x_planes,
      const std::unordered_map<int, VerticalPlanes>& y_planes,
      const std::unordered_map<int, HorizontalPlanes>& h_planes) {
    {
      auto it = x_planes.find(pid);
      if (it != x_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs().head<3>();
    }
    {
      auto it = y_planes.find(pid);
      if (it != y_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs().head<3>();
    }
    {
      auto it = h_planes.find(pid);
      if (it != h_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs().head<3>();
    }
    return Eigen::Vector3d::UnitZ();
  }

  // Helper: get plane distance coefficient 'd'
  static double get_plane_d(
      int pid,
      const std::unordered_map<int, VerticalPlanes>& x_planes,
      const std::unordered_map<int, VerticalPlanes>& y_planes,
      const std::unordered_map<int, HorizontalPlanes>& h_planes) {
    {
      auto it = x_planes.find(pid);
      if (it != x_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs()[3];
    }
    {
      auto it = y_planes.find(pid);
      if (it != y_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs()[3];
    }
    {
      auto it = h_planes.find(pid);
      if (it != h_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs()[3];
    }
    return 0.0;
  }

  // Helper: get plane coefficients [nx, ny, nz, d]
  static Eigen::Vector4d get_plane_coeffs(
      int pid,
      const std::unordered_map<int, VerticalPlanes>& x_planes,
      const std::unordered_map<int, VerticalPlanes>& y_planes,
      const std::unordered_map<int, HorizontalPlanes>& h_planes) {
    {
      auto it = x_planes.find(pid);
      if (it != x_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs();
    }
    {
      auto it = y_planes.find(pid);
      if (it != y_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs();
    }
    {
      auto it = h_planes.find(pid);
      if (it != h_planes.end() && it->second.plane_node)
        return it->second.plane_node->estimate().coeffs();
    }
    return Eigen::Vector4d(0, 0, 1, 0);
  }

  /**
   * @brief Compute scan range profile — 16-bin spatial layout fingerprint.
   *
   * Projects point cloud to XY plane, divides into angular sectors,
   * and computes median range per sector. Captures spatial layout
   * (openings, corridors, junctions) as a compact fingerprint.
   */
  static std::vector<float> compute_scan_range_profile(
      const pcl::PointCloud<PointT>::Ptr& cloud, int num_bins = 16) {
    const float MAX_RANGE = 30.0f;  // normalization factor
    std::vector<float> profile(num_bins, 0.0f);

    if (!cloud || cloud->empty()) return profile;

    // Collect ranges per angular bin
    std::vector<std::vector<float>> bin_ranges(num_bins);
    const float bin_width = 2.0f * M_PI / num_bins;

    for (const auto& pt : cloud->points) {
      float range = std::sqrt(pt.x * pt.x + pt.y * pt.y);
      if (range < 0.1f) continue;  // skip points at origin

      float angle = std::atan2(pt.y, pt.x);  // [-pi, pi]
      angle += M_PI;  // shift to [0, 2*pi]
      int bin = static_cast<int>(angle / bin_width);
      bin = std::max(0, std::min(bin, num_bins - 1));

      bin_ranges[bin].push_back(range);
    }

    // Compute median per bin, normalized
    for (int i = 0; i < num_bins; i++) {
      if (bin_ranges[i].empty()) {
        profile[i] = 1.0f;  // no return = open space, max range
        continue;
      }
      std::sort(bin_ranges[i].begin(), bin_ranges[i].end());
      float median = bin_ranges[i][bin_ranges[i].size() / 2];
      profile[i] = std::min(median / MAX_RANGE, 1.0f);
    }

    // Circular-shift so the max-range bin is at index 0.
    // This makes the profile invariant to the robot's heading.
    auto max_it = std::max_element(profile.begin(), profile.end());
    if (max_it != profile.end()) {
      std::rotate(profile.begin(), max_it, profile.end());
    }

    return profile;
  }

  /**
   * @brief 2D sinusoidal positional encoding of (x, y) coordinates.
   *
   * Returns [sin(x/λ), cos(x/λ), sin(y/λ), cos(y/λ)].
   * The wavelength λ = 5 m is chosen so parallel warehouse aisles
   * (typically 3-8 m apart) get maximally different encodings.
   */
  static std::array<float, 4> compute_positional_encoding(
      double x, double y, double wavelength = 5.0) {
    double sx = x / wavelength;
    double sy = y / wavelength;
    return {{
        static_cast<float>(std::sin(sx)),
        static_cast<float>(std::cos(sx)),
        static_cast<float>(std::sin(sy)),
        static_cast<float>(std::cos(sy))
    }};
  }

 public:
  // ── Tensor output for GNN inference ──

  static constexpr int NODE_TYPE_DIM = 4;    // one-hot: keyframe, plane, room, object
  static constexpr int CLIP_DIM = 512;
  static constexpr int SCAN_PROFILE_DIM = 16; // scan range profile bins
  static constexpr int PE_DIM = 4;            // sin(x/λ), cos(x/λ), sin(y/λ), cos(y/λ)
  static constexpr int FEATURE_DIM = 533;     // CLIP(512) + floor(1) + scan_profile(16) + PE(4)
  static constexpr int TOTAL_NODE_DIM = NODE_TYPE_DIM + FEATURE_DIM;  // 537
  static constexpr int NUM_OBJ_CLASSES = 5;

  struct SubgraphTensors {
    std::vector<float> node_features;  // [N * TOTAL_NODE_DIM]
    std::vector<int> edge_src;         // [E] (bidirectional)
    std::vector<int> edge_dst;         // [E]
    int num_nodes = 0;
    int feat_dim = TOTAL_NODE_DIM;
  };

  /**
   * @brief Build node feature + edge index tensors for GNN inference.
   *        Mirrors the Python build_dataset.py encoding exactly.
   */
  static SubgraphTensors build_tensors(
      const KeyFrame::Ptr& kf,
      const std::unordered_map<int, VerticalPlanes>& x_vert_planes,
      const std::unordered_map<int, VerticalPlanes>& y_vert_planes,
      const std::unordered_map<int, HorizontalPlanes>& hort_planes,
      const std::unordered_map<int, Rooms>& rooms_vec) {

    SubgraphTensors out;
    int next_id = 0;
    std::unordered_map<int, int> plane_to_node;

    // Helper to add a node feature vector
    auto add_node = [&](const std::vector<float>& feat) {
      for (float f : feat) out.node_features.push_back(f);
      return next_id++;
    };

    // Helper to add bidirectional edge
    auto add_edge = [&](int src, int dst) {
      out.edge_src.push_back(src);
      out.edge_dst.push_back(dst);
      out.edge_src.push_back(dst);
      out.edge_dst.push_back(src);
    };

    // ── Keyframe node ──
    std::vector<float> kf_feat(TOTAL_NODE_DIM, 0.0f);
    kf_feat[0] = 1.0f;  // type = keyframe
    if (kf->clip_embedding) {
      const auto& emb = kf->clip_embedding.value();
      for (size_t i = 0; i < emb.size() && i < CLIP_DIM; i++) {
        kf_feat[NODE_TYPE_DIM + i] = emb[i];
      }
    }
    kf_feat[NODE_TYPE_DIM + CLIP_DIM] = static_cast<float>(kf->floor_level);
    // Scan range profile (16 bins) — heading-invariant via circular shift
    auto scan_profile = compute_scan_range_profile(kf->cloud);
    for (int i = 0; i < SCAN_PROFILE_DIM && i < (int)scan_profile.size(); i++) {
      kf_feat[NODE_TYPE_DIM + CLIP_DIM + 1 + i] = scan_profile[i];
    }
    // Positional encoding (4 dims) — breaks aliasing between identical structures
    Eigen::Vector3d kf_pos_early = Eigen::Vector3d::Zero();
    if (kf->node) {
      kf_pos_early = kf->node->estimate().translation();
    }
    auto pe = compute_positional_encoding(kf_pos_early.x(), kf_pos_early.y());
    for (int i = 0; i < PE_DIM; i++) {
      kf_feat[NODE_TYPE_DIM + CLIP_DIM + 1 + SCAN_PROFILE_DIM + i] = pe[i];
    }
    int kf_node_id = add_node(kf_feat);

    // Get keyframe position
    Eigen::Vector3d kf_pos = Eigen::Vector3d::Zero();
    if (kf->node) {
      kf_pos = kf->node->estimate().translation();
    }

    // Compute KF heading for relative bearing
    Eigen::Vector3d kf_heading = Eigen::Vector3d::UnitX();
    if (kf->node) {
      kf_heading = kf->node->estimate().rotation() * Eigen::Vector3d::UnitX();
    }
    double kf_yaw = std::atan2(kf_heading.y(), kf_heading.x());

    // ── Plane nodes ──
    // First pass: collect plane IDs and their coefficients for aggregate features
    std::vector<int> all_plane_ids;
    for (int pid : kf->x_plane_ids) all_plane_ids.push_back(pid);
    for (int pid : kf->y_plane_ids) all_plane_ids.push_back(pid);
    for (int pid : kf->hort_plane_ids) all_plane_ids.push_back(pid);

    // Precompute per-plane aggregate features (mean pairwise angle, min gap)
    std::unordered_map<int, float> plane_mean_angle;  // pid → mean angle to other planes
    std::unordered_map<int, float> plane_min_gap;     // pid → min gap to parallel planes

    for (int pid_i : all_plane_ids) {
      Eigen::Vector3d n_i = get_plane_normal(pid_i, x_vert_planes, y_vert_planes, hort_planes);
      double d_i = get_plane_d(pid_i, x_vert_planes, y_vert_planes, hort_planes);

      float angle_sum = 0.0f;
      int angle_count = 0;
      float min_gap = 100.0f;

      for (int pid_j : all_plane_ids) {
        if (pid_i == pid_j) continue;
        Eigen::Vector3d n_j = get_plane_normal(pid_j, x_vert_planes, y_vert_planes, hort_planes);
        double d_j = get_plane_d(pid_j, x_vert_planes, y_vert_planes, hort_planes);

        double dot = n_i.dot(n_j);
        dot = std::max(-1.0, std::min(1.0, dot));
        float angle = static_cast<float>(std::acos(std::abs(dot)));
        angle_sum += angle;
        angle_count++;

        // Gap distance for near-parallel planes (angle < 20°)
        if (angle < 0.35f) {  // ~20 degrees
          float gap = static_cast<float>(std::abs(d_i - d_j));
          min_gap = std::min(min_gap, gap);
        }
      }

      plane_mean_angle[pid_i] = angle_count > 0 ? angle_sum / angle_count : 0.0f;
      plane_min_gap[pid_i] = min_gap < 99.0f ? min_gap : 0.0f;
    }

    auto add_plane_tensor = [&](const std::vector<int>& plane_ids,
                                const auto& planes_map,
                                int plane_type_idx) {
      for (int pid : plane_ids) {
        if (plane_to_node.count(pid)) continue;
        auto it = planes_map.find(pid);
        if (it == planes_map.end() || !it->second.plane_node) continue;

        Eigen::Vector4d coeffs = it->second.plane_node->estimate().coeffs();
        double nx = coeffs[0], ny = coeffs[1], nz = coeffs[2], d_val = coeffs[3];

        // Pose-invariant features
        double relative_d = std::abs(nx * kf_pos.x() + ny * kf_pos.y() + nz * kf_pos.z() + d_val);
        double relative_bearing = std::atan2(ny, nx) - kf_yaw;
        while (relative_bearing > M_PI) relative_bearing -= 2.0 * M_PI;
        while (relative_bearing < -M_PI) relative_bearing += 2.0 * M_PI;

        std::vector<float> feat(TOTAL_NODE_DIM, 0.0f);
        feat[1] = 1.0f;  // type = plane
        feat[NODE_TYPE_DIM + 0] = static_cast<float>(nx);  // nx
        feat[NODE_TYPE_DIM + 1] = static_cast<float>(ny);  // ny
        feat[NODE_TYPE_DIM + 2] = static_cast<float>(nz);  // nz
        feat[NODE_TYPE_DIM + 3] = static_cast<float>(relative_d) / 10.0f;  // relative_d (pose-invariant)
        feat[NODE_TYPE_DIM + 4] = static_cast<float>(relative_bearing) / M_PI;  // normalized [-1, 1]
        feat[NODE_TYPE_DIM + 5 + plane_type_idx] = 1.0f;  // plane type one-hot (3 slots)
        // Per-plane aggregate inter-plane features
        feat[NODE_TYPE_DIM + 8] = plane_mean_angle[pid] / (M_PI / 2.0f);  // normalized [0, 1]
        feat[NODE_TYPE_DIM + 9] = plane_min_gap[pid] / 10.0f;  // corridor width, normalized

        int nid = add_node(feat);
        plane_to_node[pid] = nid;

        // KF → plane edge
        add_edge(kf_node_id, nid);
      }
    };

    add_plane_tensor(kf->x_plane_ids, x_vert_planes, 0);  // x_vert
    add_plane_tensor(kf->y_plane_ids, y_vert_planes, 1);   // y_vert

    // Horizontal planes — use same pose-invariant encoding
    for (int pid : kf->hort_plane_ids) {
      if (plane_to_node.count(pid)) continue;
      auto it = hort_planes.find(pid);
      if (it == hort_planes.end() || !it->second.plane_node) continue;

      Eigen::Vector4d coeffs = it->second.plane_node->estimate().coeffs();
      double nx = coeffs[0], ny = coeffs[1], nz = coeffs[2], d_val = coeffs[3];

      double relative_d = std::abs(nx * kf_pos.x() + ny * kf_pos.y() + nz * kf_pos.z() + d_val);
      double relative_bearing = std::atan2(ny, nx) - kf_yaw;
      while (relative_bearing > M_PI) relative_bearing -= 2.0 * M_PI;
      while (relative_bearing < -M_PI) relative_bearing += 2.0 * M_PI;

      std::vector<float> feat(TOTAL_NODE_DIM, 0.0f);
      feat[1] = 1.0f;
      feat[NODE_TYPE_DIM + 0] = static_cast<float>(nx);
      feat[NODE_TYPE_DIM + 1] = static_cast<float>(ny);
      feat[NODE_TYPE_DIM + 2] = static_cast<float>(nz);
      feat[NODE_TYPE_DIM + 3] = static_cast<float>(relative_d) / 10.0f;
      feat[NODE_TYPE_DIM + 4] = static_cast<float>(relative_bearing) / M_PI;
      feat[NODE_TYPE_DIM + 7] = 1.0f;  // horizontal type (index 5+2=7)
      feat[NODE_TYPE_DIM + 8] = plane_mean_angle[pid] / (M_PI / 2.0f);
      feat[NODE_TYPE_DIM + 9] = plane_min_gap[pid] / 10.0f;

      int nid = add_node(feat);
      plane_to_node[pid] = nid;
      add_edge(kf_node_id, nid);
    }

    // ── Plane ↔ Plane spatial edges ──
    std::vector<std::pair<int, int>> plane_pairs;
    for (const auto& [gpid, nid] : plane_to_node) {
      plane_pairs.push_back({gpid, nid});
    }
    for (size_t i = 0; i < plane_pairs.size(); i++) {
      for (size_t j = i + 1; j < plane_pairs.size(); j++) {
        add_edge(plane_pairs[i].second, plane_pairs[j].second);
      }
    }

    // ── Room node ──
    for (const auto& [rid, room] : rooms_vec) {
      if (room.room_keyframes.count(kf->id()) == 0) continue;

      double cx = 0, cy = 0, cz = 0;
      int num_walls = 0;
      if (room.node) {
        Eigen::Isometry3d room_pose = room.node->estimate();
        cx = room_pose.translation().x();
        cy = room_pose.translation().y();
        cz = room_pose.translation().z();
      }
      if (room.plane_x1_id >= 0) num_walls++;
      if (room.plane_x2_id >= 0) num_walls++;
      if (room.plane_y1_id >= 0) num_walls++;
      if (room.plane_y2_id >= 0) num_walls++;

      std::vector<float> feat(TOTAL_NODE_DIM, 0.0f);
      feat[2] = 1.0f;  // type = room
      feat[NODE_TYPE_DIM + 0] = static_cast<float>(cx) / 10.0f;
      feat[NODE_TYPE_DIM + 1] = static_cast<float>(cy) / 10.0f;
      feat[NODE_TYPE_DIM + 2] = static_cast<float>(cz) / 10.0f;
      feat[NODE_TYPE_DIM + 3] = static_cast<float>(num_walls) / 4.0f;
      // Positional encoding for room center
      auto room_pe = compute_positional_encoding(cx, cy);
      for (int i = 0; i < PE_DIM; i++) {
        feat[NODE_TYPE_DIM + 4 + i] = room_pe[i];
      }

      int room_nid = add_node(feat);
      add_edge(kf_node_id, room_nid);

      // Room → bounding plane edges
      auto add_room_plane = [&](int plane_id) {
        if (plane_id < 0) return;
        auto pit = plane_to_node.find(plane_id);
        if (pit != plane_to_node.end()) {
          add_edge(room_nid, pit->second);
        }
      };
      add_room_plane(room.plane_x1_id);
      add_room_plane(room.plane_x2_id);
      add_room_plane(room.plane_y1_id);
      add_room_plane(room.plane_y2_id);

      break;
    }

    // ── Object nodes ──
    if (kf->detected_objects) {
      const auto& objects = kf->detected_objects.value();
      for (size_t i = 0; i < objects.size(); i++) {
        const std::string& obj_name = objects[i];
        if (obj_name.empty() || obj_name == " ") continue;

        int class_idx = -1;
        for (size_t c = 0; c < OBJECT_CLASSES.size(); c++) {
          if (obj_name == OBJECT_CLASSES[c]) {
            class_idx = static_cast<int>(c);
            break;
          }
        }
        if (class_idx < 0) continue;

        float confidence = 0.0f;
        if (kf->object_confidences) {
          auto cit = kf->object_confidences.value().find(obj_name);
          if (cit != kf->object_confidences.value().end()) {
            confidence = cit->second;
          }
        }

        std::vector<float> feat(TOTAL_NODE_DIM, 0.0f);
        feat[3] = 1.0f;  // type = object
        feat[NODE_TYPE_DIM + class_idx] = 1.0f;
        feat[NODE_TYPE_DIM + NUM_OBJ_CLASSES] = confidence;

        int obj_nid = add_node(feat);
        add_edge(kf_node_id, obj_nid);
      }
    }

    out.num_nodes = next_id;
    return out;
  }
};

}  // namespace s_graphs

#endif  // SUBGRAPH_DUMPER_HPP
