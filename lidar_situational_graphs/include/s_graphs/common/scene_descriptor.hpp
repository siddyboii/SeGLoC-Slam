#ifndef SCENE_DESCRIPTOR_HPP
#define SCENE_DESCRIPTOR_HPP

#include <s_graphs/common/keyframe.hpp>
#include <s_graphs/common/planes.hpp>
#include <s_graphs/common/rooms.hpp>

#include <Eigen/Eigen>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace s_graphs {

/**
 * @brief Per-keyframe scene descriptor combining structural + semantic features.
 *
 * Encodes the local environment around a keyframe using:
 *   - Structural features from S-Graphs (observed planes, room membership)
 *   - Semantic features (CLIP embedding, YOLO object histogram)
 *
 * Used for loop closure verification: compare two descriptors to decide if
 * two keyframes observe the same physical place.
 */
struct SceneDescriptor {
  // ── Feature dimensions ──
  static constexpr int STRUCTURAL_DIM = 10;
  static constexpr int OBJECT_CLASSES = 5;  // table, tree, desks, monitor, pillar
  static constexpr int CLIP_DIM = 512;
  static constexpr int TOTAL_DIM = STRUCTURAL_DIM + OBJECT_CLASSES + CLIP_DIM;

  std::vector<float> descriptor;  // fixed-size feature vector

  // ── Known YOLO class names (order determines histogram index) ──
  static inline const std::vector<std::string> KNOWN_CLASSES = {
      "Table", "Tree", "Desks", "Computer Monitor", "Pillar"};

  /**
   * @brief Extract a scene descriptor for a keyframe using surrounding S-Graphs
   * data.
   *
   * @param keyframe         The target keyframe
   * @param x_vert_planes    X-aligned vertical planes from S-Graphs
   * @param y_vert_planes    Y-aligned vertical planes from S-Graphs
   * @param hort_planes      Horizontal planes from S-Graphs
   * @param rooms_vec        Rooms from S-Graphs
   * @return SceneDescriptor with populated descriptor vector
   */
  static SceneDescriptor extract(
      const KeyFrame::Ptr& keyframe,
      const std::unordered_map<int, VerticalPlanes>& x_vert_planes,
      const std::unordered_map<int, VerticalPlanes>& y_vert_planes,
      const std::unordered_map<int, HorizontalPlanes>& hort_planes,
      const std::unordered_map<int, Rooms>& rooms_vec) {
    SceneDescriptor desc;
    desc.descriptor.resize(TOTAL_DIM, 0.0f);

    int idx = 0;

    // ── 1. Structural features from planes ──

    // Count of observed planes by type
    int n_x = static_cast<int>(keyframe->x_plane_ids.size());
    int n_y = static_cast<int>(keyframe->y_plane_ids.size());
    int n_h = static_cast<int>(keyframe->hort_plane_ids.size());
    desc.descriptor[idx++] = static_cast<float>(n_x) / 5.0f;      // normalized
    desc.descriptor[idx++] = static_cast<float>(n_y) / 5.0f;
    desc.descriptor[idx++] = static_cast<float>(n_h) / 3.0f;
    desc.descriptor[idx++] = static_cast<float>(n_x + n_y + n_h) / 10.0f;

    // Average plane distance from keyframe position
    double total_dist = 0.0;
    int plane_count = 0;
    Eigen::Vector3d kf_pos = Eigen::Vector3d::Zero();
    if (keyframe->node) {
      kf_pos = keyframe->node->estimate().translation();
    }

    // Plane normal angle histogram (4 bins: 0-90, 90-180, 180-270, 270-360)
    std::vector<float> normal_hist(4, 0.0f);

    auto process_planes = [&](const std::vector<int>& plane_ids,
                              const auto& planes_map) {
      for (int pid : plane_ids) {
        auto it = planes_map.find(pid);
        if (it == planes_map.end() || !it->second.plane_node) continue;

        Eigen::Vector4d coeffs = it->second.plane_node->estimate().coeffs();
        Eigen::Vector3d normal = coeffs.head<3>();
        double d = coeffs[3];

        // Distance from keyframe to plane: |n·p + d|
        double dist = std::abs(normal.dot(kf_pos) + d);
        total_dist += dist;
        plane_count++;

        // Normal angle → histogram bin
        double angle = std::atan2(normal.y(), normal.x());  // [-π, π]
        if (angle < 0) angle += 2.0 * M_PI;                 // [0, 2π]
        int bin = static_cast<int>(angle / (M_PI / 2.0));
        bin = std::min(bin, 3);
        normal_hist[bin] += 1.0f;
      }
    };

    process_planes(keyframe->x_plane_ids, x_vert_planes);
    process_planes(keyframe->y_plane_ids, y_vert_planes);

    // Process horizontal planes (separate container type)
    for (int pid : keyframe->hort_plane_ids) {
      auto it = hort_planes.find(pid);
      if (it == hort_planes.end() || !it->second.plane_node) continue;
      Eigen::Vector4d coeffs = it->second.plane_node->estimate().coeffs();
      double dist = std::abs(coeffs.head<3>().dot(kf_pos) + coeffs[3]);
      total_dist += dist;
      plane_count++;
    }

    // Average distance to observed planes (normalized by 10m)
    desc.descriptor[idx++] =
        (plane_count > 0) ? static_cast<float>(total_dist / plane_count) / 10.0f
                          : 0.0f;

    // Normal histogram (normalize to sum=1)
    float hist_sum = std::accumulate(normal_hist.begin(), normal_hist.end(), 0.0f);
    for (int i = 0; i < 4; i++) {
      desc.descriptor[idx++] = (hist_sum > 0) ? normal_hist[i] / hist_sum : 0.25f;
    }

    // ── 2. Room membership ──
    int room_id = -1;
    float room_area = 0.0f;
    for (const auto& [rid, room] : rooms_vec) {
      if (room.room_keyframes.count(keyframe->id())) {
        room_id = rid;
        // Estimate room area from bounding planes (rough)
        if (room.node) {
          room_area = 1.0f;  // room exists → normalized flag
        }
        break;
      }
    }
    desc.descriptor[idx++] = (room_id >= 0) ? 1.0f : 0.0f;  // in_room flag

    // ── 3. Object histogram from YOLO ──
    if (keyframe->detected_objects) {
      for (const auto& obj_name : keyframe->detected_objects.value()) {
        for (size_t c = 0; c < KNOWN_CLASSES.size(); c++) {
          if (obj_name == KNOWN_CLASSES[c]) {
            desc.descriptor[STRUCTURAL_DIM + c] += 1.0f;
          }
        }
      }
      // Normalize object histogram
      float obj_sum = 0.0f;
      for (int c = 0; c < OBJECT_CLASSES; c++) {
        obj_sum += desc.descriptor[STRUCTURAL_DIM + c];
      }
      if (obj_sum > 0.0f) {
        for (int c = 0; c < OBJECT_CLASSES; c++) {
          desc.descriptor[STRUCTURAL_DIM + c] /= obj_sum;
        }
      }
    }

    // ── 4. CLIP embedding ──
    if (keyframe->clip_embedding) {
      const auto& emb = keyframe->clip_embedding.value();
      for (size_t i = 0; i < emb.size() && i < CLIP_DIM; i++) {
        desc.descriptor[STRUCTURAL_DIM + OBJECT_CLASSES + i] = emb[i];
      }
    }

    return desc;
  }

  /**
   * @brief Compute weighted similarity between two scene descriptors.
   *
   * Uses separate comparison for structural vs CLIP components,
   * combined with configurable weights. This allows ablation studies
   * (structural-only, CLIP-only, combined).
   *
   * @return Similarity in [0, 1], higher = more similar
   */
  static double similarity(const std::vector<float>& a,
                            const std::vector<float>& b,
                            double structural_weight = 0.5,
                            double clip_weight = 0.5) {
    if (a.size() != b.size() || a.empty()) return 0.0;

    // ── Structural similarity (first STRUCTURAL_DIM + OBJECT_CLASSES dims) ──
    int struct_end = STRUCTURAL_DIM + OBJECT_CLASSES;
    double struct_dot = 0.0, struct_norm_a = 0.0, struct_norm_b = 0.0;
    for (int i = 0; i < struct_end; i++) {
      struct_dot += a[i] * b[i];
      struct_norm_a += a[i] * a[i];
      struct_norm_b += b[i] * b[i];
    }
    double struct_sim = 0.0;
    if (struct_norm_a > 1e-8 && struct_norm_b > 1e-8) {
      struct_sim = struct_dot / (std::sqrt(struct_norm_a) * std::sqrt(struct_norm_b));
    }

    // ── CLIP similarity (remaining CLIP_DIM dims) ──
    double clip_dot = 0.0, clip_norm_a = 0.0, clip_norm_b = 0.0;
    for (int i = struct_end; i < static_cast<int>(a.size()); i++) {
      clip_dot += a[i] * b[i];
      clip_norm_a += a[i] * a[i];
      clip_norm_b += b[i] * b[i];
    }
    double clip_sim = 0.0;
    if (clip_norm_a > 1e-8 && clip_norm_b > 1e-8) {
      clip_sim = clip_dot / (std::sqrt(clip_norm_a) * std::sqrt(clip_norm_b));
    }

    // ── Weighted combination ──
    // Normalize weights if only one component is available
    bool has_struct = (struct_norm_a > 1e-8 && struct_norm_b > 1e-8);
    bool has_clip = (clip_norm_a > 1e-8 && clip_norm_b > 1e-8);

    if (has_struct && has_clip) {
      return structural_weight * struct_sim + clip_weight * clip_sim;
    } else if (has_struct) {
      return struct_sim;
    } else if (has_clip) {
      return clip_sim;
    }
    return 0.0;
  }
};

}  // namespace s_graphs

#endif  // SCENE_DESCRIPTOR_HPP
