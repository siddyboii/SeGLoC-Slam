#include <s_graphs/backend/loop_mapper.hpp>
#include <s_graphs/common/scene_descriptor.hpp>
#include <g2o/edge_semantic_consistency.hpp>
#include <fstream>
#include <iomanip>

namespace s_graphs {

LoopMapper::LoopMapper(const rclcpp::Node::SharedPtr node, std::mutex& graph_mutex)
    : shared_graph_mutex(graph_mutex), node_(node) {
  inf_calclator.reset(new InformationMatrixCalculator(node));
  fitness_score_thresh = node->get_parameter("fitness_score_thresh").get_parameter_value().get<double>();
  use_zone_prefilter_ = node->get_parameter("use_zone_prefilter").get_parameter_value().get<bool>();
  zone_confidence_threshold_ = node->get_parameter("zone_confidence_threshold").get_parameter_value().get<double>();

  // ── CSV metrics logger ──
  std::string metrics_dir = "/tmp/sgraphs_metrics";
  // Create directory if it doesn't exist
  std::string mkdir_cmd = "mkdir -p " + metrics_dir;
  std::system(mkdir_cmd.c_str());
  loop_metrics_csv_.open(metrics_dir + "/loop_metrics.csv", std::ios::out | std::ios::trunc);
  if (loop_metrics_csv_.is_open()) {
    loop_metrics_csv_ << "kf1_id,kf2_id,has_clip,clip_sim,semantic_alpha,"
                      << "icp_fitness,object_overlap,quality_gate,"
                      << "dyn_factor,zone_factor,info_matrix_norm,"
                      << "edge_added,zone1,zone2,same_zone" << std::endl;
    RCLCPP_INFO(node_->get_logger(), "[LoopMapper] CSV logger opened: %s/loop_metrics.csv",
                metrics_dir.c_str());
  }
}

LoopMapper::~LoopMapper() {
  if (loop_metrics_csv_.is_open()) {
    loop_metrics_csv_.close();
    RCLCPP_INFO(node_->get_logger(), "[LoopMapper] CSV logger closed");
  }
}

void LoopMapper::set_zone_cache(std::shared_ptr<ZoneCache> zone_cache) {
  zone_cache_ = zone_cache;
  RCLCPP_INFO(node_->get_logger(), "[LoopMapper] Zone cache set (prefilter=%s, threshold=%.2f)",
              use_zone_prefilter_ ? "ON" : "OFF", zone_confidence_threshold_);
}

int LoopMapper::validate_loop_closures(
    const std::shared_ptr<GraphSLAM>& covisibility_graph,
    double chi2_threshold) {
  auto* graph = dynamic_cast<g2o::SparseOptimizer*>(covisibility_graph->graph.get());
  
  std::vector<g2o::EdgeLoopClosure*> bad_edges;
  for (auto& edge : graph->edges()) {
    auto* loop_edge = dynamic_cast<g2o::EdgeLoopClosure*>(edge);
    if (!loop_edge) continue;
    
    loop_edge->computeError();
    double chi2 = loop_edge->chi2();
    
    if (chi2 > chi2_threshold) {
      bad_edges.push_back(loop_edge);
      // Log the vertices
      auto* v1 = dynamic_cast<g2o::VertexSE3*>(loop_edge->vertices()[0]);
      auto* v2 = dynamic_cast<g2o::VertexSE3*>(loop_edge->vertices()[1]);
      RCLCPP_WARN(node_->get_logger(),
          "\033[31m[CHI2_REJECT] Removing loop edge KF%d↔KF%d (chi²=%.2f > %.2f)\033[0m",
          v1->id(), v2->id(), chi2, chi2_threshold);
    }
  }
  
  for (auto* edge : bad_edges) {
    graph->removeEdge(edge);
  }
  return bad_edges.size();
}

void LoopMapper::add_loops(const std::shared_ptr<GraphSLAM>& covisibility_graph,
                           const std::vector<Loop::Ptr>& loops) {
  // RCLCPP_WARN(node_->get_logger(), "╔════════════════════════════════════════════════════════════╗");
  // RCLCPP_WARN(node_->get_logger(), "║         LOOP CLOSURE SEMANTIC WEIGHTING PIPELINE            ║");
  // RCLCPP_WARN(node_->get_logger(), "╚════════════════════════════════════════════════════════════╝");
  // RCLCPP_WARN(node_->get_logger(), "[SEMANTIC LOOP MAPPER] Processing %zu loop closures", loops.size());
  
  int semantic_weighted_loops = 0;
  int geometry_only_loops = 0;
  double avg_clip_similarity = 0.0;
  double avg_alpha = 0.0;
  
  for (const auto& loop : loops) {
    // RCLCPP_WARN(node_->get_logger(), "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // ── Per-loop CSV tracking variables ──
    double csv_clip_sim = -1.0;        // -1 means unavailable
    double csv_object_overlap = -1.0;
    double csv_quality_gate = -1.0;
    double csv_dyn_factor = -1.0;
    double csv_zone_factor = -1.0;
    int csv_zone1 = -1, csv_zone2 = -1;
    bool csv_same_zone = false;
    bool csv_has_clip = false;
    bool csv_edge_added = false;
    RCLCPP_WARN(node_->get_logger(), "[LOOP] Between Keyframe %d ↔ Keyframe %d", 
                loop->key1->node->id(), loop->key2->node->id());
    
    Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
    
    // Get the ICP fitness score from the loop
    double icp_fitness = loop->fitness_score;
    
    // Calculate geometry-only information matrix
    Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix(
        loop->key1->cloud, loop->key2->cloud, relpose);
    
    // RCLCPP_WARN(node_->get_logger(), "[GEOMETRY] ICP fitness score: %.6f", icp_fitness);
    // // RCLCPP_WARN(node_->get_logger(), "[GEOMETRY] Information matrix condition number: %.4f",
    //             information_matrix.norm());

    double semantic_alpha = 1.0;
    bool BASELINE_MODE = false;
    std::string clip_status = "❌ UNAVAILABLE";
    bool semantic_rejected = false;

    if (!BASELINE_MODE) {

    // ── Scene Descriptor Verification ──
    // Compare structural+semantic scene descriptors for loop closure verification
    if (loop->key1->scene_descriptor && loop->key2->scene_descriptor) {
      // Compute weighted similarity (structural vs CLIP components)
      // Weights: 0.5 structural + 0.5 CLIP (default — tune for ablation)
      double scene_sim = SceneDescriptor::similarity(
          loop->key1->scene_descriptor.value(),
          loop->key2->scene_descriptor.value(),
          0.5, 0.5);  // structural_weight, clip_weight

      // Also compute component-wise similarities for logging
      double struct_only_sim = SceneDescriptor::similarity(
          loop->key1->scene_descriptor.value(),
          loop->key2->scene_descriptor.value(),
          1.0, 0.0);  // structural only
      double clip_only_sim = SceneDescriptor::similarity(
          loop->key1->scene_descriptor.value(),
          loop->key2->scene_descriptor.value(),
          0.0, 1.0);  // CLIP only

      RCLCPP_WARN(node_->get_logger(),
          "[SCENE_DESC] KF%d↔KF%d  similarity=%.4f  "
          "(structural=%.4f, clip=%.4f)",
          loop->key1->node->id(), loop->key2->node->id(),
          scene_sim, struct_only_sim, clip_only_sim);

      // ── Decision thresholds ──
      // Indoor CLIP scores cluster in [0.85, 0.96] — thresholds must be tight
      if (scene_sim < 0.83) {
        // HARD REJECT: below the typical indoor range → different place
        RCLCPP_WARN(node_->get_logger(),
            "[SCENE_DESC] ✗ REJECTED — similarity %.4f < 0.83 threshold", scene_sim);
        semantic_rejected = true;
        clip_status = "✗ REJECTED by scene descriptor";
      } else if (scene_sim < 0.88) {
        // SOFT PENALTY: borderline — reduce information matrix
        semantic_alpha = 0.5;
        RCLCPP_WARN(node_->get_logger(),
            "[SCENE_DESC] ⚠ LOW confidence (%.4f) — α = 0.5", scene_sim);
        clip_status = "⚠ LOW confidence scene match";
        semantic_weighted_loops++;
      } else if (scene_sim > 0.93) {
        // BOOST: very high match — strong same-place confidence
        semantic_alpha = 1.5;
        RCLCPP_WARN(node_->get_logger(),
            "[SCENE_DESC] ✓ STRONG match (%.4f) — α = 1.5", scene_sim);
        clip_status = "✓ STRONG scene match";
        semantic_weighted_loops++;
      } else {
        // NEUTRAL: 0.93-0.96 — moderate match, keep α = 1.0
        RCLCPP_WARN(node_->get_logger(),
            "[SCENE_DESC] → NEUTRAL match (%.4f) — α = 1.0", scene_sim);
        clip_status = "→ NEUTRAL scene match";
        semantic_weighted_loops++;
      }

      avg_alpha += semantic_alpha;

    } else {
      geometry_only_loops++;
      RCLCPP_WARN(node_->get_logger(),
          "[SCENE_DESC] ❌ One or both keyframes missing scene descriptors — geometry only");
    }

    }  // end if (!BASELINE_MODE)

    // Skip this loop if semantically rejected
    if (semantic_rejected) {
      RCLCPP_WARN(node_->get_logger(),
          "[LOOP] ✗ Skipping loop KF%d↔KF%d — rejected by scene descriptor",
          loop->key1->node->id(), loop->key2->node->id());
      // Write rejection to CSV
      if (loop_metrics_csv_.is_open()) {
        loop_metrics_csv_
            << loop->key1->node->id() << ","
            << loop->key2->node->id() << ","
            << 0 << ","  // has_clip
            << "N/A" << ","  // clip_sim
            << semantic_alpha << ","
            << icp_fitness << ","
            << "N/A" << ","  // object_overlap
            << "N/A" << ","  // quality_gate
            << "N/A" << ","  // dyn_factor
            << "N/A" << ","  // zone_factor
            << 0.0 << ","  // info_matrix_norm
            << 0 << ","  // edge_added = false
            << "REJECTED_BY_SCENE_DESC\n";
      }
      continue;  // skip to next loop
    }

    // Apply semantic weighting
    information_matrix *= semantic_alpha;
    
    RCLCPP_WARN(node_->get_logger(), "[SCALED MATRIX] Condition number: %.4f",
                information_matrix.norm());

    shared_graph_mutex.lock();

    // ── Relative pose sanity diagnostics ──
    Eigen::Vector3d t = relpose.translation();
    Eigen::AngleAxisd aa(relpose.rotation());
    double trans_dist = t.norm();
    double rot_deg = aa.angle() * 180.0 / M_PI;
    RCLCPP_WARN(node_->get_logger(),
                "[RELPOSE] KF%d↔KF%d  translation=(%.3f, %.3f, %.3f) norm=%.3fm  "
                "rotation=%.1f°  fitness=%.4f",
                loop->key1->node->id(), loop->key2->node->id(),
                t.x(), t.y(), t.z(), trans_dist, rot_deg, loop->fitness_score);
    // Warn if the relative pose looks suspicious
    if (trans_dist > 5.0) {
      RCLCPP_ERROR(node_->get_logger(),
                   "[RELPOSE] ⚠ LARGE TRANSLATION (%.3fm > 5m) — likely bad ICP result!", trans_dist);
    }
    if (rot_deg > 45.0 && rot_deg < 135.0) {
      RCLCPP_WARN(node_->get_logger(),
                  "[RELPOSE] ⚠ DIAGONAL ROTATION (%.1f°) — check if 180° fallback applied correctly", rot_deg);
    }

    std::cout << "loop found between keyframes " << loop->key1->node->id() << " and "
              << loop->key2->node->id() << std::endl;

    set_data(loop->key1->node);
    set_data(loop->key2->node);

    bool kf1_stair = get_floor_data(loop->key1->node);
    bool kf2_stair = get_floor_data(loop->key2->node);

    if (!kf1_stair && !kf2_stair) {
      g2o::EdgeLoopClosure* edge = covisibility_graph->add_loop_closure_edge(
          loop->key1->node, loop->key2->node, relpose, information_matrix);
      covisibility_graph->add_robust_kernel(edge, "Huber", 1.0);
      csv_edge_added = true;
      
      RCLCPP_WARN(node_->get_logger(), 
                  "[EDGE ADDED] ✓ Loop closure edge added with semantic weighting [%s]",
                  clip_status.c_str());

      // ── Semantic Consistency Factor ──
      // Add a soft distance constraint from GNN embedding similarity
      bool enable_sem_factor = node_->get_parameter("enable_semantic_factor")
                                   .get_parameter_value().get<bool>();
      if (enable_sem_factor &&
          loop->key1->gnn_embedding && loop->key2->gnn_embedding) {
        const auto& emb1 = loop->key1->gnn_embedding.value();
        const auto& emb2 = loop->key2->gnn_embedding.value();
        double gnn_sim = 0.0;
        for (size_t i = 0; i < emb1.size() && i < emb2.size(); i++) {
          gnn_sim += emb1[i] * emb2[i];
        }

        double sim_thresh = node_->get_parameter("semantic_factor_sim_thresh")
                                .get_parameter_value().get<double>();
        if (gnn_sim > sim_thresh) {
          double info_weight = node_->get_parameter("semantic_factor_info_weight")
                                   .get_parameter_value().get<double>();
          double max_dist = node_->get_parameter("semantic_factor_max_dist")
                                .get_parameter_value().get<double>();
          double gamma = node_->get_parameter("semantic_factor_gamma")
                             .get_parameter_value().get<double>();

          auto sem_edge = covisibility_graph->add_semantic_consistency_edge(
              loop->key1->node, loop->key2->node,
              gnn_sim, info_weight, max_dist, gamma);
          covisibility_graph->add_robust_kernel(sem_edge, "Huber", 2.0);

          double expected_dist = g2o::EdgeSemanticConsistency::similarityToDistance(
              gnn_sim, max_dist, gamma);
          std::cout << "\033[35m[SEMANTIC_FACTOR]\033[0m KF"
                    << loop->key1->node->id() << " ↔ KF"
                    << loop->key2->node->id()
                    << " (sim=" << std::fixed << std::setprecision(3) << gnn_sim
                    << ", expected_dist=" << expected_dist << "m"
                    << ", info=" << std::setprecision(4)
                    << g2o::EdgeSemanticConsistency::similarityToInformation(gnn_sim, info_weight)
                    << ")" << std::endl;
        }
      }
    } else {
      csv_edge_added = false;
      RCLCPP_WARN(node_->get_logger(), 
                  "[EDGE SKIPPED] ✗ Not adding edge - node on floor/stairs");
    }
    shared_graph_mutex.unlock();

    // ── Write per-loop CSV row ──
    if (loop_metrics_csv_.is_open()) {
      loop_metrics_csv_
          << loop->key1->node->id() << ","
          << loop->key2->node->id() << ","
          << (csv_has_clip ? 1 : 0) << ","
          << (csv_clip_sim >= 0 ? std::to_string(csv_clip_sim) : "N/A") << ","
          << semantic_alpha << ","
          << icp_fitness << ","
          << (csv_object_overlap >= 0 ? std::to_string(csv_object_overlap) : "N/A") << ","
          << (csv_quality_gate >= 0 ? std::to_string(csv_quality_gate) : "N/A") << ","
          << (csv_dyn_factor >= 0 ? std::to_string(csv_dyn_factor) : "N/A") << ","
          << (csv_zone_factor >= 0 ? std::to_string(csv_zone_factor) : "N/A") << ","
          << information_matrix.norm() << ","
          << (csv_edge_added ? 1 : 0) << ","
          << csv_zone1 << ","
          << csv_zone2 << ","
          << (csv_same_zone ? 1 : 0)
          << std::endl;
    }
  }
  
  // Summary statistics
  // RCLCPP_WARN(node_->get_logger(), "╔════════════════════════════════════════════════════════════╗");
  // RCLCPP_WARN(node_->get_logger(), "║               SEMANTIC WEIGHTING SUMMARY                    ║");
  // RCLCPP_WARN(node_->get_logger(), "╚════════════════════════════════════════════════════════════╝");
  // RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Total loops: %zu", loops.size());
  // RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Semantically weighted: %d (%.1f%%)", 
  //             semantic_weighted_loops, 
  //             loops.empty() ? 0.0 : (semantic_weighted_loops * 100.0 / loops.size()));
  // RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Geometry-only: %d (%.1f%%)", 
  //             geometry_only_loops,
  //             loops.empty() ? 0.0 : (geometry_only_loops * 100.0 / loops.size()));
  
  if (semantic_weighted_loops > 0) {
    RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Avg CLIP similarity: %.4f", 
                avg_clip_similarity / semantic_weighted_loops);
    RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Avg scaling factor: %.4f", 
                avg_alpha / semantic_weighted_loops);
  }
  
  // RCLCPP_WARN(node_->get_logger(), "╔════════════════════════════════════════════════════════════╗");
}

void LoopMapper::set_data(g2o::VertexSE3* keyframe_node) {
  auto current_key_data = dynamic_cast<OptimizationData*>(keyframe_node->userData());
  if (current_key_data) {
    current_key_data->set_loop_closure_info(true);
  } else {
    OptimizationData* data = new OptimizationData();
    data->set_loop_closure_info(true);
    keyframe_node->setUserData(data);
  }
}

bool LoopMapper::get_floor_data(g2o::VertexSE3* keyframe_node) {
  bool on_stairs = false;
  auto current_key_data = dynamic_cast<OptimizationData*>(keyframe_node->userData());
  if (current_key_data) {
    current_key_data->get_stair_node_info(on_stairs);
  }
  return on_stairs;
}

double LoopMapper::computeClipSimilarity(const std::vector<float>& emb1,
                                         const std::vector<float>& emb2)
{
  if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
    RCLCPP_WARN(node_->get_logger(), 
                "[CLIP SIMILARITY] ERROR - Embedding size mismatch! emb1: %zu, emb2: %zu",
                emb1.size(), emb2.size());
    return 0.5;  // Neutral weight if embeddings unavailable or mismatched
  }
  
  double dot_product = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;

  for (size_t i = 0; i < emb1.size(); ++i)
  {
    dot_product += emb1[i] * emb2[i];
    norm1 += emb1[i] * emb1[i];
    norm2 += emb2[i] * emb2[i];
  }

  if (norm1 <= 0.0 || norm2 <= 0.0) {
    RCLCPP_WARN(node_->get_logger(), 
                "[CLIP SIMILARITY] ERROR - Zero norm detected! norm1: %.4f, norm2: %.4f",
                norm1, norm2);
    return 0.5;
  }

  double cosine_sim = dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
  
  RCLCPP_DEBUG(node_->get_logger(), 
               "[CLIP] Computed cosine similarity: %.6f (embedding dim: %zu)",
               cosine_sim, emb1.size());

  return (cosine_sim + 1.0) / 2.0;
}

double LoopMapper::confidenceToAlpha(double clip_similarity) {
  // Tunable parameters
  const double alpha_min = 0.1;   // Minimum scaling (low confidence)
  const double alpha_max = 2.0;   // Maximum scaling (high confidence)
  const double threshold = 0.7;   // Similarity threshold for neutral weighting
  
  double alpha = 1.0;
  
  if (clip_similarity >= threshold) {
    // High similarity: boost information matrix (more trust in loop)
    double t = (clip_similarity - threshold) / (1.0 - threshold);
    alpha = 1.0 + t * (alpha_max - 1.0);
    RCLCPP_DEBUG(node_->get_logger(), 
                 "[ALPHA CALC] High similarity (%.4f >= %.4f) → BOOST alpha = %.4f",
                 clip_similarity, threshold, alpha);
  } else {
    // Low similarity: reduce information matrix (less trust)
    double t = clip_similarity / threshold;
    alpha = alpha_min + t * (1.0 - alpha_min);
    RCLCPP_DEBUG(node_->get_logger(), 
                 "[ALPHA CALC] Low similarity (%.4f < %.4f) → REDUCE alpha = %.4f",
                 clip_similarity, threshold, alpha);
  }
  
  return alpha;
}

double LoopMapper::computeObjectOverlap(const std::vector<std::string>& objects1,
                                         const std::vector<std::string>& objects2) {
  if (objects1.empty() || objects2.empty()) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "[OBJECT OVERLAP] One or both keyframes have no objects");
    return 0.5;  // Neutral if either has no objects
  }

  // Count common object types
  int overlap = 0;
  for (const auto& obj1 : objects1) {
    // Check if this object type appears in objects2
    if (std::find(objects2.begin(), objects2.end(), obj1) != objects2.end()) {
      overlap++;
    }
  }

  // Jaccard Index: intersection / union
  int total_unique = objects1.size() + objects2.size() - overlap;
  double jaccard_index = (total_unique > 0) ? (double)overlap / total_unique : 0.5;

  RCLCPP_DEBUG(node_->get_logger(),
               "[OBJECT OVERLAP] objects1: %zu, objects2: %zu, overlap: %d, "
               "Jaccard index: %.4f",
               objects1.size(), objects2.size(), overlap, jaccard_index);

  return jaccard_index;  // [0, 1]
}

}  // namespace s_graphs