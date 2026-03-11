#include <s_graphs/backend/loop_mapper.hpp>
#include <fstream>

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

void LoopMapper::add_loops(const std::shared_ptr<GraphSLAM>& covisibility_graph,
                           const std::vector<Loop::Ptr>& loops) {
  RCLCPP_WARN(node_->get_logger(), "╔════════════════════════════════════════════════════════════╗");
  RCLCPP_WARN(node_->get_logger(), "║         LOOP CLOSURE SEMANTIC WEIGHTING PIPELINE            ║");
  RCLCPP_WARN(node_->get_logger(), "╚════════════════════════════════════════════════════════════╝");
  RCLCPP_WARN(node_->get_logger(), "[SEMANTIC LOOP MAPPER] Processing %zu loop closures", loops.size());
  
  int semantic_weighted_loops = 0;
  int geometry_only_loops = 0;
  double avg_clip_similarity = 0.0;
  double avg_alpha = 0.0;
  
  for (const auto& loop : loops) {
    RCLCPP_WARN(node_->get_logger(), "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

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
    
    RCLCPP_WARN(node_->get_logger(), "[GEOMETRY] ICP fitness score: %.6f", icp_fitness);
    RCLCPP_WARN(node_->get_logger(), "[GEOMETRY] Information matrix condition number: %.4f",
                information_matrix.norm());

    double semantic_alpha = 1.0;
    std::string clip_status = "❌ UNAVAILABLE";
    
    // Check for CLIP embeddings
    if (loop->key1->clip_embedding && loop->key2->clip_embedding) {
      RCLCPP_WARN(node_->get_logger(), "[CLIP] ✓ Both keyframes have CLIP embeddings!");
      
      double clip_sim = computeClipSimilarity(
          loop->key1->clip_embedding.value(),
          loop->key2->clip_embedding.value());
      csv_has_clip = true;
      csv_clip_sim = clip_sim;
      
      // Compute object overlap
      double object_overlap = 0.5;  // Default neutral
      if (loop->key1->detected_objects && loop->key2->detected_objects) {
        object_overlap = computeObjectOverlap(
            loop->key1->detected_objects.value(),
            loop->key2->detected_objects.value());
        csv_object_overlap = object_overlap;
        
        // Log object overlap with confidence assessment
        std::string object_confidence;
        if (object_overlap >= 0.7) {
          object_confidence = "🟢 STRONG (objects match well)";
        } else if (object_overlap >= 0.5) {
          object_confidence = "🟡 MODERATE (partial object match)";
        } else {
          object_confidence = "🔴 WEAK (objects don't match)";
        }
        RCLCPP_WARN(node_->get_logger(),
                    "[OBJECT OVERLAP] Score: %.4f %s", object_overlap,
                    object_confidence.c_str());
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "[OBJECT OVERLAP] ❌ One or both keyframes missing objects");
      }
      
      semantic_alpha = confidenceToAlpha(clip_sim);
      
      // Apply object overlap adjustment
      if (object_overlap > 0.7) {
        semantic_alpha *= 1.15;  // Boost by 15% if objects strongly match
        RCLCPP_WARN(node_->get_logger(),
                    "[SEMANTIC BOOST] Strong object match → α *= 1.15");
      } else if (object_overlap < 0.3) {
        semantic_alpha *= 0.85;  // Reduce by 15% if objects don't match
        RCLCPP_WARN(node_->get_logger(),
                    "[SEMANTIC PENALTY] Weak object match → α *= 0.85");
      }
      
      // GEOMETRY-SEMANTIC INTERACTION: Modulate semantic influence based on ICP fitness
      // When geometry is very reliable, semantic signals matter less (they should just confirm)
      // When geometry is unreliable, semantic signals become critical
      double geometric_reliability = 1.0 - (icp_fitness / fitness_score_thresh);
      // Clamp to [0, 1]: 0 = fitness at threshold (unreliable), 1 = perfect match (very reliable)
      geometric_reliability = std::max(0.0, std::min(1.0, geometric_reliability));
      
      // When geometry is excellent (rel ~1.0), dampen semantic deviations
      // When geometry is poor (rel ~0.0), amplify semantic deviations
      double damping_factor = 0.5;  // Controls how much geometry reliability dampens semantics
      double semantic_influence = 1.0 - (geometric_reliability * damping_factor);
      
      // Apply geometry-semantic interaction
      // If semantic_alpha would boost (> 1.0), reduce the boost when geometry is excellent
      // If semantic_alpha would penalize (< 1.0), reduce the penalty when geometry is excellent
      double alpha_before_geometry = semantic_alpha;
      semantic_alpha = 1.0 + (semantic_alpha - 1.0) * semantic_influence;
      
      RCLCPP_WARN(node_->get_logger(),
                  "[GEOMETRY-SEMANTIC] Geometric reliability: %.4f, "
                  "Semantic influence factor: %.4f, "
                  "Alpha adjusted: %.4f → %.4f",
                  geometric_reliability, semantic_influence, 
                  alpha_before_geometry, semantic_alpha);
      
      // IMAGE QUALITY GATING: Reduce semantic trust when image quality is poor
      // IQA score from Python node stored in keyframe->image_brightness
      // IQA score is NORMALIZED to [0, 1] range using sigmoid in Python
      double quality_gate = 1.0;  // Default: full semantic trust
      if (loop->key1->image_brightness && loop->key2->image_brightness) {
        double q1 = static_cast<double>(loop->key1->image_brightness.value());
        double q2 = static_cast<double>(loop->key2->image_brightness.value());
        double min_quality = std::min(q1, q2);
        
        // IQA score is already normalized to [0, 1] by sigmoid in Python
        // Use directly as quality gate
        quality_gate = min_quality;
        
        // Blend: when quality is low, regress alpha toward neutral (1.0)
        double alpha_before_quality = semantic_alpha;
        semantic_alpha = quality_gate * semantic_alpha + (1.0 - quality_gate) * 1.0;
        csv_quality_gate = quality_gate;
        
        RCLCPP_WARN(node_->get_logger(),
                    "[IMAGE QUALITY] q1=%.4f, q2=%.4f, min=%.4f, gate=%.4f, "
                    "Alpha: %.4f → %.4f",
                    q1, q2, min_quality, quality_gate,
                    alpha_before_quality, semantic_alpha);
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "[IMAGE QUALITY] ❌ Quality scores unavailable (IQA node not running?)");
      }

      // === SCENE DYNAMICITY FACTOR (DynaTrack) ===
      // Penalize loop closures in highly dynamic scenes where point cloud
      // matching is less reliable due to moving objects.
      if (loop->key1->scene_dynamicity && loop->key2->scene_dynamicity) {
        double dyn1 = static_cast<double>(loop->key1->scene_dynamicity.value());
        double dyn2 = static_cast<double>(loop->key2->scene_dynamicity.value());
        double max_dyn = std::max(dyn1, dyn2);

        // Penalize proportionally: high dynamicity → reduce trust
        // dyn_factor ranges from 1.0 (static scene) to 0.7 (fully dynamic)
        double dyn_penalty_strength = 0.3;  // max 30% reduction
        double dyn_factor = 1.0 - max_dyn * dyn_penalty_strength;

        double alpha_before_dyn = semantic_alpha;
        semantic_alpha *= dyn_factor;
        csv_dyn_factor = dyn_factor;

        RCLCPP_WARN(node_->get_logger(),
                    "[DYNAMICITY] dyn1=%.3f, dyn2=%.3f, max=%.3f, factor=%.3f, "
                    "Alpha: %.4f → %.4f",
                    dyn1, dyn2, max_dyn, dyn_factor,
                    alpha_before_dyn, semantic_alpha);
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "[DYNAMICITY] ❌ Dynamicity scores unavailable (DynaTrack not running?)");
      }

      // === ZONE-BASED SEMANTIC COHERENCE FACTOR ===
      // If both keyframes belong to known zones, use zone information to
      // modulate loop closure trust:
      // - Same zone → boost (semantically coherent)
      // - Different zones with shared support planes → neutral/mild penalty
      // - Different zones with NO shared planes → strong penalty
      if (zone_cache_ && use_zone_prefilter_) {
        int z1 = zone_cache_->get_zone_for_keyframe(loop->key1->node->id());
        int z2 = zone_cache_->get_zone_for_keyframe(loop->key2->node->id());
        csv_zone1 = z1;
        csv_zone2 = z2;

        if (z1 >= 0 && z2 >= 0) {
          double zone_factor = 1.0;
          if (z1 == z2) {
            // Same zone — strong semantic agreement, boost confidence
            double zone_conf = zone_cache_->get_zone_confidence(z1);
            zone_factor = 1.0 + zone_conf * 0.2;  // up to 20% boost
            RCLCPP_WARN(node_->get_logger(),
                        "[ZONE] ✓ SAME zone %d (conf=%.2f), boost factor=%.3f",
                        z1, zone_conf, zone_factor);
          } else {
            // Different zones — check shared support planes
            auto planes1 = zone_cache_->get_support_plane_ids(z1);
            auto planes2 = zone_cache_->get_support_plane_ids(z2);
            int shared_planes = 0;
            for (int pid : planes1) {
              if (planes2.count(pid)) shared_planes++;
            }

            if (shared_planes > 0) {
              // Adjacent zones (shared walls) — mild penalty
              zone_factor = 0.9;
              RCLCPP_WARN(node_->get_logger(),
                          "[ZONE] ⚠ Different zones %d↔%d, %d shared planes, "
                          "mild penalty factor=%.3f",
                          z1, z2, shared_planes, zone_factor);
            } else {
              // Distant zones (no shared walls) — strong penalty
              double conf1 = zone_cache_->get_zone_confidence(z1);
              double conf2 = zone_cache_->get_zone_confidence(z2);
              double min_conf = std::min(conf1, conf2);
              // Scale penalty by confidence: high-confidence zones → stronger penalty
              zone_factor = 1.0 - min_conf * 0.5;  // up to 50% reduction
              zone_factor = std::max(0.3, zone_factor);  // never go below 30%

              RCLCPP_WARN(node_->get_logger(),
                          "[ZONE] ✗ DIFFERENT zones %d↔%d, NO shared planes, "
                          "confs=(%.2f, %.2f), penalty factor=%.3f",
                          z1, z2, conf1, conf2, zone_factor);
            }
          }

          double alpha_before_zone = semantic_alpha;
          semantic_alpha *= zone_factor;
          csv_zone_factor = zone_factor;
          csv_same_zone = (z1 == z2);
          RCLCPP_WARN(node_->get_logger(),
                      "[ZONE] Alpha: %.4f → %.4f", alpha_before_zone, semantic_alpha);
        } else {
          RCLCPP_WARN(node_->get_logger(),
                      "[ZONE] Keyframe(s) not in any zone (kf%d→z%d, kf%d→z%d)",
                      loop->key1->node->id(), z1,
                      loop->key2->node->id(), z2);
        }
      } else if (!zone_cache_) {
        RCLCPP_WARN(node_->get_logger(),
                    "[ZONE] ❌ Zone cache not available");
      }
      
      semantic_weighted_loops++;
      avg_clip_similarity += clip_sim;
      avg_alpha += semantic_alpha;
      
      // Determine confidence level
      std::string confidence_level;
      if (clip_sim >= 0.85) {
        confidence_level = "🟢 VERY HIGH";
      } else if (clip_sim >= 0.70) {
        confidence_level = "🟡 HIGH";
      } else if (clip_sim >= 0.50) {
        confidence_level = "🟠 MEDIUM";
      } else {
        confidence_level = "🔴 LOW";
      }
      
      RCLCPP_WARN(node_->get_logger(), 
                  "[CLIP SIMILARITY] %.4f %s", clip_sim, confidence_level.c_str());
      RCLCPP_WARN(node_->get_logger(), 
                  "[INFORMATION MATRIX SCALING] Alpha (after geometry-semantic adjustment) = %.4f", semantic_alpha);
      
      // Show scaling direction
      if (semantic_alpha > 1.0) {
        RCLCPP_WARN(node_->get_logger(), 
                    "[SCALING] 📈 BOOSTING information matrix (trust loop closure)");
      } else if (semantic_alpha < 1.0) {
        RCLCPP_WARN(node_->get_logger(), 
                    "[SCALING] 📉 REDUCING information matrix (less trust in loop)");
      } else {
        RCLCPP_WARN(node_->get_logger(), 
                    "[SCALING] ➡️  NEUTRAL scaling (alpha = 1.0)");
      }
      
      clip_status = "✓ USED FOR WEIGHTING";
    } else {
      geometry_only_loops++;
      RCLCPP_WARN(node_->get_logger(), 
                  "[CLIP] Missing embeddings - Using geometry-only weighting (alpha = 1.0)");
      if (!loop->key1->clip_embedding) {
        RCLCPP_WARN(node_->get_logger(), "  → Keyframe %d: No CLIP embedding", loop->key1->node->id());
      }
      if (!loop->key2->clip_embedding) {
        RCLCPP_WARN(node_->get_logger(), "  → Keyframe %d: No CLIP embedding", loop->key2->node->id());
      }
    }

    // Apply semantic weighting
    information_matrix *= semantic_alpha;
    
    RCLCPP_WARN(node_->get_logger(), "[SCALED MATRIX] Condition number: %.4f",
                information_matrix.norm());

    shared_graph_mutex.lock();
    
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
  RCLCPP_WARN(node_->get_logger(), "╔════════════════════════════════════════════════════════════╗");
  RCLCPP_WARN(node_->get_logger(), "║               SEMANTIC WEIGHTING SUMMARY                    ║");
  RCLCPP_WARN(node_->get_logger(), "╚════════════════════════════════════════════════════════════╝");
  RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Total loops: %zu", loops.size());
  RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Semantically weighted: %d (%.1f%%)", 
              semantic_weighted_loops, 
              loops.empty() ? 0.0 : (semantic_weighted_loops * 100.0 / loops.size()));
  RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Geometry-only: %d (%.1f%%)", 
              geometry_only_loops,
              loops.empty() ? 0.0 : (geometry_only_loops * 100.0 / loops.size()));
  
  if (semantic_weighted_loops > 0) {
    RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Avg CLIP similarity: %.4f", 
                avg_clip_similarity / semantic_weighted_loops);
    RCLCPP_WARN(node_->get_logger(), "[SUMMARY] Avg scaling factor: %.4f", 
                avg_alpha / semantic_weighted_loops);
  }
  
  RCLCPP_WARN(node_->get_logger(), "╔════════════════════════════════════════════════════════════╗");
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