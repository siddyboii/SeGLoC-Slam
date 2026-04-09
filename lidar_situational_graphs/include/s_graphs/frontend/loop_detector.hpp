/*
Copyright (c) 2023, University of Luxembourg
All rights reserved.

Redistributions and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*/

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <g2o/types/slam3d/vertex_se3.h>

#include <boost/format.hpp>
#include <s_graphs/backend/graph_slam.hpp>
#include <s_graphs/common/keyframe.hpp>
#include <s_graphs/common/point_types.hpp>
#include <s_graphs/common/registrations.hpp>
#include <s_graphs/common/scene_descriptor.hpp>

#include <algorithm>
#include <set>

namespace s_graphs {

/**
 * @brief Struct Loop
 */
struct Loop {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Loop>;

  /**
   * @brief Constructor of struct Loop
   *
   * @param key1
   * @param key2
   * @param relpose
   * @param fitness_score ICP scan matching fitness score
   */
  Loop(const KeyFrame::Ptr& key1,
       const KeyFrame::Ptr& key2,
       const Eigen::Matrix4f& relpose,
       double fitness_score)
      : key1(key1), key2(key2), relative_pose(relpose), fitness_score(fitness_score) {}

 public:
  KeyFrame::Ptr key1;
  KeyFrame::Ptr key2;
  Eigen::Matrix4f relative_pose;
  double fitness_score;  // ICP scan matching fitness score from registration
};

/**
 * @brief This class finds loops by scam matching and adds them to the pose graph
 */
class LoopDetector {
 public:
  /**
   * @brief Constructor of the class LoopDetector
   *
   * @param node
   */
  ~LoopDetector() {
    std::cout << "\n\033[1;35m[LOOP_STATS] ═══ FINAL SUMMARY ═══\033[0m" << std::endl;
    print_loop_stats();
  }

  LoopDetector(const rclcpp::Node::SharedPtr node, std::mutex& graph_mutex)
      : shared_graph_mutex(graph_mutex) {
    distance_thresh =
        node->get_parameter("distance_thresh").get_parameter_value().get<double>();
    accum_distance_thresh = node->get_parameter("accum_distance_thresh")
                                .get_parameter_value()
                                .get<double>();
    distance_from_last_edge_thresh =
        node->get_parameter("min_edge_interval").get_parameter_value().get<double>();

    fitness_score_max_range = node->get_parameter("fitness_score_max_range")
                                  .get_parameter_value()
                                  .get<double>();
    icp_trans_cap = node->get_parameter("icp_trans_cap").get_parameter_value().get<double>();
    fitness_score_thresh =
        node->get_parameter("fitness_score_thresh").get_parameter_value().get<double>();

    keyframe_matching_threshold = node->get_parameter("keyframe_matching_threshold")
                                      .get_parameter_value()
                                      .get<double>();

    // Semantic loop proposal parameters
    enable_semantic_proposal_ = node->get_parameter("enable_semantic_loop_proposal")
                                    .get_parameter_value()
                                    .get<bool>();
    semantic_proposal_thresh_ = node->get_parameter("semantic_proposal_thresh")
                                    .get_parameter_value()
                                    .get<double>();
    semantic_proposal_max_candidates_ = node->get_parameter("semantic_proposal_max_candidates")
                                            .get_parameter_value()
                                            .get<int>();

    // GNN-specific threshold (separate from scene descriptor threshold)
    gnn_proposal_thresh_ = node->get_parameter("gnn_proposal_thresh")
                               .get_parameter_value()
                               .get<double>();

    // Temporal consistency: require N consecutive keyframes to agree
    temporal_consistency_required_ = node->get_parameter("loop_temporal_consistency")
                                        .get_parameter_value()
                                        .get<int>();
    std::cout << "[LOOP_DETECT] Temporal consistency: "
              << temporal_consistency_required_ << " consecutive matches required"
              << std::endl;

    s_graphs::registration_params params;
    params = {
        node->get_parameter("registration_method")
            .get_parameter_value()
            .get<std::string>(),
        node->get_parameter("reg_num_threads").get_parameter_value().get<int>(),
        node->get_parameter("reg_transformation_epsilon")
            .get_parameter_value()
            .get<double>(),
        node->get_parameter("reg_maximum_iterations").get_parameter_value().get<int>(),
        node->get_parameter("reg_max_correspondence_distance")
            .get_parameter_value()
            .get<double>(),
        node->get_parameter("reg_correspondence_randomness")
            .get_parameter_value()
            .get<int>(),
        node->get_parameter("reg_resolution").get_parameter_value().get<double>(),
        node->get_parameter("reg_use_reciprocal_correspondences")
            .get_parameter_value()
            .get<bool>(),
        node->get_parameter("reg_max_optimizer_iterations")
            .get_parameter_value()
            .get<int>(),
        node->get_parameter("reg_nn_search_method")
            .get_parameter_value()
            .get<std::string>()};

    registration = select_registration_method(params);
    last_edge_accum_distance = 0.0;
  }

  /**
   * @brief Detect loops and add them to the pose graph
   *
   * @param keyframes
   *          Keyframes
   * @param new_keyframes
   *          Newly registered keyframes
   * @return Loop vector
   */
  std::vector<Loop::Ptr> detect(const std::map<int, KeyFrame::Ptr>& keyframes,
                                const std::deque<KeyFrame::Ptr>& new_keyframes) {
    std::vector<Loop::Ptr> detected_loops;
    for (const auto& new_keyframe : new_keyframes) {
      detect_call_count_++;

      // Geometric candidates (spatial proximity)
      auto candidates = find_candidates(keyframes, new_keyframe);
      int geo_count = static_cast<int>(candidates.size());

      // Track which candidates are semantic (GNN or DESC)
      std::set<int> semantic_candidate_ids;
      std::set<int> gnn_candidate_ids;

      // Semantic candidates (scene descriptor / GNN similarity)
      if (enable_semantic_proposal_) {
        auto semantic_candidates = find_semantic_candidates(keyframes, new_keyframe);
        int sem_count = 0;
        for (const auto& sc : semantic_candidates) {
          if (candidates.find(sc.first) == candidates.end()) {
            candidates.insert(sc);
            sem_count++;
          }
          semantic_candidate_ids.insert(sc.first);
          // Check if this candidate used GNN
          if (new_keyframe->gnn_embedding && sc.second->gnn_embedding) {
            gnn_candidate_ids.insert(sc.first);
          }
        }
        if (sem_count > 0) {
          int total_gnn = static_cast<int>(gnn_candidate_ids.size());
          int total_desc = static_cast<int>(semantic_candidate_ids.size()) - total_gnn;
          std::cout << "\033[36m[LOOP_DETECT]\033[0m KF" << new_keyframe->id()
                    << ": " << geo_count << " geometric + " << sem_count
                    << " new semantic (" << total_gnn << " GNN, "
                    << total_desc << " DESC)" << std::endl;
        }
      }

      auto loop = matching(candidates, new_keyframe);
      if (loop) {
        int matched_id = loop->key2->node->id();

        // Determine source of the winning candidate
        std::string source;
        if (gnn_candidate_ids.count(matched_id)) {
          source = "GNN";
        } else if (semantic_candidate_ids.count(matched_id)) {
          source = "DESC";
        } else {
          source = "GEO";
        }

        // ── Temporal consistency gate ──
        // Check if this match is in the same spatial neighborhood as the pending match.
        // We use accum_distance (trajectory length) because graph node IDs are not sequential.
        double matched_accum_dist = loop->key2->accum_distance;
        bool same_region = (pending_matched_accum_dist_ >= 0.0) &&
                           (std::abs(matched_accum_dist - pending_matched_accum_dist_) <= 5.0);

        if (same_region) {
          pending_consecutive_count_++;
          // Update to the latest loop (freshest ICP result)
          pending_loop_ = loop;
          pending_matched_accum_dist_ = matched_accum_dist;
          pending_source_ = source;
        } else {
          // New or different region — start fresh streak
          pending_consecutive_count_ = 1;
          pending_loop_ = loop;
          pending_matched_accum_dist_ = matched_accum_dist;
          pending_source_ = source;
        }

        std::cout << "\033[33m[LOOP_PENDING]\033[0m KF" << new_keyframe->id()
                  << " ↔ KF" << loop->key2->id()
                  << " (source=" << source
                  << ", score=" << loop->fitness_score
                  << ", streak=" << pending_consecutive_count_
                  << "/" << temporal_consistency_required_ << ")" << std::endl;

        // Accept only when streak reaches the required count
        if (pending_consecutive_count_ >= temporal_consistency_required_) {
          detected_loops.push_back(pending_loop_);

          // Track source statistics
          if (pending_source_ == "GNN") loops_from_gnn_++;
          else if (pending_source_ == "DESC") loops_from_desc_++;
          else loops_from_geo_++;

          std::cout << "\033[32m[LOOP_VERIFIED]\033[0m KF"
                    << pending_loop_->key1->id()
                    << " ↔ KF" << pending_loop_->key2->id()
                    << " (source=" << pending_source_
                    << ", score=" << pending_loop_->fitness_score
                    << ", confirmed after " << pending_consecutive_count_
                    << " consecutive matches)" << std::endl;

          // Reset pending state after acceptance
          pending_consecutive_count_ = 0;
          pending_matched_accum_dist_ = -1.0;
          pending_loop_ = nullptr;
          pending_source_ = "";
        }
      } else {
        // No match at this keyframe — break the streak
        if (pending_consecutive_count_ > 0) {
          std::cout << "\033[31m[LOOP_REJECTED]\033[0m streak broken at "
                    << pending_consecutive_count_
                    << "/" << temporal_consistency_required_
                    << " (pending target dist=" << pending_matched_accum_dist_
                    << "m)" << std::endl;
        }
        pending_consecutive_count_ = 0;
        pending_matched_accum_dist_ = -1.0;
        pending_loop_ = nullptr;
        pending_source_ = "";
      }

      // Periodic stats
      if (detect_call_count_ % 50 == 0) {
        print_loop_stats();
      }
    }

    return detected_loops;
  }

  std::vector<Loop::Ptr> detect(const std::map<int, KeyFrame::Ptr>& keyframes,
                                const std::vector<KeyFrame::Ptr>& new_keyframes) {
    // Delegate to deque version
    std::deque<KeyFrame::Ptr> kf_deque(new_keyframes.begin(), new_keyframes.end());
    return detect(keyframes, kf_deque);
  }

  std::vector<Loop::Ptr> detectWithAllKeyframes(
      const std::map<int, KeyFrame::Ptr>& keyframes,
      const std::vector<KeyFrame::Ptr>& new_keyframes) {
    std::vector<Loop::Ptr> detected_loops;
    for (const auto& new_keyframe : new_keyframes) {
      auto loop = matching(keyframes, new_keyframe, false);
      if (loop) {
        detected_loops.push_back(loop);
      }
    }

    return detected_loops;
  }

  bool matching(const KeyFrame::Ptr& keyframe,
                const KeyFrame::Ptr& prev_keyframe,
                Eigen::Matrix4f& relative_pose) {
    relative_pose.setIdentity();
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());

    registration->setInputTarget(prev_keyframe->cloud);
    registration->setInputSource(keyframe->cloud);

    shared_graph_mutex.lock();
    Eigen::Isometry3d prev_keyframe_estimate = prev_keyframe->node->estimate();
    shared_graph_mutex.unlock();

    prev_keyframe_estimate.linear() =
        Eigen::Quaterniond(prev_keyframe_estimate.linear())
            .normalized()
            .toRotationMatrix();

    shared_graph_mutex.lock();
    Eigen::Isometry3d keyframe_estimate = keyframe->node->estimate();
    shared_graph_mutex.unlock();

    keyframe_estimate.linear() =
        Eigen::Quaterniond(keyframe_estimate.linear()).normalized().toRotationMatrix();

    Eigen::Matrix4f guess =
        (keyframe_estimate.inverse() * prev_keyframe_estimate).matrix().cast<float>();
    guess(2, 3) = 0.0;
    registration->align(*aligned, guess);

    double score = registration->getFitnessScore(fitness_score_max_range);

    if (!registration->hasConverged() || score > keyframe_matching_threshold) {
      return false;
    }

    relative_pose = registration->getFinalTransformation();
    return true;
  }

  /**
   * @brief
   *
   * @return Distance treshold
   */
  double get_distance_thresh() const { return distance_thresh; }

 private:
  /**
   * @brief Find loop candidates. A detected loop begins at one of #keyframes and ends
   * at #new_keyframe
   *
   * @param keyframes
   *          Candidate keyframes of loop start
   * @param new_keyframe
   *          Loop end keyframe
   * @return Loop candidates
   */
  std::map<int, KeyFrame::Ptr> find_candidates(
      const std::map<int, KeyFrame::Ptr>& keyframes,
      const KeyFrame::Ptr& new_keyframe) const {
    // too close to the last registered loop edge
    if (new_keyframe->accum_distance - last_edge_accum_distance <
        distance_from_last_edge_thresh) {
      return std::map<int, KeyFrame::Ptr>();
    }

    std::map<int, KeyFrame::Ptr> candidates;
    // candidates.reserve(32);

    for (const auto& k : keyframes) {
      // traveled distance between keyframes is too small
      if (new_keyframe->accum_distance - k.second->accum_distance <
          accum_distance_thresh) {
        continue;
      }

      // Minimum keyframe index gap: prevents matching nearly-consecutive frames
      // that happen to be spatially close due to odometry drift or a tight local loop.
      // At keyframe_delta_trans=2m, 5 keyframes = ~10m minimum loop path length.
      // The accum_distance_thresh already ensures sufficient traveled distance.
      int kf_gap = new_keyframe->node->id() - k.second->node->id();
      if (kf_gap < 5) {
        continue;
      }

      if (new_keyframe->floor_level != k.second->floor_level) continue;

      shared_graph_mutex.lock();
      const auto& pos1 = k.second->node->estimate().translation();
      const auto& pos2 = new_keyframe->node->estimate().translation();
      shared_graph_mutex.unlock();

      // estimated distance between keyframes is too small
      double dist = (pos1.head<2>() - pos2.head<2>()).norm();
      if (dist > distance_thresh) {
        continue;
      }

      candidates.insert({k.first, k.second});
    }

    return candidates;
  }

  /**
   * @brief Find loop candidates using scene descriptor similarity.
   * Bypasses spatial distance filter — finds candidates that LOOK similar
   * even if estimated positions are far apart due to drift.
   *
   * @param keyframes  All existing keyframes
   * @param new_keyframe  Current keyframe to find loops for
   * @return Loop candidates sorted by descriptor similarity
   */
  std::map<int, KeyFrame::Ptr> find_semantic_candidates(
      const std::map<int, KeyFrame::Ptr>& keyframes,
      const KeyFrame::Ptr& new_keyframe) const {
    // Need scene descriptor or GNN embedding on the new keyframe
    if (!new_keyframe->scene_descriptor && !new_keyframe->gnn_embedding) {
      return std::map<int, KeyFrame::Ptr>();
    }

    // Too close to the last registered loop edge
    if (new_keyframe->accum_distance - last_edge_accum_distance <
        distance_from_last_edge_thresh) {
      return std::map<int, KeyFrame::Ptr>();
    }

    // Collect (similarity, kf_id) pairs
    std::vector<std::pair<double, int>> scored_candidates;

    for (const auto& k : keyframes) {
      // Same basic filters as geometric (except NO distance_thresh filter)
      if (new_keyframe->accum_distance - k.second->accum_distance <
          accum_distance_thresh) {
        continue;
      }

      int kf_gap = new_keyframe->node->id() - k.second->node->id();
      if (kf_gap < 5) {
        continue;
      }

      if (new_keyframe->floor_level != k.second->floor_level) continue;

      double sim = -1.0;
      std::string method = "none";

      // Prefer GNN embeddings if both keyframes have them
      if (new_keyframe->gnn_embedding && k.second->gnn_embedding) {
        const auto& emb_a = new_keyframe->gnn_embedding.value();
        const auto& emb_b = k.second->gnn_embedding.value();
        // Dot product of L2-normalized vectors = cosine similarity
        sim = 0.0;
        for (size_t i = 0; i < emb_a.size() && i < emb_b.size(); i++) {
          sim += emb_a[i] * emb_b[i];
        }
        method = "GNN";
      } else if (new_keyframe->scene_descriptor && k.second->scene_descriptor) {
        // Fallback to scene descriptor
        sim = SceneDescriptor::similarity(
            new_keyframe->scene_descriptor.value(),
            k.second->scene_descriptor.value(),
            0.5, 0.5);
        method = "DESC";
      }

      // Use separate thresholds for GNN vs DESC
      double threshold = (method == "GNN") ? gnn_proposal_thresh_ : semantic_proposal_thresh_;
      if (sim > threshold) {
        scored_candidates.push_back({sim, k.first});
      }
    }

    // Sort by similarity (highest first)
    std::sort(scored_candidates.begin(), scored_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Take top-K
    std::map<int, KeyFrame::Ptr> candidates;
    int count = 0;
    for (const auto& sc : scored_candidates) {
      if (count >= semantic_proposal_max_candidates_) break;
      auto it = keyframes.find(sc.second);
      if (it != keyframes.end()) {
        candidates.insert(*it);
        std::string method = (it->second->gnn_embedding && new_keyframe->gnn_embedding)
            ? "GNN" : "DESC";
        std::cout << "[SEMANTIC_PROPOSAL] Candidate KF" << sc.second
                  << " (" << method << " sim=" << sc.first << ")" << std::endl;
        count++;
      }
    }

    return candidates;
  }

  /**
   * @brief To validate a loop candidate this function applies a scan matching between
   * keyframes consisting the loop. If they are matched well, the loop is added to the
   * pose graph
   *
   * @param candidate_keyframes
   *          candidate keyframes of loop start
   * @param new_keyframe
   *          loop end keyframe
   * @return Loop pointer
   */
  Loop::Ptr matching(const std::map<int, KeyFrame::Ptr>& candidate_keyframes,
                     const KeyFrame::Ptr& new_keyframe,
                     bool use_prior = true) {
    if (candidate_keyframes.empty() || new_keyframe->cloud->points.empty()) {
      return nullptr;
    }

    registration->setInputTarget(new_keyframe->cloud);

    double best_score = std::numeric_limits<double>::max();
    KeyFrame::Ptr best_matched;
    Eigen::Matrix4f relative_pose;

    auto t1 = rclcpp::Clock{}.now();

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    for (const auto& candidate : candidate_keyframes) {
      if (candidate.second->cloud->points.empty()) continue;
      registration->setInputSource(candidate.second->cloud);

      shared_graph_mutex.lock();
      Eigen::Isometry3d new_keyframe_estimate = new_keyframe->node->estimate();
      shared_graph_mutex.unlock();

      new_keyframe_estimate.linear() =
          Eigen::Quaterniond(new_keyframe_estimate.linear())
              .normalized()
              .toRotationMatrix();

      shared_graph_mutex.lock();
      Eigen::Isometry3d candidate_estimate = candidate.second->node->estimate();
      shared_graph_mutex.unlock();

      candidate_estimate.linear() = Eigen::Quaterniond(candidate_estimate.linear())
                                        .normalized()
                                        .toRotationMatrix();

      if (use_prior) {
        Eigen::Matrix4f guess = (new_keyframe_estimate.inverse() * candidate_estimate)
                                    .matrix()
                                    .cast<float>();
        guess(2, 3) = 0.0;

        // Try the odometry-based guess first
        registration->align(*aligned, guess);
        double score = registration->getFitnessScore(fitness_score_max_range);
        Eigen::Matrix4f best_pose = registration->getFinalTransformation();
        bool converged = registration->hasConverged();

        // If odometry guess failed, also try with 180-degree yaw rotation applied.
        // This handles the case where robot retraces a path from the opposite direction:
        // the odometry-derived relative pose has correct translation but the scan
        // orientations are ~180° apart, causing ICP to land in a local minimum.
        if (!converged || score > fitness_score_thresh) {
          Eigen::Matrix4f rot180 = Eigen::Matrix4f::Identity();
          rot180(0, 0) = -1.0f;  // cos(180°)
          rot180(1, 1) = -1.0f;  // cos(180°)
          Eigen::Matrix4f guess180 = guess * rot180;
          registration->align(*aligned, guess180);
          double score180 = registration->getFitnessScore(fitness_score_max_range);
          if (registration->hasConverged() && score180 < score) {
            score = score180;
            best_pose = registration->getFinalTransformation();
            converged = true;
          }
        }

        if (!converged || score > best_score) {
          continue;
        }

        // ── Pose sanity check: reject implausible yaw angles ──
        // For a wheeled robot, valid loop closures have relative yaw near:
        //   0°   → same direction (robot passed same spot same way)
        //   180° → opposite direction (robot retraced path in reverse)
        // Angles in [30°, 150°] indicate ICP landed in a wrong local minimum.
        Eigen::Matrix3f R = best_pose.block<3, 3>(0, 0);
        float yaw = std::atan2(R(1, 0), R(0, 0));
        float yaw_deg = std::abs(yaw * 180.0f / M_PI);
        // if (yaw_deg > 30.0f && yaw_deg < 150.0f) {
        //   std::cout << "loop rejected: implausible yaw=" << yaw_deg
        //             << "° (not near 0° or 180°), score=" << score << std::endl;
        //   continue;
        // }

        // ── ICP consistency check: reject perceptual aliases ──
        // The ICP result must be geometrically consistent with the odometry guess.
        // A perceptual alias occurs when two structurally identical places (e.g. two
        // sections of the same corridor) are matched. ICP gives a low fitness score
        // because the geometry IS identical — but the true displacement is wrong.
        // Detection: the ICP result translation should not deviate far from the guess
        // translation. If guess says keyframes are ~Xm apart and ICP says ~1m apart,
        // one of them is wrong — and since ICP local minima are common, reject it.
        float guess_trans = guess.block<3, 1>(0, 3).norm();
        float icp_trans = best_pose.block<3, 1>(0, 3).norm();
        // Allow ICP to correct the guess by up to distance_thresh in translation.
        // Beyond that, ICP has jumped to a different local minimum.
        if (std::abs(icp_trans - guess_trans) > static_cast<float>(distance_thresh)) {
          std::cout << "loop rejected: ICP translation inconsistency — "
                    << "guess=" << guess_trans << "m, icp=" << icp_trans
                    << "m, diff=" << std::abs(icp_trans - guess_trans)
                    << "m > thresh=" << distance_thresh << std::endl;
          continue;
        }

        // ── Absolute ICP translation cap ──
        // A true loop closure means the robot is physically at the same spot.
        // ICP should report near-zero relative translation (< 2m).
        // Adjacent identical aisles produce ~3m+ translation (one aisle width).
        if (icp_trans > icp_trans_cap) {
          std::cout << "\033[33m[LOOP_REJECT]\033[0m ICP translation too large: "
                    << icp_trans << "m > " << icp_trans_cap << "m cap" << std::endl;
          continue;
        }

        best_score = score;
        best_matched = candidate.second;
        relative_pose = best_pose;
      } else {
        Eigen::Matrix4f guess;
        guess << 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1, 0, 0, 0, 0, 1;
        registration->align(*aligned, guess);

        double score = registration->getFitnessScore(fitness_score_max_range);
        if (!registration->hasConverged() || score > best_score) {
          continue;
        }

        best_score = score;
        best_matched = candidate.second;
        relative_pose = registration->getFinalTransformation();
      }
      // std::cout << "." << std::flush;
    }

    auto t2 = rclcpp::Clock{}.now();
    // std::cout << " done" << std::endl;matching
    // std::cout << "best_score: " << boost::format("%.3f") % best_score
    //           << "    time: " << boost::format("%.3f") % (t2 - t1).seconds() <<
    //           "[sec]"
    //           << std::endl;

    if (best_score > fitness_score_thresh) {
      return nullptr;
    }

    // std::cout << "loop found!!" << std::endl;
    // std::cout
    //     << "relpose: " << relative_pose.block<3, 1>(0, 3) << " - "
    //     << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose()
    //     << std::endl;

    last_edge_accum_distance = new_keyframe->accum_distance;

    return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose,best_score);
  }

  void print_loop_stats() const {
    int total = loops_from_geo_ + loops_from_gnn_ + loops_from_desc_;
    std::cout << "\033[1;35m[LOOP_STATS]\033[0m "
              << "Total verified: " << total
              << " | GEO: " << loops_from_geo_
              << " | GNN: " << loops_from_gnn_
              << " | DESC: " << loops_from_desc_
              << " (checked " << detect_call_count_ << " keyframes)"
              << std::endl;
  }

 private:
  std::mutex& shared_graph_mutex;

  double distance_thresh;
  double accum_distance_thresh;
  double distance_from_last_edge_thresh;

  double fitness_score_max_range;
  double icp_trans_cap;
  double fitness_score_thresh;
  double keyframe_matching_threshold;

  double last_edge_accum_distance;

  // Semantic loop proposal parameters
  bool enable_semantic_proposal_ = false;
  double semantic_proposal_thresh_ = 0.93;
  double gnn_proposal_thresh_ = 0.5;
  int semantic_proposal_max_candidates_ = 3;

  // Temporal consistency — require N consecutive keyframes to agree
  int temporal_consistency_required_ = 2;
  int pending_consecutive_count_ = 0;
  double pending_matched_accum_dist_ = -1.0;
  Loop::Ptr pending_loop_ = nullptr;
  std::string pending_source_ = "";

  // Loop source counters
  int loops_from_geo_ = 0;
  int loops_from_gnn_ = 0;
  int loops_from_desc_ = 0;
  int detect_call_count_ = 0;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace s_graphs

#endif  // LOOP_DETECTOR_HPP
