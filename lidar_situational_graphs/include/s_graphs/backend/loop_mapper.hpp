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

#ifndef LOOP_MAPPER_HPP
#define LOOP_MAPPER_HPP

#include <fstream>
#include <g2o/edge_loop_closure.hpp>
#include <s_graphs/common/information_matrix_calculator.hpp>
#include <s_graphs/common/optimization_data.hpp>
#include <s_graphs/frontend/loop_detector.hpp>
#include <s_graphs/frontend/zone_cache.hpp>

namespace s_graphs {

class LoopMapper {
 public:
  LoopMapper(const rclcpp::Node::SharedPtr node, std::mutex& graph_mutex);
  ~LoopMapper();

 public:
  void add_loops(const std::shared_ptr<GraphSLAM>& covisibility_graph,
                 const std::vector<Loop::Ptr>& loops);

  /// Set the zone cache for zone-based loop closure prefiltering
  void set_zone_cache(std::shared_ptr<ZoneCache> zone_cache);

 private:
  void set_data(g2o::VertexSE3* keyframe_node);
  bool get_floor_data(g2o::VertexSE3* keyframe_node);
    /**
   * @brief Compute cosine similarity between two CLIP embeddings
   * @param emb1 First embedding (512-dim)
   * @param emb2 Second embedding (512-dim)
   * @return Similarity score in range [0, 1]
   */
  double computeClipSimilarity(const std::vector<float>& emb1, 
                               const std::vector<float>& emb2);

/**
   * @brief Convert semantic confidence to information matrix scaling factor
   * @param clip_similarity CLIP cosine similarity [0, 1]
   * @return Scaling factor alpha for information matrix
   */
  double confidenceToAlpha(double clip_similarity);

  /**
   * @brief Compute object overlap between two keyframes (Jaccard index)
   * @param objects1 First keyframe's detected objects
   * @param objects2 Second keyframe's detected objects
   * @return Overlap score in range [0, 1]
   */
  double computeObjectOverlap(const std::vector<std::string>& objects1,
                              const std::vector<std::string>& objects2);

 private:
  std::mutex& shared_graph_mutex;
  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<InformationMatrixCalculator> inf_calclator;
  double fitness_score_thresh;  // Threshold for ICP fitness score (for weighting logic)

  // Zone-based prefiltering
  std::shared_ptr<ZoneCache> zone_cache_;
  bool use_zone_prefilter_ = true;
  double zone_confidence_threshold_ = 0.3;

  // CSV metrics logger
  std::ofstream loop_metrics_csv_;
};

}  // namespace s_graphs

#endif  // LOOP_MAPPER_HPP