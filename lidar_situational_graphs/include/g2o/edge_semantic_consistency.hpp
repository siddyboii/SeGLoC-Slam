/*
Copyright (c) 2024
Semantic Consistency Factor for GNN-based loop closure

A soft distance constraint between two SE3 vertices based on GNN
embedding similarity. Unlike EdgeLoopClosure (which encodes a precise
6-DOF relative pose from ICP), this edge encodes only a 1-DOF distance
constraint derived from semantic similarity.

Error: e = ||t1 - t2|| - expected_distance(similarity)
*/

#ifndef EDGE_SEMANTIC_CONSISTENCY_HPP
#define EDGE_SEMANTIC_CONSISTENCY_HPP

#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <Eigen/Dense>
#include <cmath>

namespace g2o {

/**
 * @brief Semantic consistency edge: soft distance constraint from GNN similarity.
 *
 * Measurement: expected distance (double) computed from GNN cosine similarity.
 * Error: scalar — deviation of actual distance from expected distance.
 * Information: 1×1 matrix scaled by similarity confidence.
 *
 * This edge constrains only the DISTANCE between two keyframes, not their
 * relative orientation. This is intentional: GNN similarity indicates
 * "same place" but provides no heading information.
 */
class EdgeSemanticConsistency
    : public BaseBinaryEdge<1, double, VertexSE3, VertexSE3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSemanticConsistency() : BaseBinaryEdge<1, double, VertexSE3, VertexSE3>() {
    gnn_similarity_ = 0.0;
  }

  /**
   * @brief Compute error = actual_distance - expected_distance
   */
  void computeError() override {
    const VertexSE3* v1 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3* v2 = static_cast<const VertexSE3*>(_vertices[1]);

    // Actual Euclidean distance between the two keyframe positions
    Eigen::Vector3d t1 = v1->estimate().translation();
    Eigen::Vector3d t2 = v2->estimate().translation();
    double actual_dist = (t1 - t2).norm();

    // Expected distance is stored as the measurement
    double expected_dist = _measurement;

    // Error: how far off the actual distance is from semantic expectation
    _error[0] = actual_dist - expected_dist;
  }

  /**
   * @brief Analytical Jacobian for faster optimization.
   *
   * d(error)/d(v1) and d(error)/d(v2) — derivatives of the distance
   * w.r.t. the SE3 vertex parameters.
   */
  void linearizeOplus() override {
    const VertexSE3* v1 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3* v2 = static_cast<const VertexSE3*>(_vertices[1]);

    Eigen::Vector3d t1 = v1->estimate().translation();
    Eigen::Vector3d t2 = v2->estimate().translation();
    Eigen::Vector3d diff = t1 - t2;
    double dist = diff.norm();

    // Jacobian of distance w.r.t. translation
    Eigen::Vector3d dd_dt;
    if (dist > 1e-6) {
      dd_dt = diff / dist;  // unit direction vector
    } else {
      dd_dt = Eigen::Vector3d::Zero();  // at same position, gradient is zero
    }

    // J1: d(error)/d(v1) — 1×6 (3 translation + 3 rotation)
    // Distance depends only on translation, not rotation
    _jacobianOplusXi.setZero();
    _jacobianOplusXi(0, 0) = dd_dt.x();
    _jacobianOplusXi(0, 1) = dd_dt.y();
    _jacobianOplusXi(0, 2) = dd_dt.z();
    // Rotation components are zero (distance is rotation-invariant)

    // J2: d(error)/d(v2) — opposite sign
    _jacobianOplusXj.setZero();
    _jacobianOplusXj(0, 0) = -dd_dt.x();
    _jacobianOplusXj(0, 1) = -dd_dt.y();
    _jacobianOplusXj(0, 2) = -dd_dt.z();
  }

  bool read(std::istream& is) override {
    is >> _measurement >> gnn_similarity_;
    is >> _information(0, 0);
    return true;
  }

  bool write(std::ostream& os) const override {
    os << _measurement << " " << gnn_similarity_ << " ";
    os << _information(0, 0);
    return true;
  }

  // Store the GNN similarity for logging/debugging
  void setGNNSimilarity(double sim) { gnn_similarity_ = sim; }
  double gnnSimilarity() const { return gnn_similarity_; }

  /**
   * @brief Convert GNN cosine similarity to expected Euclidean distance.
   *
   * Mapping: expected_dist = max_dist * (1 - sim)^gamma
   *
   * @param sim        Cosine similarity in [0, 1]
   * @param max_dist   Maximum expected distance (for sim ≈ 0)
   * @param gamma      Curve shape parameter (>1 = steeper near sim=1)
   * @return Expected distance in meters
   */
  static double similarityToDistance(double sim,
                                     double max_dist = 15.0,
                                     double gamma = 1.5) {
    sim = std::max(0.0, std::min(1.0, sim));  // clamp
    return max_dist * std::pow(1.0 - sim, gamma);
  }

  /**
   * @brief Compute information weight from similarity.
   *
   * Higher similarity → more confident → higher information weight.
   *
   * @param sim          Cosine similarity
   * @param base_weight  Base information weight
   * @return Information matrix value (1×1)
   */
  static double similarityToInformation(double sim, double base_weight = 1.0) {
    // Quadratic scaling: confident edges have 4x the weight of borderline ones
    return base_weight * sim * sim;
  }

 private:
  double gnn_similarity_;
};

}  // namespace g2o

#endif  // EDGE_SEMANTIC_CONSISTENCY_HPP
