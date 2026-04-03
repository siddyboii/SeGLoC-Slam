#ifndef G2O_EDGE_ZONE_KEYFRAME_HPP
#define G2O_EDGE_ZONE_KEYFRAME_HPP

#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/core/eigen_types.h>
#include <g2o/core/base_vertex.h>

#include "vertex_zone.hpp"

namespace g2o {

/**
 * @brief EdgeZoneKeyframe
 *
 * Soft factor between a keyframe pose and a zone pose.
 *
 * Vertex 0: keyframe pose (VertexSE3)
 * Vertex 1: zone pose      (VertexZone)
 *
 * Measurement meaning:
 *   T_kz = T_wk^-1 * T_wz
 *
 * Error:
 *   e = Log( T_kz^-1 * (T_wk^-1 * T_wz) )
 *
 * The edge is a soft constraint, intended to keep keyframes in the same
 * semantic zone near that zone's centroid pose.
 */
class EdgeZoneKeyframe : public BaseBinaryEdge<6, Eigen::Isometry3d, VertexSE3, VertexZone> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeZoneKeyframe();
  ~EdgeZoneKeyframe();

  /**
   * @brief Compute the residual/error of the factor.
   */
  void computeError() override;

  /**
   * @brief Linearize the edge.
   *
   * BaseBinaryEdge already provides a default linearization path in many cases,
   * but we keep this virtual override explicit for clarity and compatibility.
   */
  void linearizeOplus() override;

  /**
   * @brief This edge always supports an initial estimate.
   */
//   bool initialEstimatePossible(const OptimizableGraph::VertexSet& from,
//                                OptimizableGraph::Vertex* to) override;
  double initialEstimatePossible(const OptimizableGraph::VertexSet& from,
                                 OptimizableGraph::Vertex* to) override;

  /**
   * @brief Initialize the estimate of the zone vertex from the keyframe pose and measurement.
   */
  void initialEstimate(const OptimizableGraph::VertexSet& from,
                       OptimizableGraph::Vertex* to) override;

  bool read(std::istream& is) override;
  bool write(std::ostream& os) const override;
};

}  // namespace g2o

#endif  // G2O_EDGE_ZONE_KEYFRAME_HPP