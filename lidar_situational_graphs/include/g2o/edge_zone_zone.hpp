#ifndef G2O_EDGE_ZONE_ZONE_HPP
#define G2O_EDGE_ZONE_ZONE_HPP

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include "g2o/vertex_zone.hpp"

namespace g2o {

/**
 * @brief Soft relation between two semantic zones.
 *
 * Vertex 0: Zone i
 * Vertex 1: Zone j
 *
 * Measurement is the relative transform between the two zone centroids.
 */
class EdgeZoneZone
    : public BaseBinaryEdge<6, Eigen::Isometry3d, VertexZone, VertexZone> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeZoneZone();
  ~EdgeZoneZone() override;

  void computeError() override;
  void linearizeOplus() override;

  double initialEstimatePossible(const OptimizableGraph::VertexSet& from,
                                 OptimizableGraph::Vertex* to) override;

  void initialEstimate(const OptimizableGraph::VertexSet& from,
                       OptimizableGraph::Vertex* to) override;

  bool read(std::istream& is) override;
  bool write(std::ostream& os) const override;
};

}  // namespace g2o

#endif  // G2O_EDGE_ZONE_ZONE_HPP