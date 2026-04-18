#include "g2o/edge_zone_zone.hpp"

#include <g2o/core/optimizable_graph.h>

namespace g2o {

EdgeZoneZone::EdgeZoneZone() = default;
EdgeZoneZone::~EdgeZoneZone() = default;

void EdgeZoneZone::computeError() {
  const auto* z_i = static_cast<const VertexZone*>(_vertices[0]);
  const auto* z_j = static_cast<const VertexZone*>(_vertices[1]);

  const Eigen::Isometry3d& T_wi = z_i->estimate();
  const Eigen::Isometry3d& T_wj = z_j->estimate();

  const Eigen::Isometry3d T_ij_pred = T_wi.inverse() * T_wj;
  const Eigen::Isometry3d T_err = _measurement.inverse() * T_ij_pred;

  _error = g2o::internal::toVectorMQT(T_err);
}

void EdgeZoneZone::linearizeOplus() {
  BaseBinaryEdge<6, Eigen::Isometry3d, VertexZone, VertexZone>::linearizeOplus();
}

double EdgeZoneZone::initialEstimatePossible(const OptimizableGraph::VertexSet& from,
                                             OptimizableGraph::Vertex* to) {
  (void)from;
  (void)to;
  return 1.0;
}

void EdgeZoneZone::initialEstimate(const OptimizableGraph::VertexSet& from,
                                   OptimizableGraph::Vertex* to) {
  (void)from;

  auto* z_i = static_cast<VertexZone*>(_vertices[0]);
  auto* z_j = static_cast<VertexZone*>(_vertices[1]);

  if (!z_i || !z_j || !to) return;

  if (to == z_j) {
    const Eigen::Isometry3d T_wi = z_i->estimate();
    z_j->setEstimate(T_wi * _measurement);
  } else if (to == z_i) {
    const Eigen::Isometry3d T_wj = z_j->estimate();
    z_i->setEstimate(T_wj * _measurement.inverse());
  }
}

bool EdgeZoneZone::read(std::istream& is) {
  (void)is;
  return true;
}

bool EdgeZoneZone::write(std::ostream& os) const {
  (void)os;
  return true;
}

}  // namespace g2o