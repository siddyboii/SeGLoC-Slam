#include "g2o/edge_zone_keyframe.hpp"

#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/stuff/misc.h>
#include <iostream>
#include <g2o/core/eigen_types.h>

namespace g2o {
    
EdgeZoneKeyframe::EdgeZoneKeyframe() = default;
EdgeZoneKeyframe::~EdgeZoneKeyframe() = default;

void EdgeZoneKeyframe::computeError() {
  const auto* v_kf = static_cast<const VertexSE3*>(_vertices[0]);
  const auto* v_zone = static_cast<const VertexZone*>(_vertices[1]);

  const Eigen::Isometry3d& T_wk = v_kf->estimate();
  const Eigen::Isometry3d& T_wz = v_zone->estimate();

  // Predicted relative transform from keyframe to zone
  const Eigen::Isometry3d T_kz_pred = T_wk.inverse() * T_wz;

  // Measurement is also a keyframe-to-zone relative pose
  const Eigen::Isometry3d T_kz_meas = _measurement;

  // Error transform: measurement^{-1} * prediction
  const Eigen::Isometry3d T_err = T_kz_meas.inverse() * T_kz_pred;

  // Convert to 6D vector [dx dy dz droll dpitch dyaw]
  _error = g2o::internal::toVectorMQT(T_err);
}

void EdgeZoneKeyframe::linearizeOplus() {
  // Use the default numerical/analytical-style linearization path from g2o.
  // For a first correct implementation, this is acceptable and stable.
  BaseBinaryEdge<6, Eigen::Isometry3d, VertexSE3, VertexZone>::linearizeOplus();
}

// bool EdgeZoneKeyframe::initialEstimatePossible(const OptimizableGraph::VertexSet& from,
//                                                OptimizableGraph::Vertex* to) {
//   // If one vertex is known, the other can be initialized from the measurement.
//   // This is a soft pose-edge, so initial estimate is possible in either direction.
//   return true;
// }
double EdgeZoneKeyframe::initialEstimatePossible(const OptimizableGraph::VertexSet& from,
                                                 OptimizableGraph::Vertex* to) {
  (void)from;
  (void)to;
  return 1.0;
}
void EdgeZoneKeyframe::initialEstimate(const OptimizableGraph::VertexSet& from,
                                       OptimizableGraph::Vertex* to) {
  (void)from;
  auto* v_kf = static_cast<VertexSE3*>(_vertices[0]);
  auto* v_zone = static_cast<VertexZone*>(_vertices[1]);

  if (!v_kf || !v_zone || !to) {
    return;
  }

  // If the zone vertex needs an initial pose, estimate it from the keyframe pose
  // and the stored measurement:
  //
  //   T_wz = T_wk * T_kz_meas
  //
  // where T_kz_meas = _measurement
  //
  // If the target is the keyframe, we can similarly infer from zone pose.
  if (to == v_zone) {
    const Eigen::Isometry3d T_wk = v_kf->estimate();
    const Eigen::Isometry3d T_kz_meas = _measurement;
    v_zone->setEstimate(T_wk * T_kz_meas);
  } else if (to == v_kf) {
    const Eigen::Isometry3d T_wz = v_zone->estimate();
    const Eigen::Isometry3d T_kz_meas = _measurement;
    v_kf->setEstimate(T_wz * T_kz_meas.inverse());
  }
}
bool EdgeZoneKeyframe::read(std::istream& is) {
    (void)is;
  // Minimal stub for now
  return true;
}

bool EdgeZoneKeyframe::write(std::ostream& os) const {
    (void)os;
  // Minimal stub for now
  return os.good();
}

}  // namespace g2o