// #ifndef G2O_VERTEX_ZONE
// #define G2O_VERTEX_ZONE

// #include <g2o/types/slam3d/g2o_types_slam3d_api.h>
// #include <g2o/types/slam3d/isometry3d_mappings.h>

// #include "vertex_room.hpp"

// namespace g2o {

// /**
//  * @brief VertexZone
//  * Zone is a place-level landmark / latent summary node.
//  * It reuses the same SE3 pose representation as VertexRoom.
//  */
// class G2O_TYPES_SLAM3D_API VertexZone : public VertexRoom {
//  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//   VertexZone() {
//     _numOplusCalls = 0;
//     setToOriginImpl();
//     updateCache();
//   }
// };

// }  // namespace g2o

// #endif  // G2O_VERTEX_ZONE
#ifndef G2O_VERTEX_ZONE_HPP
#define G2O_VERTEX_ZONE_HPP

#include <g2o/core/base_vertex.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {

/**
 * @brief VertexZone
 *
 * A dedicated pose vertex for zone centroids.
 * Stores an SE3 pose estimate as Eigen::Isometry3d.
 */
class VertexZone : public BaseVertex<6, Eigen::Isometry3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexZone() {
    setToOriginImpl();
  }

  void setToOriginImpl() override {
    _estimate = Eigen::Isometry3d::Identity();
  }

  void oplusImpl(const double* update) override {
    Eigen::Matrix<double, 6, 1> v;
    for (int i = 0; i < 6; ++i) {
      v[i] = update[i];
    }

    Eigen::Isometry3d up = Eigen::Isometry3d::Identity();
    up.translation() = v.head<3>();

    Eigen::Vector3d omega = v.tail<3>();
    double theta = omega.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    if (theta > 1e-12) {
      Eigen::Vector3d axis = omega / theta;
      R = Eigen::AngleAxisd(theta, axis).toRotationMatrix();
    }

    up.linear() = R;
    _estimate = up * _estimate;
  }

  bool read(std::istream& is) override { return true; }
  bool write(std::ostream& os) const override { return true; }
};

}  // namespace g2o

#endif  // G2O_VERTEX_ZONE_HPP