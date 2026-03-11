#pragma once
#include <unordered_map>
#include <vector>
#include <set>
#include <mutex>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include "situational_graphs_msgs/msg/zone.hpp"

namespace s_graphs {

class ZoneCache {
 public:
  ZoneCache();
  ~ZoneCache();

  /// Initialize ROS subscription (call once from the node that owns the cache)
  /// @param node shared pointer used to create subscription
  /// @param topic topic name (default "zones")
  void init(rclcpp::Node::SharedPtr node, const std::string& topic = "zones");

  // ─── Query functions (thread-safe, const) ───

  /// Returns zone id for a keyframe id, or -1 if unknown
  int get_zone_for_keyframe(int keyframe_id) const;

  /// Returns keyframe ids in zone (empty if none)
  std::vector<int> get_keyframes_in_zone(int zone_id) const;

  /// Returns support plane ids for zone (empty if none)
  std::set<int> get_support_plane_ids(int zone_id) const;

  /// Returns zone confidence (0.0 if unknown)
  double get_zone_confidence(int zone_id) const;

  /// Returns true if zone exists
  bool has_zone(int zone_id) const;

  /// Returns floor_id for a zone (-1 if unknown)
  int get_zone_floor(int zone_id) const;

  /// Returns top semantic labels for a zone
  std::vector<std::string> get_zone_labels(int zone_id) const;

 private:
  void zone_callback(const situational_graphs_msgs::msg::Zone::SharedPtr msg);

  mutable std::mutex mtx_;
  std::unordered_map<int, std::vector<int>> zone_to_kfs_;
  std::unordered_map<int, int> kf_to_zone_;
  std::unordered_map<int, std::set<int>> zone_support_planes_;
  std::unordered_map<int, double> zone_confidence_;
  std::unordered_map<int, uint64_t> zone_version_;
  std::unordered_map<int, int> zone_floor_;
  std::unordered_map<int, std::vector<std::string>> zone_labels_;

  rclcpp::Subscription<situational_graphs_msgs::msg::Zone>::SharedPtr sub_;
  rclcpp::Node::SharedPtr node_;
};

}  // namespace s_graphs
