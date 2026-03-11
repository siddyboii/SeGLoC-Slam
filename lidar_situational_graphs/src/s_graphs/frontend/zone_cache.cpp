#include "s_graphs/frontend/zone_cache.hpp"
#include <algorithm>

namespace s_graphs {

ZoneCache::ZoneCache() {}
ZoneCache::~ZoneCache() {}

void ZoneCache::init(rclcpp::Node::SharedPtr node, const std::string& topic) {
  node_ = node;
  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.transient_local();
  sub_ = node_->create_subscription<situational_graphs_msgs::msg::Zone>(
      topic, qos,
      std::bind(&ZoneCache::zone_callback, this, std::placeholders::_1));
  RCLCPP_INFO(node_->get_logger(), "[ZoneCache] Subscribed to '%s'", topic.c_str());
}

void ZoneCache::zone_callback(
    const situational_graphs_msgs::msg::Zone::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mtx_);
  int zid = static_cast<int>(msg->zone_id);
  uint64_t ver = static_cast<uint64_t>(msg->version);
  double conf = static_cast<double>(msg->confidence);

  // DELETE action → remove zone
  if (static_cast<int>(msg->action) == 2) {
    auto it = zone_to_kfs_.find(zid);
    if (it != zone_to_kfs_.end()) {
      for (int kf : it->second) {
        kf_to_zone_.erase(kf);
      }
      zone_to_kfs_.erase(it);
    }
    zone_support_planes_.erase(zid);
    zone_confidence_.erase(zid);
    zone_version_.erase(zid);
    zone_floor_.erase(zid);
    zone_labels_.erase(zid);
    RCLCPP_INFO(node_->get_logger(), "[ZoneCache] Deleted zone %d", zid);
    return;
  }

  // Skip older versions (but update confidence)
  auto v_it = zone_version_.find(zid);
  if (v_it != zone_version_.end() && v_it->second >= ver) {
    zone_confidence_[zid] = conf;
    return;
  }

  // Update keyframe mappings
  std::vector<int> kfs;
  kfs.reserve(msg->keyframe_ids.size());
  for (auto k : msg->keyframe_ids) kfs.push_back(static_cast<int>(k));

  // Remove old kf→zone mapping for this zone
  auto old_it = zone_to_kfs_.find(zid);
  if (old_it != zone_to_kfs_.end()) {
    for (int old_kf : old_it->second) {
      kf_to_zone_.erase(old_kf);
    }
  }

  zone_to_kfs_[zid] = kfs;
  for (int kf : kfs) kf_to_zone_[kf] = zid;

  // Parse support plane ids
  std::set<int> sup;
  for (const auto& p : msg->supporting_planes) {
    if (p.id != 0) {
      sup.insert(static_cast<int>(p.id));
    }
  }
  zone_support_planes_[zid] = sup;

  // Store metadata
  zone_confidence_[zid] = conf;
  zone_version_[zid] = ver;
  zone_floor_[zid] = static_cast<int>(msg->floor_id);

  std::vector<std::string> labels(msg->top_labels.begin(), msg->top_labels.end());
  zone_labels_[zid] = labels;

  int action = static_cast<int>(msg->action);
  RCLCPP_INFO(node_->get_logger(),
      "[ZoneCache] Zone %d %s: %zu kfs, %zu planes, conf=%.2f, labels=%zu",
      zid, (action == 0 ? "CREATED" : "UPDATED"),
      kfs.size(), sup.size(), conf, labels.size());
}

int ZoneCache::get_zone_for_keyframe(int keyframe_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = kf_to_zone_.find(keyframe_id);
  if (it == kf_to_zone_.end()) return -1;
  return it->second;
}

std::vector<int> ZoneCache::get_keyframes_in_zone(int zone_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = zone_to_kfs_.find(zone_id);
  if (it == zone_to_kfs_.end()) return {};
  return it->second;
}

std::set<int> ZoneCache::get_support_plane_ids(int zone_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = zone_support_planes_.find(zone_id);
  if (it == zone_support_planes_.end()) return {};
  return it->second;
}

double ZoneCache::get_zone_confidence(int zone_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = zone_confidence_.find(zone_id);
  if (it == zone_confidence_.end()) return 0.0;
  return it->second;
}

bool ZoneCache::has_zone(int zone_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return zone_to_kfs_.find(zone_id) != zone_to_kfs_.end();
}

int ZoneCache::get_zone_floor(int zone_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = zone_floor_.find(zone_id);
  if (it == zone_floor_.end()) return -1;
  return it->second;
}

std::vector<std::string> ZoneCache::get_zone_labels(int zone_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = zone_labels_.find(zone_id);
  if (it == zone_labels_.end()) return {};
  return it->second;
}

}  // namespace s_graphs
