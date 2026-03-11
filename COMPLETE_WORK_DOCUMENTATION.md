# Complete Work Documentation: Semantic Loop Closure for S-Graphs with Zone-Based Prefiltering

**Author:** Subhanshu  
**Date:** February 2026  
**Status:** Implementation Complete, Testing Pending  
**Framework:** ROS2 Humble, CUDA 12.1, Docker

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Implementation Details](#implementation-details)
5. [File Changes & Modifications](#file-changes--modifications)
---

## Executive Summary

### What Was Built

A **comprehensive semantic loop closure system** for S-Graphs SLAM that integrates:
- YOLO World object detection (open-vocabulary)
- CLIP feature embeddings (semantic understanding)
- DynaTrack optical flow (dynamic scene detection)
- Zone-based semantic prefiltering (novel spatial reasoning)
- Multi-factor information matrix weighting

### Key Achievement

Successfully integrated **4 independent systems** into a coherent pipeline that weights loop closures not just by geometry, but by:
- Semantic similarity (CLIP embeddings)
- Object consistency (detected objects)
- Image quality (IQA scores)
- Scene dynamicity (moving objects)
- Semantic zone coherence (structural + semantic grouping)

### Current State

✅ **Fully implemented and building**  
✅ **Docker-based reproducible setup**  
✅ **All code compiles with zero errors**  

---

## Problem Statement

### The Core Issue: False Loop Closures in SLAM

Modern LiDAR SLAM systems like S-Graphs detect loop closures purely through **geometric matching** (ICP):

```
Two point clouds match geometrically? → Accept loop closure
↓
Problem: Similar-looking rooms trap the system
↓
Result: CATASTROPHIC map failures
```

**Real-world example:**
```
Two kitchens in same building
Both have: 
  - Similar counter geometry
  - Similar point cloud density
  - Similar ICP fitness scores (0.95+)

Geometric SLAM accepts loop closure ✓
Reality: WRONG! Different floors, different kitchens ✗
Result: Map collapses, trajectory corrupted forever
```

### Why Semantic Information Helps

Human spatial understanding uses:
1. **Object context**: "I see a bed, stove, toilet → different rooms"
2. **Semantic grouping**: "Bedroom on 2nd floor, kitchen on 1st floor"
3. **Structural adjacency**: "Kitchen shares wall with dining room"
4. **Scene dynamics**: "Moving person = dynamic scene = less reliable point cloud"

**S-Graphs before this work:** Only used geometry, ignored semantics.

---

## Solution Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LiDAR Points    Camera Image    Odometry                   │
│      ↓               ↓               ↓                       │
│   S-Graphs Node (existing)                                  │
│      ↓                                                       │
│  KeyFrame Created                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              SEMANTIC EXTRACTION (NEW)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │ YOLO World   │  │ CLIP Model  │  │ Image Quality│       │
│  │ Detector     │  │ Embeddings  │  │ Assessment   │       │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ↓                                  │
│            KeyframeSemantic Message                         │
│  {objects, confidences, embeddings, quality_score}         │
│                           ↓                                 │
│                   Publish on ROS topic                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         DYNAMIC SCENE DETECTION (NEW - DynaTrack)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BEV Optical Flow → Cluster Moving Pixels                   │
│         ↓                                                    │
│  Compute Scene Dynamicity [0, 1]                            │
│         ↓                                                    │
│  DynamicObjects Message (num_clusters, scene_dynamicity)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          ZONE CREATION (NEW - zones_node.py)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Subscribe to:                                              │
│    - KeyframeSemantic (objects + CLIP embeddings)           │
│    - RoomsData (geometric rooms)                            │
│    - GraphKeyframes (canonical poses)                       │
│                                                              │
│  Processing:                                                │
│    1. Match semantics → keyframes (time/spatial)            │
│    2. Aggregate objects per room                            │
│    3. Compute room signature (Dirichlet smoothing)          │
│    4. Merge adjacent rooms into zones (cosine sim > 0.65)   │
│    5. Publish Zone messages with:                           │
│       - zone_id, floor_id, centroid                         │
│       - keyframe_ids, room_ids                              │
│       - support_plane_ids (shared walls)                    │
│       - top_labels, top_label_confidences                   │
│       - zone_confidence, version                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│      ZONE CACHE & LOOP CLOSURE WEIGHTING (NEW)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ZoneCache subscribes to Zone messages                      │
│       ↓                                                      │
│  Build lookup tables:                                       │
│    - zone_to_kfs[z_id] → [kf1, kf2, ...]                   │
│    - kf_to_zone[kf_id] → z_id                              │
│    - zone_support_planes[z_id] → {plane1, plane2, ...}     │
│    - zone_confidence[z_id] → 0.0-1.0                       │
│                                                              │
│  LoopMapper applies multi-factor weighting:                 │
│                                                              │
│    semantic_alpha = 1.0                                     │
│    × geometry_semantic_interaction (ICP reliability)        │
│    × quality_gate (image quality [0,1])                     │
│    × dyn_factor (scene dynamicity penalty)                  │
│    × zone_factor (zone coherence)                           │
│                                                              │
│    information_matrix *= semantic_alpha                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
                   POSE GRAPH OPTIMIZATION
```

### Multi-Factor Weighting Pipeline

```
LOOP CLOSURE DETECTION: Two keyframes match geometrically
                              ↓
        ┌───────────────────────────────────────────┐
        │     Compute semantic_alpha = 1.0          │
        └───────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ FACTOR 1: CLIP Similarity & Object Overlap         │
    ├─────────────────────────────────────────────────────┤
    │ • Cosine similarity of 512-dim CLIP embeddings     │
    │ • Jaccard index of detected objects                │
    │ • Result: clip_sim ∈ [0, 1]                        │
    │ • Alpha: confidenceToAlpha(clip_sim)               │
    │   - clip_sim >= 0.7: boost (up to 2.0x)            │
    │   - clip_sim < 0.7: reduce (down to 0.1x)          │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ FACTOR 2: Geometry-Semantic Interaction            │
    ├─────────────────────────────────────────────────────┤
    │ • Compute geometric_reliability = 1 - (ICP_fitness  │
    │                                    / fitness_thresh) │
    │ • When geometry excellent: dampen semantic effects  │
    │ • When geometry poor: amplify semantic effects      │
    │ • Interaction: alpha = 1.0 + (alpha - 1.0) *      │
    │               semantic_influence                    │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ FACTOR 3: Image Quality Gating (IQA)               │
    ├─────────────────────────────────────────────────────┤
    │ • IQA Python node scores image [0, 1]              │
    │ • min_quality = min(iqa_score_kf1, iqa_score_kf2)  │
    │ • quality_gate = min_quality                       │
    │ • When poor quality: regress alpha toward 1.0      │
    │ • alpha = quality_gate * alpha + (1-quality_gate)*1│
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ FACTOR 4: Scene Dynamicity (DynaTrack)             │
    ├─────────────────────────────────────────────────────┤
    │ • From DynaTrack: scene_dynamicity ∈ [0, 1]        │
    │ • High dynamicity = moving objects = unreliable     │
    │ • dyn_factor = 1.0 - max_dyn * 0.3                 │
    │ • Range: 1.0 (static) → 0.7 (fully dynamic)        │
    │ • semantic_alpha *= dyn_factor                      │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ FACTOR 5: Zone Coherence (NOVEL)                   │
    ├─────────────────────────────────────────────────────┤
    │ • Check zones of both keyframes (z1, z2)            │
    │ • CASE 1: z1 == z2 (same zone)                      │
    │   - zone_factor = 1.0 + zone_conf * 0.2             │
    │   - Boost up to 20%                                 │
    │ • CASE 2: z1 ≠ z2, shared support planes            │
    │   - zone_factor = 0.9                               │
    │   - Mild 10% penalty (adjacent rooms OK)            │
    │ • CASE 3: z1 ≠ z2, NO shared planes                │
    │   - zone_factor = max(0.3, 1.0 - min_conf * 0.5)   │
    │   - Up to 50% penalty (distant rooms)               │
    │ • semantic_alpha *= zone_factor                     │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ FINAL: Scale Information Matrix                     │
    ├─────────────────────────────────────────────────────┤
    │ information_matrix *= semantic_alpha                │
    │                                                      │
    │ Result:                                              │
    │ • High semantic agreement → more informative edge   │
    │ • Low semantic agreement → less trusted edge        │
    │ • Optimizer gives appropriate weight to constraint  │
    └─────────────────────────────────────────────────────┘
```

### Zone Label Computation Flow

```
YOLO Detection
  "bed" (0.95), "pillow" (0.88), "wardrobe" (0.91)
         ↓
OBJECT_TO_LABEL Mapping (hardcoded rules)
  bed → {bedroom: 1.0}
  pillow → {bedroom: 0.8}
  wardrobe → {bedroom: 0.6}
         ↓
Aggregate with Confidences
  bedroom: 0.95*1.0 + 0.88*0.8 + 0.91*0.6 = 2.174
  total_weight: 0.95 + 0.88 + 0.91 = 2.74
         ↓
Dirichlet Smoothing (add pseudo-count α₀=0.5)
  V = 1 (number of room types)
  denominator = 2.74 + 0.5*1 = 3.24
  bedroom_prob = (2.174 + 0.5) / 3.24 = 0.826
         ↓
Room Signature: {"bedroom": 0.826}
         ↓
Merge with Similar Adjacent Rooms (cosine sim > 0.65)
  If nearby room also has bedroom signature → merge into zone
  If different signature (e.g., bathroom) → create separate zone
         ↓
Zone Message Published
  top_labels: ["bedroom"]
  top_label_confidences: [0.826]
  zone_confidence: 0.85 (system's certainty about zone)
```

---

## Implementation Details

### 1. KeyframeSemantic Message Publishing

**File Modified:** `lidar_situational_graphs/src/s_graphs/common/s_graphs.cpp`

**What it does:**
- Runs YOLO World on camera image → detects objects
- Extracts CLIP embeddings for detected regions
- Publishes KeyframeSemantic ROS message with:
  - Detected object names + confidence scores
  - 512-dimensional CLIP embeddings
  - Image quality score (IQA)
  - Keyframe metadata (ID, floor, timestamp)

**Key functions:**
```cpp
void SGraphsNode::processSemantic(const cv::Mat& image, 
                                  const std_msgs::msg::Header& header,
                                  KeyFrame::Ptr keyframe)

void SGraphsNode::publishDetectionVisualization(const cv::Mat& image,
                                               const std::vector<Detection>& detections,
                                               const std_msgs::msg::Header& header)
```

**Code excerpt:**
```cpp
// In processSemantic()
if (enable_yolo_ && frame_count_ % process_every_n_frames_ == 0) {
    auto detections = yolo_detector_->detect(image);
    for (const auto& det : detections) {
        if (det.confidence >= confidence_threshold_) {
            // Extract CLIP embedding for this detection
            auto embedding = clip_extractor_->extract(image, det.bbox);
            // Publish to ROS
        }
    }
}
```

---

### 2. DynaTrack Integration (Scene Dynamicity)

**Package:** `tracking_OF` (external, integrated)

**What it does:**
- Computes BEV optical flow from LiDAR point clouds
- Clusters moving pixels → identifies dynamic objects
- Produces per-frame dynamicity score [0, 1]

**Files Modified:**
- `tracking_OF/src/of_track.cpp` — Fixed noise issues:
  - `min_flow_magnitude_`: 0.5 → 1.5 (ignore tiny movements)
  - `min_cluster_size_`: 10 → 15 (require larger clusters)
  - `final_mask`: Fixed to use actual magnitude from masked_flow
  - Dynamicity scaling: `ratio * 5.0f` (20% dynamic = 1.0 score)

**Callback in s_graphs:**
```cpp
void SGraphsNode::dynamic_objects_callback(
    const situational_graphs_msgs::msg::DynamicObjects::SharedPtr dyn_msg) {
    {
        std::lock_guard<std::mutex> lock(dynamicity_mutex_);
        latest_scene_dynamicity_ = dyn_msg->scene_dynamicity;
        latest_num_dynamic_clusters_ = dyn_msg->num_dynamic_clusters;
        dynamicity_received_ = true;
    }
}
```

**Stored in keyframes:**
```cpp
keyframe->scene_dynamicity = latest_scene_dynamicity_;
keyframe->num_dynamic_clusters = latest_num_dynamic_clusters_;
```

---

### 3. Zone Creation (zones_node.py)

**File:** `lidar_situational_graphs/scripts/zones_node.py` (818 lines)

**What it does:**
- Subscribes to KeyframeSemantic, RoomsData, GraphKeyframes
- Matches semantics to keyframes (time-first, then spatial)
- Aggregates objects per room
- Computes room signatures using Dirichlet smoothing
- Merges adjacent rooms into semantic zones
- Publishes Zone messages

**Key Classes:**
```python
class ZonesNode(Node):
    def cb_kf_sem(self, msg):
        # Match semantic to nearest keyframe
        # Aggregate detected objects
        
    def cb_roomsdata(self, msg):
        # Extract room geometry + planes
        # Associate keyframes
        # Compute room signature
        # Try merge into zone
        
    def compute_room_signature(self, kf_list):
        # Aggregate objects with Dirichlet smoothing
        # Return probability over room types
        
    def try_merge_room_into_zone(self, room_id, room_sig):
        # Check adjacency + semantic similarity
        # Merge if cosine_sim > 0.65
        
    def publish_zone(self, zid, action=0):
        # Publish Zone message with all metadata
```

**OBJECT_TO_LABEL Mapping:**
```python
OBJECT_TO_LABEL = {
    'bed': {'bedroom': 1.0},
    'toilet': {'bathroom': 1.0},
    'stove': {'kitchen': 1.0},
    'desk': {'office': 0.9},
    'chair': {'office': 0.4, 'dining': 0.4},
    'shelf': {'storage': 0.8},
    # ... 13 more entries
}
```

**Message Publishing:**
```cpp
// Zone message fields
zone_id: int32
floor_id: int32
centroid: geometry_msgs/Pose
polygon: geometry_msgs/Polygon
room_ids: int32[]
keyframe_ids: int32[]
top_labels: string[]
top_label_confidences: float32[]
confidence: float32
supporting_planes: PlaneData[]
supporting_wall_ids: int32[]
action: uint8 (0=CREATE, 1=UPDATE, 2=DELETE)
version: uint64
```

---

### 4. ZoneCache (C++ Subscriber)

**File:** `lidar_situational_graphs/include/s_graphs/frontend/zone_cache.hpp`
**File:** `lidar_situational_graphs/src/s_graphs/frontend/zone_cache.cpp`

**What it does:**
- Subscribes to Zone messages
- Maintains thread-safe lookup tables:
  - `zone_to_kfs_[z_id]` → keyframe list
  - `kf_to_zone_[kf_id]` → zone ID
  - `zone_support_planes_[z_id]` → plane IDs
  - `zone_confidence_[z_id]` → confidence [0, 1]
  - `zone_floor_[z_id]` → floor level
  - `zone_labels_[z_id]` → top semantic labels

**Public API:**
```cpp
int get_zone_for_keyframe(int keyframe_id);
std::vector<int> get_keyframes_in_zone(int zone_id);
std::set<int> get_support_plane_ids(int zone_id);
double get_zone_confidence(int zone_id);
bool has_zone(int zone_id);
int get_zone_floor(int zone_id);
std::vector<std::string> get_zone_labels(int zone_id);
```

**Thread-safety:**
```cpp
mutable std::mutex mtx_;
// All queries use: std::lock_guard<std::mutex> lock(mtx_);
```

---

### 5. LoopMapper Integration

**Files Modified:**
- `loop_mapper.hpp` — Added zone_cache member, set_zone_cache() method
- `loop_mapper.cpp` — Implemented zone-based weighting factor

**Zone Weighting Logic:**
```cpp
if (zone_cache_ && use_zone_prefilter_) {
    int z1 = zone_cache_->get_zone_for_keyframe(loop->key1->node->id());
    int z2 = zone_cache_->get_zone_for_keyframe(loop->key2->node->id());
    
    if (z1 == z2) {
        // Same zone → boost
        double zone_conf = zone_cache_->get_zone_confidence(z1);
        zone_factor = 1.0 + zone_conf * 0.2;  // up to 20% boost
    } else if (z1 >= 0 && z2 >= 0) {
        // Different zones
        auto planes1 = zone_cache_->get_support_plane_ids(z1);
        auto planes2 = zone_cache_->get_support_plane_ids(z2);
        
        int shared_planes = 0;
        for (int pid : planes1) {
            if (planes2.count(pid)) shared_planes++;
        }
        
        if (shared_planes > 0) {
            // Adjacent zones → mild penalty
            zone_factor = 0.9;
        } else {
            // Distant zones → strong penalty
            double conf1 = zone_cache_->get_zone_confidence(z1);
            double conf2 = zone_cache_->get_zone_confidence(z2);
            double min_conf = std::min(conf1, conf2);
            zone_factor = 1.0 - min_conf * 0.5;
            zone_factor = std::max(0.3, zone_factor);
        }
    }
    
    semantic_alpha *= zone_factor;
}
```

---

## File Changes & Modifications

### New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `situational_graphs_msgs/msg/Zone.msg` | Zone message definition | 17 |
| `zone_cache.hpp` | Zone lookup cache header | 62 |
| `zone_cache.cpp` | Zone lookup cache implementation | 140 |

### Modified Files

| File | Changes | Lines Changed |
|------|---------|----------------|
| `s_graphs.hpp` | Added zone_cache member, parameters | +8 |
| `s_graphs.cpp` | Initialize ZoneCache, parameters | +14 |
| `loop_mapper.hpp` | Added zone_cache member, set_zone_cache() | +8 |
| `loop_mapper.cpp` | Zone weighting logic, constructor changes | +65 |
| `zones_node.py` | Fixed import (situational_graphs_msgs) | 1 line |
| `CMakeLists.txt` | Added zones_node.py install, Zone.msg build | +2 |
| `s_graphs_launch.py` | Added zones_node launch | +10 |
| `CMakeLists.txt` (msgs) | Added Zone.msg to rosidl_generate_interfaces | +1 |

### Build System Changes

**situational_graphs_msgs/CMakeLists.txt:**
```cmake
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyframeSemantic.msg"
  "msg/DynamicObjects.msg"
  "msg/Zone.msg"  # NEW
  # ... other messages
)
```

**lidar_situational_graphs/CMakeLists.txt:**
```cmake
install(
  PROGRAMS
    ${PROJECT_NAME}/map2odom_publisher.py
    ${PROJECT_NAME}/tf_filter.py
    ${PROJECT_NAME}/iqa_ros.py
    ${PROJECT_NAME}/iqm.py
    scripts/zones_node.py  # NEW
  DESTINATION lib/${PROJECT_NAME}
)
```

