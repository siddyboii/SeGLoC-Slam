# Deep Codebase Analysis: S-Graphs Custom Fork
## From Commit `2be157b` → `a8b6bbb`
### Author: Subhanshu | Period: Jan 12, 2026 → Feb 28, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Commit Timeline](#2-commit-timeline)
3. [High-Level What Changed (Numbers)](#3-high-level-what-changed-numbers)
4. [Phase 1 — Camera Integration Foundation (Commits 1–2)](#4-phase-1--camera-integration-foundation-commits-12)
5. [Phase 2 — YOLO + CLIP Semantic Pipeline (Commits 3–4)](#5-phase-2--yolo--clip-semantic-pipeline-commits-34)
6. [Phase 3 — Geometric Validation with ICP (Commit 5)](#6-phase-3--geometric-validation-with-icp-commit-5)
7. [Phase 4 — DynaTrack Dynamic Object Tracking (Commits 6–7)](#7-phase-4--dynatrack-dynamic-object-tracking-commits-67)
8. [Phase 5 — Zone-Based Semantic Prefiltering (Commit 8)](#8-phase-5--zone-based-semantic-prefiltering-commit-8)
9. [Phase 6 — Metrics, Analysis & Final Integration (Commit 9)](#9-phase-6--metrics-analysis--final-integration-commit-9)
10. [File-by-File Change Log](#10-file-by-file-change-log)
11. [Architecture Evolution Diagram](#11-architecture-evolution-diagram)
12. [New Data Structures](#12-new-data-structures)
13. [New ROS Messages](#13-new-ros-messages)
14. [New ROS Parameters](#14-new-ros-parameters)
15. [New Nodes & Scripts](#15-new-nodes--scripts)
16. [Build System Changes](#16-build-system-changes)
17. [Infrastructure Changes](#17-infrastructure-changes)
18. [What Was NOT Changed (Original S-Graphs Core)](#18-what-was-not-changed-original-s-graphs-core)
19. [Complete Summary Table](#19-complete-summary-table)

---

## 1. Executive Summary

The original S-Graphs codebase (`2be157b`) was the **unmodified upstream release** from the authors — a pure LiDAR-based Situational Graphs SLAM system that fused odometry + point clouds into a hierarchical graph of keyframes, walls, rooms, and floors.

Over 47 days (Jan 12 → Feb 28, 2026), across **9 commits** and **82 files changed**, the system was transformed into a **multi-modal semantic SLAM pipeline** by adding:

1. **Camera input** (RGB images from Alphasense cam0) synchronized with LiDAR + odometry  
2. **YOLO-World object detection** (TensorRT GPU-accelerated, 5 classes)  
3. **CLIP visual embedding extraction** (ONNXRuntime GPU, 512-dim feature vectors)  
4. **Image Quality Assessment (IQA)** Python node computing a normalized quality gate  
5. **DynaTrack** — BEV optical flow dynamic object tracking (git submodule)  
6. **Semantic loop closure weighting** — a full multi-factor pipeline modulating the g2o information matrix using CLIP similarity, object overlap, IQA score, dynamicity, and zone identity  
7. **Zone-based prefiltering** — a Python node (`zones_node.py`) that clusters keyframes into semantic zones using YOLO labels + room plane structure, with a C++ `ZoneCache` feeding zone data to loop closure  
8. **Real-time metrics recording** and **post-run analysis** scripts  
9. A dedicated **CUDA Docker environment** (CUDA 12.1 + TensorRT + ONNXRuntime + ROS2 Humble)

The total additions: **+10,671 lines** of new code across C++ headers, C++ implementation files, Python scripts, ROS message definitions, launch files, Docker infrastructure, and documentation.

---

## 2. Commit Timeline

| # | Hash | Date | Message |
|---|------|------|---------|
| 0 | `2be157b` | Jan 12, 2026 | **BASELINE** — Original S-Graphs by authors |
| 1 | `2a0bf72` | Jan 18, 2026 | Camera integration — CLIP verified, model not included |
| 2 | `786cd5b` | Jan 25, 2026 | Camera integration + bag runs with any custom bag |
| 3 | `a4590dd` | Jan 28, 2026 | YOLO + CLIP inferencing running in pipeline |
| 4 | `03af988` | Feb 10, 2026 | Added geometric validation with ICP scores |
| 5 | `c8fe6b2` | Feb 20, 2026 | DynaTrack tracking built, not yet in main code |
| 6 | `ee52fd0` | Feb 23, 2026 00:35 | Tracking part added, needs validation |
| 7 | `a421bdc` | Feb 23, 2026 23:27 | Last working code without zone logic |
| 8 | `f65990e` | Feb 23, 2026 01:09 | Zone-based logic integrated and built |
| 9 | `a8b6bbb` | Feb 28, 2026 | **FINAL** — Working code with metrics + odom corrections |

---

## 3. High-Level What Changed (Numbers)

| Metric | Value |
|--------|-------|
| Total files changed | **82** |
| Lines added | **+10,671** |
| Lines removed | **−167** |
| New C++ source files | **5** |
| New C++ header files | **5** |
| New Python scripts | **9** |
| New ROS message types | **3** |
| New launch files | **2** |
| New Docker environments | **1** |
| New git submodules | **1** (DynaTrack) |

---

## 4. Phase 1 — Camera Integration Foundation (Commits 1–2)

### Commits: `2a0bf72` (Jan 18) and `786cd5b` (Jan 25)

**Goal:** Wire a camera image stream into the existing dual-synchronizer (odom + pointcloud), making it a triple-synchronizer.

---

### 4.1 `lidar_situational_graphs/include/s_graphs/common/s_graphs.hpp`

**What changed:**

- Added two new includes at the top:
  ```cpp
  #include <cv_bridge/cv_bridge.h>
  #include <opencv2/opencv.hpp>
  ```
- Added three new includes for semantic/zone headers:
  ```cpp
  #include <s_graphs/common/yolo_object_detector.hpp>
  #include <s_graphs/common/clip_feature_extractor.hpp>
  #include <s_graphs/frontend/zone_cache.hpp>
  ```
- Added new ROS message type includes:
  ```cpp
  #include "sensor_msgs/msg/image.hpp"
  #include "std_msgs/msg/float64.hpp"
  #include "situational_graphs_msgs/msg/keyframe_semantic.hpp"
  #include "situational_graphs_msgs/msg/dynamic_objects.hpp"
  ```

**Synchronizer changed from dual to triple:**
```cpp
// BEFORE (original):
typedef message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::PointCloud2> ApproxSyncPolicy;
std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

// AFTER:
typedef message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::PointCloud2,
    sensor_msgs::msg::Image> TripleSyncPolicy;
std::shared_ptr<message_filters::Synchronizer<TripleSyncPolicy>> sync;
```

**New callback declared:**
```cpp
void cloud_image_odom_callback(
    const nav_msgs::msg::Odometry::SharedPtr odom_msg,
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
    const sensor_msgs::msg::Image::SharedPtr image_msg);
```

**New subscriber member added:**
```cpp
message_filters::Subscriber<sensor_msgs::msg::Image> image_sub;
```

**New member variables added in private section (48 new fields total):**
```
// Semantic models
std::unique_ptr<YOLOWorldTensorRT> yolo_detector_
std::unique_ptr<ClipFeatureExtractor> clip_extractor_

// Semantic publishers
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_image_pub_
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detection_markers_pub_
rclcpp::Publisher<situational_graphs_msgs::msg::KeyframeSemantic>::SharedPtr keyframe_semantic_pub_

// Semantic parameters
std::string yolo_model_path_, clip_model_path_
double confidence_threshold_
int process_every_n_frames_
bool enable_yolo_, enable_clip_, visualize_detections_

// Semantic state
long frame_count_ = 0, processed_count_ = 0
std::vector<float> last_embedding_
std::mutex keyframe_mutex_

// IQA subscription
rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr iqa_score_sub_
std::mutex iqa_mutex_
double latest_iqa_score_ = 0.5
bool iqa_score_received_ = false

// DynaTrack subscription
rclcpp::Subscription<situational_graphs_msgs::msg::DynamicObjects>::SharedPtr dynamic_objects_sub_
std::mutex dynamicity_mutex_
float latest_scene_dynamicity_ = 0.0f
int latest_num_dynamic_clusters_ = 0
bool dynamicity_received_ = false

// Zone prefiltering
std::shared_ptr<ZoneCache> zone_cache_
bool use_zone_prefilter_ = true
double zone_confidence_threshold_ = 0.3
```

**New method declarations:**
```cpp
void initializeSemanticModels();
void processSemantic(const cv::Mat& image, const std_msgs::msg::Header& header, KeyFrame::Ptr keyframe);
void publishDetectionVisualization(const cv::Mat& image, const std::vector<Detection>& detections, const std_msgs::msg::Header& header);
void publishDetectionMarkers(const std::vector<Detection>& detections, const std_msgs::msg::Header& header);
void dynamic_objects_callback(const situational_graphs_msgs::msg::DynamicObjects::SharedPtr dyn_msg);
void iqa_score_callback(const std_msgs::msg::Float64::SharedPtr msg);
```

---

### 4.2 `lidar_situational_graphs/include/s_graphs/common/keyframe.hpp`

**What changed:** Added **20 new optional fields** to the `KeyFrame` struct, turning each keyframe from a pure geometric record into a rich multimodal record.

```cpp
// ADDED: Camera data
boost::optional<cv::Mat> image_rgb;           // Raw RGB image at this keyframe
boost::optional<cv::Mat> depth_img;           // Depth image (reserved for future use)

// ADDED: CLIP features
boost::optional<std::vector<float>> clip_embedding;  // 512-dimensional visual embedding

// ADDED: YOLO detections
boost::optional<std::vector<std::string>> detected_objects;       // Class names
boost::optional<std::map<std::string, float>> object_confidences; // Per-class confidence

// ADDED: Scene classification
boost::optional<std::string> scene_type;       // Room type label
boost::optional<float> scene_confidence;       // Confidence of scene label

// ADDED: Image quality metrics (from IQA node)
boost::optional<float> image_brightness;       // Normalized IQA score [0,1]
boost::optional<float> image_sharpness;        // Reserved [0,1]

// ADDED: Scene dynamicity (from DynaTrack)
boost::optional<float> scene_dynamicity;       // [0,1]: 0=static, 1=highly dynamic
boost::optional<int>   num_dynamic_clusters;   // Number of dynamic object clusters

// ADDED: OpenCV include
#include <opencv2/core.hpp>
```

Also added a comment explaining `boost::optional`:  
> "boost::optional mean the variable may or may not contain the values"

---

### 4.3 `lidar_situational_graphs/src/s_graphs/common/s_graphs.cpp`

**What changed in the constructor `SGraphsNode::SGraphsNode()`:**

- The `image_sub` subscriber is now created and subscribed to `/alphasense/cam0/image_raw`
- The triple sync policy is instantiated and `cloud_image_odom_callback` is registered
- The original dual sync code is commented out
- Four new subscribers are created: `iqa_score_sub_`, `dynamic_objects_sub_` (new), plus the existing flow is preserved
- Three new semantic publishers are created: `detection_image_pub_`, `detection_markers_pub_`, `keyframe_semantic_pub_`
- 7 new semantic parameters are read from the node parameter server

**New callback body `cloud_image_odom_callback()` (replacing the old `cloud_callback()`):**
- Takes odom + pointcloud + image simultaneously
- Converts image via `cv_bridge::toCvCopy` → `cv::Mat` (BGR8)
- Creates keyframe as before, then additionally sets `keyframe->image_rgb = image_rgb.clone()`
- Attaches dynamicity data from the DynaTrack mutex-protected state
- Spawns a detached `std::thread` to call `processSemantic()` asynchronously
- The old `cloud_callback()` body (~250 lines) is preserved but commented out entirely

**New method `initializeSemanticModels()`:**
```
- Checks that yolo_model_path_ and clip_model_path_ are not empty
- Creates YOLOWorldTensorRT instance from TRT engine file
- Creates ClipFeatureExtractor instance from ONNX weights file
- Logs success/failure for each model
- Catches exceptions and logs errors without crashing
```

**New method `processSemantic(image, header, keyframe)`:**
```
- Runs YOLO detector: infer(image_rgb, confidence_threshold_)
- Extracts object names and confidences → stores in keyframe->detected_objects, keyframe->object_confidences
- Attaches IQA score to keyframe->image_brightness if iqa_score_received_
- Runs CLIP extractor: extract_embedding(image_rgb) → stores in keyframe->clip_embedding
- Publishes KeyframeSemantic message on topic s_graphs/keyframe_semantic
- Optionally calls publishDetectionVisualization() and publishDetectionMarkers()
- Frame-rate controlled by process_every_n_frames_
- All YOLO/CLIP work guarded by keyframe_mutex_
```

**New method `publishDetectionVisualization()`:**
- Draws bounding boxes and labels on image using OpenCV
- Converts back to ROS Image message via cv_bridge
- Publishes on `s_graphs/semantic/detection_image`

**New method `publishDetectionMarkers()`:**
- Creates RViz `visualization_msgs::MarkerArray` TEXT_VIEW_FACING markers
- Positions them at normalized detection coordinates
- Publishes on `s_graphs/semantic/detection_markers`

**New method `iqa_score_callback()`:**
```cpp
void SGraphsNode::iqa_score_callback(const std_msgs::msg::Float64::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(iqa_mutex_);
  latest_iqa_score_ = msg->data;
  iqa_score_received_ = true;
}
```

**New method `dynamic_objects_callback()`:**
```cpp
// Stores scene_dynamicity and num_dynamic_clusters from DynaTrack
// under dynamicity_mutex_ for thread-safe access from cloud_image_odom_callback
```

**In `declare_ros_params()`:** Added 9 new parameter declarations:
```
yolo_model_path, clip_model_path, confidence_threshold,
process_every_n_frames, enable_yolo, enable_clip,
visualize_detections, use_zone_prefilter, zone_confidence_threshold
```

**In `init_subclass()`:** After loop_mapper creation, now also:
```cpp
zone_cache_ = std::make_shared<ZoneCache>();
zone_cache_->init(shared_from_this(), "zones");
loop_mapper->set_zone_cache(zone_cache_);
initializeSemanticModels();  // Initialize YOLO + CLIP
```

---

## 5. Phase 2 — YOLO + CLIP Semantic Pipeline (Commits 3–4)

### Commits: `a4590dd` (Jan 28) and `03af988` (Feb 10)

---

### 5.1 NEW FILE: `lidar_situational_graphs/include/s_graphs/common/yolo_object_detector.hpp`

A **complete YOLO-World TensorRT C++ class** was written from scratch.

**Class: `YOLOWorldTensorRT`**

| Element | Detail |
|---------|--------|
| Inference backend | NVIDIA TensorRT (`NvInfer.h`, CUDA runtime) |
| Input | File path string OR `cv::Mat` directly |
| Output | `std::vector<Detection>` |
| Classes detected | `{"Persons", "Windows", "Lights", "Door", ""}` (5 classes) |
| Model format | `.trt` pre-compiled TensorRT engine |
| Batch size | Fixed at 1 |
| NMS | Custom `applyNMS()` with IoU threshold |
| GPU buffers | 4 buffers: input, boxes, scores, classes |

**`Detection` struct:**
```cpp
struct Detection {
    float x1, y1, x2, y2;   // bounding box
    float confidence;
    int class_id;
    std::string class_name;
};
```

**Key private methods:**
- `preprocessImage(string path)` — loads image from disk, letter-box pads to model input size
- `preprocessImage(cv::Mat image)` — direct cv::Mat version for ROS integration
- `postprocessOutputs(float threshold)` — parses raw GPU output tensors → Detection objects
- `allocateBuffers()` — allocates GPU + CPU buffers for TRT I/O
- `calculateIoU()` — IoU between two Detections
- `applyNMS()` — greedy NMS to remove duplicate boxes

---

### 5.2 NEW FILE: `lidar_situational_graphs/src/s_graphs/common/yolo_object_detector.cpp`

**464 lines** implementing the `YOLOWorldTensorRT` class.

Key implementation details:
- Constructor parses TRT engine file via `nvinfer1::IRuntime::createRuntime()`
- `allocateBuffers()` uses `cudaMalloc` + `cudaMemcpy` patterns
- Preprocessing: letter-box resize (maintains aspect ratio with zero-padding), normalization to [0,1]
- `postprocessOutputs`: iterates over all anchors, applies sigmoid to scores, filters by confidence threshold, then applies NMS
- Destructor: `cudaFree()` all GPU buffers

---

### 5.3 NEW FILE: `lidar_situational_graphs/include/s_graphs/common/clip_feature_extractor.hpp`

A **complete CLIP visual encoder C++ class** wrapping ONNXRuntime.

**Class: `ClipFeatureExtractor`**

| Element | Detail |
|---------|--------|
| Inference backend | ONNX Runtime (`onnxruntime_cxx_api.h`) |
| Input size | 224 × 224 RGB |
| Output | `std::vector<float>` — 512-dimensional L2-normalized embedding |
| Model format | `.onnx` (visual encoder half of CLIP, e.g., `visual.onnx`) |
| Normalization | ImageNet mean/std: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

**Public API:**
```cpp
std::vector<float> extract_embedding(const cv::Mat& image);
float compute_similarity(const std::vector<float>& emb1, const std::vector<float>& emb2);
```

**Key private methods:**
- `load_ort_params()` — creates `Ort::Session` from ONNX file, reads input/output binding names
- `ortForward(cv::Mat bgr)` — runs full forward pass, returns raw output tensor as float vector
- `makeInputTensor(bgr, chw, dims)` — converts HWC BGR → CHW float32 with ImageNet normalization
- `init_input_size_safe_()` — parses input tensor shape from ONNX model safely

---

### 5.4 NEW FILE: `lidar_situational_graphs/src/s_graphs/common/clip_feature_extractor.cpp`

**176 lines** implementing `ClipFeatureExtractor`.

Implements L2 normalization of the output embedding and safe error handling if the model cannot be loaded.

---

### 5.5 NEW FILE: `lidar_situational_graphs/include/s_graphs/common/semantic_evidence.hpp`

A **data aggregation struct** `SemanticEvidence` collecting every semantic signal for a location:

```
// LiDAR-based semantics
double plane_stability_score          [0,1]
std::vector<double> plane_normals_variance
std::vector<double> plane_distance_variance
int num_detected_planes
double spatial_coverage               [0,1]
double geometric_distinctiveness      [0,1]
size_t point_count
double point_density                  points/m³
double point_distribution_uniformity  [0,1]

// Camera-based semantics
boost::optional<std::vector<float>> clip_embedding    512-dim
float clip_embedding_magnitude
std::vector<std::string> detected_objects
std::map<std::string, float> object_confidences
int num_high_confidence_objects
float object_detection_confidence_mean
boost::optional<std::string> scene_type
float scene_distinctiveness           [0,1]
std::map<std::string, float> scene_class_probabilities
float image_brightness                [0,255]
float image_sharpness                 [0,1]
float image_motion_blur               [0,1]
float image_saturation                [0,1]

// Visibility & consistency
double appearance_consistency         [0,1]
double temporal_stability             [0,1]
double visibility_confidence          [0,1]
bool is_well_observed_lidar
bool is_well_observed_camera

// Multi-modal fusion output
double combined_distinctiveness       [0,1]
float modality_agreement_score        [0,1]
float overall_semantic_confidence     [0,1] ← FINAL
```

This struct is used for future full multi-modal fusion but is defined and ready in the include tree.

---

### 5.6 NEW FILE: `lidar_situational_graphs/apps/semantic_test_node.cpp`

**422-line standalone ROS2 node** for testing semantic inference outside of the full S-Graphs pipeline.

**What it does:**
- Subscribes to a camera image topic (configurable)
- Runs YOLO + CLIP on each frame (throttled by `process_every_n_frames`)
- Publishes:
  - Detection image with bounding boxes drawn
  - Detection `MarkerArray` for RViz
  - CLIP embedding magnitude as a diagnostic Float32
- Prints per-frame inference timing (YOLO ms, CLIP ms, total ms)
- Provides enable/disable flags for each model independently

**Why it exists:** Validated that TensorRT YOLO and ONNXRuntime CLIP run inside the ROS2 execution environment before integrating into the main S-Graphs node.

---

### 5.7 NEW FILE: `lidar_situational_graphs/launch/semantic_test_launch.py`

A 104-line launch file that runs just the `semantic_test_node` with configurable parameters for quick testing.

---

### 5.8 `lidar_situational_graphs/config/s_graphs.yaml`

**What changed:**

Renamed frame IDs from `body` → `base_link`:
```yaml
# BEFORE:
plane_extraction_frame_id: body
plane_visualization_frame_id: body_elevated

# AFTER:
plane_extraction_frame_id: base_link
plane_visualization_frame_id: base_link
```

Changed g2o solver type:
```yaml
# BEFORE:
g2o_solver_type: "lm_var_cholmod"

# AFTER:
g2o_solver_type: "lm_pcg"   # more compatible, no CHOLMOD dependency
```

Added 7 new semantic parameters at the end:
```yaml
yolo_model_path: "/workspace/src/s_graphs/models/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.trt"
clip_model_path: "/workspace/src/s_graphs/models/visual.onnx"
confidence_threshold: 0.25
process_every_n_frames: 5
enable_yolo: true
enable_clip: true
visualize_detections: true
```

---

## 6. Phase 3 — Geometric Validation with ICP (Commit 4)

### Commit: `03af988` (Feb 10, 2026)

The key change here was piping the ICP scan-matching **fitness score** from loop detection through to loop closure weighting.

---

### 6.1 `lidar_situational_graphs/include/s_graphs/frontend/loop_detector.hpp`

**What changed:**

The `Loop` struct received a new field and updated constructor:

```cpp
// BEFORE:
struct Loop {
  Loop(const KeyFrame::Ptr& key1, const KeyFrame::Ptr& key2,
       const Eigen::Matrix4f& relpose)
      : key1(key1), key2(key2), relative_pose(relpose) {}
  KeyFrame::Ptr key1, key2;
  Eigen::Matrix4f relative_pose;
};

// AFTER:
struct Loop {
  Loop(const KeyFrame::Ptr& key1, const KeyFrame::Ptr& key2,
       const Eigen::Matrix4f& relpose, double fitness_score)
      : key1(key1), key2(key2), relative_pose(relpose), fitness_score(fitness_score) {}
  KeyFrame::Ptr key1, key2;
  Eigen::Matrix4f relative_pose;
  double fitness_score;   // ← NEW: ICP scan matching fitness score from registration
};
```

The return statement in `matching()` was updated:
```cpp
// BEFORE:
return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);

// AFTER:
return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose, best_score);
```

This ensures every loop candidate carries the ICP fitness score used to find it.

---

### 6.2 `lidar_situational_graphs/include/s_graphs/backend/loop_mapper.hpp`

**What changed:**

- Added `#include <fstream>` for CSV logging
- Added `#include <s_graphs/frontend/zone_cache.hpp>`
- `set_zone_cache()` method declared
- Three new private computation methods declared:
  ```cpp
  double computeClipSimilarity(const std::vector<float>& emb1, const std::vector<float>& emb2);
  double confidenceToAlpha(double clip_similarity);
  double computeObjectOverlap(const std::vector<std::string>& objects1, const std::vector<std::string>& objects2);
  ```
- New private member variables:
  ```cpp
  rclcpp::Node::SharedPtr node_;
  double fitness_score_thresh;
  std::shared_ptr<ZoneCache> zone_cache_;
  bool use_zone_prefilter_ = true;
  double zone_confidence_threshold_ = 0.3;
  std::ofstream loop_metrics_csv_;
  ```
- Fixed a bug in the original: `#endif #LOOP_MAPPER_HPP` → `#endif  // LOOP_MAPPER_HPP`

---

### 6.3 `lidar_situational_graphs/src/s_graphs/backend/loop_mapper.cpp`

This file grew from ~60 lines to **520 lines** — the most heavily modified C++ file. It implements the entire multi-factor semantic loop closure weighting pipeline.

**Constructor changes:**
```cpp
// BEFORE: simple, only creates inf_calclator
LoopMapper(node, mutex) : shared_graph_mutex(graph_mutex) {
  inf_calclator.reset(new InformationMatrixCalculator(node));
}

// AFTER: reads 3 parameters, opens CSV logger
LoopMapper(node, mutex) : shared_graph_mutex(graph_mutex), node_(node) {
  inf_calclator.reset(new InformationMatrixCalculator(node));
  fitness_score_thresh = node->get_parameter("fitness_score_thresh").get_parameter_value().get<double>();
  use_zone_prefilter_ = node->get_parameter("use_zone_prefilter")...
  zone_confidence_threshold_ = node->get_parameter("zone_confidence_threshold")...
  
  // Opens /tmp/sgraphs_metrics/loop_metrics.csv with columns:
  // kf1_id, kf2_id, has_clip, clip_sim, semantic_alpha,
  // icp_fitness, object_overlap, quality_gate,
  // dyn_factor, zone_factor, info_matrix_norm,
  // edge_added, zone1, zone2, same_zone
}
```

**Destructor:** closes the CSV file.

**`add_loops()` — the core semantic weighting function:**

The function now implements a **5-factor information matrix scaling pipeline** for each loop closure candidate:

```
Step 0: Print LOOP CLOSURE SEMANTIC WEIGHTING PIPELINE banner
Step 1: Get geometry-only information matrix (ICP-based, original behavior)
Step 2: Get ICP fitness score from loop->fitness_score (new)
Step 3: Initialize semantic_alpha = 1.0

IF both keyframes have CLIP embeddings:
  ├── Step 4: Compute CLIP cosine similarity [0,1]
  ├── Step 5: Compute object Jaccard overlap [0,1]
  │     • overlap > 0.7 → alpha *= 1.15  (boost)
  │     • overlap < 0.3 → alpha *= 0.85  (penalize)
  ├── Step 6: Geometry-Semantic Interaction
  │     • geometric_reliability = 1 - (icp_fitness / fitness_thresh)  [0,1]
  │     • semantic_influence = 1 - (geometric_reliability * 0.5)
  │     • alpha = 1 + (alpha - 1) * semantic_influence
  │     (excellent ICP → dampen semantic deviation from neutral)
  ├── Step 7: Image Quality Gating
  │     • quality_gate = min(kf1.image_brightness, kf2.image_brightness)
  │     • alpha = quality_gate * alpha + (1 - quality_gate) * 1.0
  │     (poor image quality → pull alpha back toward 1.0)
  ├── Step 8: Scene Dynamicity Factor (DynaTrack)
  │     • max_dyn = max(kf1.scene_dynamicity, kf2.scene_dynamicity)
  │     • dyn_factor = 1.0 - max_dyn * 0.3  (up to 30% reduction)
  │     • alpha *= dyn_factor
  └── Step 9: Zone-Based Semantic Coherence
        • same zone → alpha *= (1 + zone_conf * 0.2)   up to 20% boost
        • different zones, shared planes → alpha *= 0.9  mild penalty
        • different zones, no shared planes → alpha *= (1 - min_conf * 0.5)  up to 50% reduction (min 30%)

Step 10: Apply: information_matrix *= semantic_alpha
Step 11: Add edge to g2o graph (unchanged logic)
Step 12: Write row to CSV log
Step 13: Print summary statistics
```

**`confidenceToAlpha(clip_similarity)` — maps CLIP similarity to scaling factor:**
```
alpha_min = 0.1,  alpha_max = 2.0,  threshold = 0.7
  • clip_sim >= 0.7: alpha = 1 + (sim - 0.7)/(0.3) * 1.0   [1.0 → 2.0]
  • clip_sim <  0.7: alpha = 0.1 + (sim/0.7) * 0.9          [0.1 → 1.0]
```

**`computeClipSimilarity(emb1, emb2)` — cosine similarity:**
```
cos_sim = dot(emb1, emb2) / (||emb1|| * ||emb2||)
returns (cos_sim + 1.0) / 2.0   → maps [-1,1] to [0,1]
```

**`computeObjectOverlap(objects1, objects2)` — Jaccard index:**
```
overlap_count = |objects1 ∩ objects2|
total_unique  = |objects1| + |objects2| - overlap_count
jaccard_index = overlap_count / total_unique
```

---

## 7. Phase 4 — DynaTrack Dynamic Object Tracking (Commits 5–7)

### Commits: `c8fe6b2` (Feb 20), `ee52fd0` (Feb 23 00:35), `a421bdc` (Feb 23 23:27)

---

### 7.1 NEW GIT SUBMODULE: `tracking_OF/`

Added as a git submodule pointing to commit `84ea218`.

**Contents:**
- `CMakeLists.txt` — ROS2 package build
- `package.xml` — ROS2 package metadata
- `include/` — header files
- `src/` — BEV optical flow tracking source
- `instruction.md` — usage guide
- `trackingpcl_with_classical.pdf` — algorithm documentation

**What it does:**
- Subscribes to filtered LiDAR point cloud and odometry
- Performs Bird's Eye View (BEV) projection of the point cloud
- Runs classical optical flow to detect moving clusters
- Publishes `situational_graphs_msgs/msg/DynamicObjects` message with:
  - `scene_dynamicity` — float [0,1] representing fraction of dynamic pixels
  - `num_dynamic_clusters` — number of tracked dynamic clusters
- Topic: `dynamic_objects` (remapped from `/velodyne_points` in launch)

**Integration point in `s_graphs.cpp`:**
```cpp
// In cloud_image_odom_callback():
{
  std::lock_guard<std::mutex> dyn_lock(dynamicity_mutex_);
  if (dynamicity_received_) {
    keyframe->scene_dynamicity = latest_scene_dynamicity_;
    keyframe->num_dynamic_clusters = latest_num_dynamic_clusters_;
  }
}
```

---

### 7.2 `lidar_situational_graphs/src/s_graphs/backend/floor_mapper.cpp`

**What changed:** Added defensive null-checks — a bug that could crash during startup when `graph_slam` was not yet initialized or when `floors_vec` had entries with null nodes.

```cpp
// In lookup_floors():
if (!graph_slam) {
  std::cerr << "[FLOOR_MAPPER ERROR] graph_slam is null!" << std::endl;
  return;
}

// In associate_floors():
if (floors_vec.empty()) {
  return data_association;
}
for (const auto& mapped_floor : floors_vec) {
  if (mapped_floor.second.node == nullptr) {  // ← NEW safety check
    continue;
  }
  ...
}
```

These were crash bugs discovered during integration testing.

---

### 7.3 `lidar_situational_graphs/apps/prefiltering_node.cpp`

**What changed:**

- Added extensive commented-out debug logging blocks at each processing stage (for future re-enabling)
- Changed TF error log level from `RCLCPP_INFO` to `RCLCPP_ERROR` for transform failures:
  ```cpp
  // BEFORE:
  RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s", ...);
  // AFTER:
  RCLCPP_ERROR(this->get_logger(), "[PREFILTER DEBUG] Could not transform %s to %s: %s", ...);
  ```
- Removed the `prefix=["bash -c 'exec 2>/dev/null; $0 $@'"]` from the launch node definition (the stderr suppression was hiding errors during debugging)

---

## 8. Phase 5 — Zone-Based Semantic Prefiltering (Commit 8)

### Commit: `f65990e` (Feb 23, 2026 01:09)

This is the most architecturally complex addition — a full zone management system.

---

### 8.1 NEW FILE: `lidar_situational_graphs/include/s_graphs/frontend/zone_cache.hpp`

**Class `ZoneCache`** — a thread-safe in-memory cache of zone data built from incoming `Zone` messages.

```cpp
// Query API:
int get_zone_for_keyframe(int keyframe_id) const;          // → zone_id or -1
std::vector<int> get_keyframes_in_zone(int zone_id) const;
std::set<int> get_support_plane_ids(int zone_id) const;
double get_zone_confidence(int zone_id) const;
bool has_zone(int zone_id) const;
int get_zone_floor(int zone_id) const;
std::vector<std::string> get_zone_labels(int zone_id) const;

// Internal state maps (all protected by mtx_):
std::unordered_map<int, std::vector<int>>  zone_to_kfs_
std::unordered_map<int, int>               kf_to_zone_
std::unordered_map<int, std::set<int>>     zone_support_planes_
std::unordered_map<int, double>            zone_confidence_
std::unordered_map<int, uint64_t>          zone_version_
std::unordered_map<int, int>               zone_floor_
std::unordered_map<int, std::vector<std::string>> zone_labels_
```

---

### 8.2 NEW FILE: `lidar_situational_graphs/src/s_graphs/frontend/zone_cache.cpp`

**138 lines** implementing `ZoneCache`.

The `zone_callback()` handles three actions from the `Zone` message:
- `action == CREATE` or `action == UPDATE` (version-guarded to ignore stale messages)
- `action == DELETE` — removes all entries for a zone_id

---

### 8.3 NEW ROS MESSAGE: `lidar_situational_graphs/msg/Zone.msg`

```
std_msgs/Header header
int32 zone_id
int32 floor_id
geometry_msgs/Pose centroid
geometry_msgs/Polygon polygon
int32[] room_ids
int32[] keyframe_ids
string[] top_labels
float32[] top_label_confidences
float32 confidence
situational_graphs_msgs/PlaneData[] supporting_planes
int32[] supporting_wall_ids
uint8 action        # 0=CREATE, 1=UPDATE, 2=DELETE
uint64 version
```

---

### 8.4 NEW SCRIPT: `lidar_situational_graphs/scripts/zones_node.py` (885 lines)

A large Python ROS2 node implementing the **zone creation and management logic**.

**Subscriptions:**
- `room_segmentation/room_data` (`RoomsData`) — room plane data
- `s_graphs/keyframe_semantic` (`KeyframeSemantic`) — per-keyframe YOLO objects
- `s_graphs/graph_keyframes` (`GraphKeyframes`) — keyframe poses from graph

**Publication:**
- `zones` (`Zone` messages) — create/update/delete zone events

**Algorithm:**
1. **Semantic matching:** Uses `OBJECT_TO_LABEL` dictionary to map YOLO class names → room type labels (e.g., `"persons" → {"hallway":0.5, "living_room":0.4}`)
2. **Plane-aware spatial clustering:** Uses `shapely` to build 2D polygons from room wall planes and test which keyframes fall inside
3. **Zone merging:** Adjacent zones with sufficiently similar semantic labels are merged (using Jaccard on label sets)
4. **Confidence updating:** Zone confidence is the weighted average of all keyframe semantic confidences inside it
5. **Change detection:** Sends UPDATE messages only when confidence or keyframe membership changes meaningfully

**Key rule dictionary `OBJECT_TO_LABEL`:**
```python
{
  'persons':   {'hallway': 0.5, 'living_room': 0.4, 'office': 0.3},
  'windows':   {'office': 0.6, 'living_room': 0.5, ...},
  'lights':    {'hallway': 0.6, 'office': 0.5, ...},
  'door':      {'hallway': 0.7, 'corridor': 0.6, ...},
  'chair':     {'office': 0.8, 'living_room': 0.5, ...},
  'table':     {'dining': 0.7, 'office': 0.6, ...},
  # ... 30+ object classes
}
```

---

### 8.5 `loop_mapper.cpp` — Zone Coherence Factor (already described in Section 6.3 Step 9)

The zone-based logic in `add_loops()` was added in this phase: using `ZoneCache` queries to boost/penalize loop closure information matrices based on whether the two keyframes are in the same zone, adjacent zones, or completely separate zones.

---

## 9. Phase 6 — Metrics, Analysis & Final Integration (Commit 9)

### Commit: `a8b6bbb` (Feb 28, 2026)

---

### 9.1 NEW SCRIPT: `lidar_situational_graphs/scripts/metrics_recorder.py` (293 lines)

A ROS2 node that runs alongside the system and records **6 CSV files** in real-time:

| File | Content |
|------|---------|
| `corrected_trajectory.csv` | Graph-optimized poses (x,y,z,qx,qy,qz,qw,stamp) |
| `raw_odometry.csv` | Raw scan-matching odometry poses |
| `keyframe_semantics.csv` | Per-keyframe: CLIP dim, objects, IQA score, dynamicity |
| `dynamic_objects.csv` | Per-frame DynaTrack: dynamicity, num_clusters |
| `zone_events.csv` | Zone create/update/delete events with labels, confidence |
| `image_quality.csv` | IQA score vs. time |

**Subscriptions:**
- `s_graphs/odom_pose_corrected` (PoseStamped)
- `odom` (Odometry)
- `s_graphs/keyframe_semantic` (KeyframeSemantic)
- `dynamic_objects` (DynamicObjects)
- `zones` (Zone)
- `/image_quality_score` (Float64)

---

### 9.2 NEW SCRIPT: `lidar_situational_graphs/scripts/analyse_metrics.py` (898 lines)

A comprehensive post-run analysis tool. Run from the command line after the bag has finished playing.

**Computes:**

*Trajectory Metrics:*
- ATE (Absolute Trajectory Error) RMSE — compares corrected trajectory to ground truth
- RPE (Relative Pose Error) — frame-to-frame drift
- Total drift as % of path length

*Loop Closure Metrics (from `loop_metrics.csv`):*
- Total loop count, % with semantic weighting, % geometry-only
- CLIP similarity distribution: mean, std, percentiles, ASCII histogram
- Semantic alpha distribution
- ICP fitness score distribution
- Object overlap distribution
- IQA quality gate distribution
- Dynamicity factor distribution
- Zone factor distribution
- Information matrix norm distribution

*Semantic Coverage Metrics (from `keyframe_semantics.csv`):*
- % keyframes with CLIP embeddings
- % keyframes with detected objects
- % keyframes with IQA scores
- % keyframes with dynamicity data
- Object class frequency histogram
- Scene dynamicity over time plot

*Zone Metrics (from `zone_events.csv`):*
- Number of zones created / updated / deleted
- Average keyframes per zone
- Average zone confidence distribution

**Output:** Prints formatted table to stdout, saves `metrics_summary.txt`, generates PNG plots if matplotlib is available.

---

### 9.3 NEW SCRIPT: `lidar_situational_graphs/scripts/sim_kf_pub.py` (161 lines)

Synthetic KeyframeSemantic publisher for testing `zones_node.py` without a real robot.

- Publishes at 3 Hz
- Cycles through 4 semantic modes every 30 seconds: bedroom → kitchen → living room → bathroom
- Reads actual odometry for poses
- Triggers zone creation in `zones_node.py`

---

### 9.4 NEW FILE: `lidar_situational_graphs/launch/s_graphs_hilti_launch.py` (126 lines)

A dataset-specific launch file for the HILTI-SLAM handheld dataset.

**What it adds:**
- Static TF publishers for HILTI sensor frames:
  - `base_link` → Hesai PandarXT-32 (LiDAR)
  - `base_link` → `cam0_sensor_frame` (Alphasense Camera)
  - `base_link` → `imu_sensor_frame`
- Approximate calibration values from HILTI-SLAM challenge docs
- Includes the main `s_graphs_launch.py` with HILTI-specific remappings:
  - LiDAR topic: `/hesai/pandar`
  - Camera topic: `/alphasense/cam0/image_raw`
  - IMU topic: `/alphasense/imu`

---

### 9.5 `lidar_situational_graphs/launch/s_graphs_launch.py`

**Major additions:**

New launch arguments declared (10 new arguments):
```python
"use_sim_time"          # default "true" — use /clock for bag playback
"enable_yolo"           # default "true"
"enable_clip"           # default "true"
"visualize_detections"  # default "true"
"confidence_threshold"  # default "0.25"
"process_every_n_frames"# default "5"
"yolo_model_path"       # path to .trt engine
"clip_model_path"       # path to ONNX weights
```

New nodes launched automatically:
```python
iqa_cmd        = ExecuteProcess(cmd=["python3", iqa_script_path])
dynatrack_cmd  = Node(package="tracking_of", executable="of_track_node", ...)
zones_cmd      = ExecuteProcess(cmd=["python3", zones_script_path])
metrics_cmd    = ExecuteProcess(cmd=["python3", metrics_script_path, ...])
```

`use_sim_time` parameter propagated to ALL nodes:
- prefiltering_node
- scan_matching_odometry_node
- room_segmentation / reasoning
- floor_plan_node
- s_graphs_node

The `base_frame` default changed from `"body"` → `"base_link"`.

The `prefix=["bash -c 'exec 2>/dev/null; $0 $@'"]` removed from prefiltering node (stderr suppression removed).

---

## 10. File-by-File Change Log

### Modified C++ Files

| File | Change Type | Lines +/- | Summary |
|------|-------------|-----------|---------|
| `src/s_graphs/common/s_graphs.cpp` | Modified | +670 / -7 | Triple sync, semantic pipeline, IQA/DynaTrack callbacks, new publishers, new params |
| `src/s_graphs/backend/loop_mapper.cpp` | Modified | +460 / -3 | Full semantic weighting pipeline, 5-factor alpha, CSV logger |
| `src/s_graphs/backend/floor_mapper.cpp` | Modified | +17 / -0 | Null-checks for graph_slam and floor node |
| `apps/prefiltering_node.cpp` | Modified | +27 / -10 | Debug logs (commented), error level fix, removed stderr suppressor |
| `include/s_graphs/common/s_graphs.hpp` | Modified | +124 / -16 | Triple sync type, all new members, new callbacks, new methods |
| `include/s_graphs/common/keyframe.hpp` | Modified | +23 / -1 | 10 new optional fields, cv::Mat include |
| `include/s_graphs/backend/loop_mapper.hpp` | Modified | +43 / -5 | 3 new methods, zone_cache_, CSV ofstream, node_ member, #endif fix |
| `include/s_graphs/frontend/loop_detector.hpp` | Modified | +11 / -3 | fitness_score in Loop struct and constructor |

### New C++ Files

| File | Lines | Purpose |
|------|-------|---------|
| `include/s_graphs/common/yolo_object_detector.hpp` | 87 | YOLO TensorRT class declaration |
| `src/s_graphs/common/yolo_object_detector.cpp` | 464 | YOLO TensorRT implementation |
| `include/s_graphs/common/clip_feature_extractor.hpp` | 58 | CLIP ONNXRuntime class declaration |
| `src/s_graphs/common/clip_feature_extractor.cpp` | 176 | CLIP ONNXRuntime implementation |
| `include/s_graphs/common/semantic_evidence.hpp` | 76 | SemanticEvidence aggregation struct |
| `include/s_graphs/frontend/zone_cache.hpp` | 61 | ZoneCache class declaration |
| `src/s_graphs/frontend/zone_cache.cpp` | 138 | ZoneCache implementation |
| `apps/semantic_test_node.cpp` | 422 | Standalone semantic test ROS2 node |

### New Python Files

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/zones_node.py` | 885 | Zone creation + semantic clustering node |
| `scripts/metrics_recorder.py` | 293 | Real-time ROS metrics → 6 CSV files |
| `scripts/analyse_metrics.py` | 898 | Post-run ATE/RPE/semantic analysis tool |
| `scripts/sim_kf_pub.py` | 161 | Synthetic keyframe publisher for testing |
| `scripts/sim_kf_pub.py.back` | 213 | Backup version of sim_kf_pub |
| `lidar_situational_graphs/iqa_ros.py` | 70 | IQA ROS node (installed version) |
| `lidar_situational_graphs/iqm.py` | 96 | IQM computation library (installed version) |
| `lidar_situational_graphs/tf_filter.py` | 102 | TF filter for bag playback (removes conflicting map→odom TF) |
| `IQM_Python/iqa_ros.py` | 66 | IQA node standalone (development version) |
| `IQM_Python/iqm.py` | 96 | IQM library (development version) |
| `IQM_Python/test.py` | 160 | IQM test script |

### New ROS Message Definitions

| File | New? | Content |
|------|------|---------|
| `msg/KeyframeSemantic.msg` | NEW | Per-keyframe semantic data for inter-node communication |
| `msg/Zone.msg` | NEW | Zone create/update/delete messages |
| `msg/FloorData.msg` | NEW | Simplified floor data message |

### Modified Configuration Files

| File | Change | Details |
|------|--------|---------|
| `config/s_graphs.yaml` | Modified | Frame IDs: body→base_link; solver: lm_var_cholmod→lm_pcg; +7 semantic params |
| `config/prefiltering.yaml` | Modified | base_link_frame: body→base_link |
| `package.xml` | Modified | +rosidl_default_runtime, +std_msgs, +cv_bridge, +image_transport, +libopencv-dev |
| `CMakeLists.txt` | Modified | +OpenCV, +CUDA, +TensorRT, +ONNXRuntime, +semantic_test_node build, +Python scripts install |

### New Launch Files

| File | Lines | Purpose |
|------|-------|---------|
| `launch/s_graphs_hilti_launch.py` | 126 | HILTI dataset specific launch with static TFs |
| `launch/semantic_test_launch.py` | 104 | Standalone semantic test node launcher |

### New Docker Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `docker/cuda_humble/Dockerfile` | 457 | CUDA 12.1 + TensorRT + ONNXRuntime + ROS2 Humble |
| `docker/cuda_humble/build_cuda_docker.sh` | 57 | Docker build script |
| `docker/cuda_humble/run_cuda_docker.sh` | 95 | Docker run script with GPU + volume mounts |

### New Test / Development Files

| File | Lines | Purpose |
|------|-------|---------|
| `test_codes/test_clip_standalone.cpp` | 94 | Standalone CLIP test (no ROS) |
| `test_codes/clip_feature_extractor.cpp` | 176 | Copy of CLIP extractor for standalone test |
| `test_codes/clip_feature_extractor.hpp` | 58 | Copy of CLIP header for standalone test |
| `test_codes/Makefile.test` | 45 | Build the standalone CLIP test |
| `test_codes/run_base_ros2_bag.sh` | 42 | Script to run bags in the ROS2 container |
| `test_codes/run_debug_sgraphs.sh` | 69 | Script to rebuild + run with debug output |
| `test_codes/test_image.png` | binary | Test image for CLIP/YOLO validation |

### New Ground Truth / Data Files

| File | Lines | Purpose |
|------|-------|---------|
| `gt/gt6gb.txt` | 689 | Ground truth trajectory for 6GB dataset |

### New Documentation Files

| File | Lines | Purpose |
|------|-------|---------|
| `COMPLETE_WORK_DOCUMENTATION.md` | 572 | Full project documentation |
| `SEMANTIC_INTEGRATION_CHANGES.md` | 204 | Semantic module change log |
| `basement_map/S_GRAPHS_DEBUG_REPORT.md` | 939 | Debug analysis report from basement mapping run |
| `basement_map/TECHNICAL_REFERENCE.md` | 537 | Technical reference for the basement run |

---

## 11. Architecture Evolution Diagram

### BEFORE (`2be157b` — Original S-Graphs)

```
[LiDAR] ──────────────────────→ [Prefiltering Node]
                                        │
                                        ↓ filtered_points
[IMU/GPS] ────────────────────→ [S-Graphs Node]
                                        │
                 ┌──────────── odom ──→ [ApproxSync(2)]
                 │                      │ (odom + cloud)
[Scan Matching] ─┘                      ↓
                              [cloud_callback()]
                                        │
                              [KeyFrame Creation]
                              [Planar Segmentation]
                              [Graph Optimization]
                              [Loop Closure (ICP only)]
```

### AFTER (`a8b6bbb` — Custom Semantic Extension)

```
[LiDAR] ─────────────────────→ [Prefiltering Node]
                                       │ filtered_points
[Camera (Alphasense cam0)] ──────┐     │
                                 │     ↓
[Scan Matching Odometry] ────→ [ApproxSync(3)] ← image
                                       │ (odom + cloud + image)
                                       ↓
                              [cloud_image_odom_callback()]
                                       │
                          ┌────────────┴────────────────┐
                          ↓                             ↓
               [KeyFrame Creation]            [std::thread]
               [image_rgb attached]                    │
               [dynamicity attached]                   ↓
               [Planar Segmentation]         [processSemantic()]
               [Graph Optimization]                    │
                          │                    ┌───────┴───────┐
                          │                    ↓               ↓
                          │             [YOLO TensorRT]  [CLIP ONNXRuntime]
                          │                    │               │
                          │                    ↓               ↓
                          │             [detected_objects] [clip_embedding]
                          │             stored in KeyFrame  stored in KeyFrame
                          │
                 [Loop Closure]
                          │
               [LoopMapper::add_loops()]
                          │
     ┌────────────────────┼────────────────────────────────┐
     │ Semantic Alpha Computation                          │
     │                                                     │
     │  CLIP Similarity ──→ base alpha                     │
     │  Object Overlap ───→ ± 15% adjustment               │
     │  ICP Fitness ──────→ geometry-semantic damping      │
     │  IQA Score ────────→ quality gate blending          │
     │  DynaTrack Dyn ───→ up to 30% penalty              │
     │  Zone Cache ──────→ up to ±50% zone factor         │
     │                                                     │
     │  information_matrix *= semantic_alpha               │
     └────────────────────────────────────────────────────┘
                          │
               [Edge Added to g2o Graph]
               [CSV Row Written to /tmp/sgraphs_metrics/loop_metrics.csv]

━━━━━━━━━━━━━━━━━━━━━━ Supporting Nodes ━━━━━━━━━━━━━━━━━━━━━━━━

[IQA Python Node]         → /image_quality_score (Float64)
  └── Brightness, contrast, entropy, gradient, noise
  └── Sigmoid-normalized to [0,1]
  └── Stored in keyframe->image_brightness

[DynaTrack Node]          → dynamic_objects (DynamicObjects)
  └── BEV optical flow on filtered LiDAR
  └── scene_dynamicity [0,1] + num_dynamic_clusters
  └── Stored in keyframe->scene_dynamicity

[Zones Python Node]       → zones (Zone messages)
  └── Subscribes to: KeyframeSemantic + RoomsData + GraphKeyframes
  └── OBJECT_TO_LABEL mapping + shapely spatial clustering
  └── ZoneCache (C++) receives Zone messages → feeds loop_mapper

[Metrics Recorder]        → /tmp/sgraphs_metrics/*.csv
  └── 6 CSV files: trajectory, odom, keyframe semantics,
      dynamic objects, zone events, image quality

[Analyse Metrics]         → metrics_summary.txt + PNG plots
  └── ATE, RPE, loop closure analysis, semantic coverage
```

---

## 12. New Data Structures

### `Detection` (C++ struct, `yolo_object_detector.hpp`)
```cpp
struct Detection {
    float x1, y1, x2, y2;   // bounding box corners
    float confidence;
    int class_id;
    std::string class_name;
};
```

### `SemanticEvidence` (C++ struct, `semantic_evidence.hpp`)
Full multi-modal evidence aggregation for a keyframe location — 35+ fields covering LiDAR geometry, YOLO detections, CLIP embeddings, image quality, and fusion outputs.

### `ZoneCache` (C++ class, `zone_cache.hpp`)
Thread-safe runtime cache mapping `keyframe_id → zone_id` and `zone_id → {confidence, keyframes, support_planes, labels, floor}`.

### `KeyFrame` extended fields (20 new `boost::optional<>` fields)
See Section 4.2 for the complete list.

### `Loop` extended (1 new field)
`double fitness_score` — ICP fitness score from scan matching registration.

---

## 13. New ROS Messages

### `KeyframeSemantic.msg`
```
std_msgs/Header header
int32 keyframe_id
int32 keyframe_seq
builtin_interfaces/Time stamp
int32 floor_id
int32 room_id
string[] objects
float32[] object_confidence
float32 scene_confidence
string model_name
geometry_msgs/Pose pose
```
**Purpose:** Published by `processSemantic()` → consumed by `zones_node.py` and `metrics_recorder.py`.

### `Zone.msg`
```
std_msgs/Header header
int32 zone_id
int32 floor_id
geometry_msgs/Pose centroid
geometry_msgs/Polygon polygon
int32[] room_ids
int32[] keyframe_ids
string[] top_labels
float32[] top_label_confidences
float32 confidence
situational_graphs_msgs/PlaneData[] supporting_planes
int32[] supporting_wall_ids
uint8 action       # 0=CREATE, 1=UPDATE, 2=DELETE
uint64 version
```
**Purpose:** Published by `zones_node.py` → consumed by `ZoneCache` (C++) → used by `loop_mapper.cpp`.

### `FloorData.msg` (local definition)
```
std_msgs/Header header
int32 id
geometry_msgs/Pose floor_center
int32[] keyframe_ids
```

---

## 14. New ROS Parameters

### Added to `s_graphs_node`:

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `yolo_model_path` | string | `""` | Path to TensorRT `.trt` engine file |
| `clip_model_path` | string | `""` | Path to ONNX visual encoder weights |
| `confidence_threshold` | double | `0.25` | YOLO detection confidence threshold |
| `process_every_n_frames` | int | `5` | Process semantic info every N keyframes |
| `enable_yolo` | bool | `true` | Toggle YOLO detection |
| `enable_clip` | bool | `true` | Toggle CLIP embedding extraction |
| `visualize_detections` | bool | `true` | Publish detection image visualization |
| `use_zone_prefilter` | bool | `true` | Enable zone-based loop closure prefiltering |
| `zone_confidence_threshold` | double | `0.3` | Min zone confidence to use zone factor |

### Modified in `s_graphs_node`:

| Parameter | Old Default | New Default | Reason |
|-----------|-------------|-------------|--------|
| `g2o_solver_type` | `"lm_var_cholmod"` | `"lm_pcg"` | No CHOLMOD dependency in CUDA container |

---

## 15. New Nodes & Scripts

| Node/Script | Type | Language | Executable | Purpose |
|-------------|------|----------|------------|---------|
| `s_graphs_node` | ROS2 node | C++ | `s_graphs_node` | Extended with camera + semantic (existing) |
| `semantic_test_node` | ROS2 node | C++ | `semantic_test_node` | Standalone YOLO+CLIP test harness |
| `iqa_ros.py` | ROS2 node | Python | `iqa_ros.py` | Image Quality Assessment publisher |
| `of_track_node` | ROS2 node | C++ (submodule) | `of_track_node` | DynaTrack BEV optical flow |
| `zones_node.py` | ROS2 node | Python | `zones_node.py` | Zone creation and management |
| `metrics_recorder.py` | ROS2 node | Python | `metrics_recorder.py` | Real-time CSV metrics recording |
| `analyse_metrics.py` | CLI tool | Python | `analyse_metrics.py` | Post-run metrics analysis + plots |
| `sim_kf_pub.py` | ROS2 node | Python | N/A | Synthetic keyframe publisher for testing |
| `tf_filter.py` | ROS2 node | Python | `tf_filter.py` | Filter conflicting TF transforms from bags |

---

## 16. Build System Changes

### `CMakeLists.txt`

**Dependencies added:**
```cmake
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT:
find_path(TENSORRT_INCLUDE_DIR NvInfer.h ...)
find_library(TENSORRT_LIBRARY nvinfer ...)
find_library(TENSORRT_ONNXPARSER nvonnxparser ...)

# ONNXRuntime:
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h ...)
find_library(ONNXRUNTIME_LIBRARY onnxruntime ...)
```

**Conditional compilation:**
```cmake
# If TensorRT OR ONNXRuntime not found:
list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "clip_feature_extractor\.cpp$")
list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "yolo_object_detector\.cpp$")
# → Core library compiles without semantic feature files (graceful degradation)
```

**New executable:**
```cmake
if(TENSORRT_LIBRARY AND ONNXRUNTIME_LIBRARY)
  add_executable(semantic_test_node ...)
  # Links: TensorRT, ONNXRuntime, CUDA, OpenCV
endif()
```

**Existing `s_graphs_node` gets new link libraries:**
```cmake
target_link_libraries(s_graphs_node
  ${PROJECT_NAME}_core_lib
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY}
  ${TENSORRT_ONNXPARSER}
  ${ONNXRUNTIME_LIBRARY}
  ${OpenCV_LIBRARIES}
)
```

**Python scripts installed:**
```cmake
install(PROGRAMS
  ${PROJECT_NAME}/tf_filter.py
  ${PROJECT_NAME}/iqa_ros.py
  ${PROJECT_NAME}/iqm.py
  scripts/zones_node.py
  scripts/metrics_recorder.py
  scripts/analyse_metrics.py
  DESTINATION lib/${PROJECT_NAME}
)
```

**Fixed:** `rosidl_default_generators` removed from `find_package()` and replaced with `rosidl_default_runtime` in `package.xml`. `ament_export_dependencies()` now uses the full `${DEPENDENCIES}` variable instead of only `rosidl_default_runtime`.

### `package.xml`

Added dependencies:
```xml
<depend>rosidl_default_runtime</depend>
<depend>std_msgs</depend>
<depend>cv_bridge</depend>
<depend>image_transport</depend>
<depend>libopencv-dev</depend>
```

---

## 17. Infrastructure Changes

### New `.gitignore`
```
build/
install/
log/
__pycache__/
*.pyc
*.onnx    ← model files NOT committed
*.trt     ← engine files NOT committed
/bags/
/documentation/
/final_architecture/
/code_zone/
```

### New CUDA Docker Environment (`docker/cuda_humble/`)

Three files creating a complete CUDA-capable ROS2 development environment:

**`Dockerfile` (457 lines):** (currently commented out — template form)
- Base: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- Installs: ROS2 Humble desktop, TensorRT 8.6+/10.x, ONNXRuntime GPU, OpenCV, Python dependencies (numpy, scipy, shapely, cv_bridge), all S-Graphs dependencies
- Environment variables for CUDA 12.1 and NVIDIA runtime

**`build_cuda_docker.sh`:** Builds the Docker image with `--gpus all` capability

**`run_cuda_docker.sh`:** Runs with volume mounts:
- `/bags/` for ROS2 bag files
- `/models/` for `.trt` and `.onnx` model files
- X11 socket for RViz display

### New IQM_Python Development Folder

Contains the standalone development versions of:
- `iqa_ros.py` + `iqm.py` — IQA implementation
- `test.py` — test harness
- 17 BMP/JPEG test images for IQM validation (I01 family + 5 JPEG images)

---

## 18. What Was NOT Changed (Original S-Graphs Core)

These files from the original authors remained **completely untouched**:

- `src/s_graphs/backend/graph_slam.cpp` — g2o graph management
- `src/s_graphs/backend/plane_mapper.cpp` — planar surface mapping
- `src/s_graphs/backend/gps_mapper.cpp` — GPS integration
- `src/s_graphs/backend/imu_mapper.cpp` — IMU integration
- `src/s_graphs/backend/keyframe_mapper.cpp` — keyframe graph insertion
- `src/s_graphs/backend/room_graph_generator.cpp` — local room graph
- `src/s_graphs/frontend/keyframe_updater.cpp` — keyframe selection
- `src/s_graphs/frontend/plane_analyzer.cpp` — plane extraction
- `src/s_graphs/visualization/graph_visualizer.cpp` — RViz markers
- `src/s_graphs/visualization/graph_publisher.cpp` — graph topic publishing
- All g2o edge implementations (`edge_*.cpp`)
- All test files (`testPlane`, `testRoom`, `testRoomCentreCompute`)
- `apps/s_graphs_node.cpp` — main entry point (unchanged)
- `apps/scan_matching_odometry_node.cpp` — odometry unchanged
- `apps/floor_plan_node.cpp` — floor planning unchanged
- `apps/room_segmentation_node.cpp` — room segmentation unchanged
- All original `docker/humble/` and `docker/foxy_noetic/` Dockerfiles
- Original `README.md`, `CHANGELOG.rst`, `LICENSE`

---

## 19. Complete Summary Table

| Category | Item | Status | File Location |
|----------|------|--------|---------------|
| **Synchronization** | Dual sync (odom+cloud) → Triple sync (odom+cloud+image) | Changed | `s_graphs.hpp`, `s_graphs.cpp` |
| **Camera Input** | RGB image subscription (`/alphasense/cam0/image_raw`) | Added | `s_graphs.cpp` L162 |
| **Keyframe Data** | RGB image attached to each keyframe | Added | `keyframe.hpp`, `s_graphs.cpp` |
| **Keyframe Data** | CLIP 512-dim embedding per keyframe | Added | `keyframe.hpp` |
| **Keyframe Data** | YOLO detected objects per keyframe | Added | `keyframe.hpp` |
| **Keyframe Data** | IQA score per keyframe | Added | `keyframe.hpp` |
| **Keyframe Data** | DynaTrack dynamicity per keyframe | Added | `keyframe.hpp` |
| **Object Detection** | YOLO-World TensorRT class | New | `yolo_object_detector.{hpp,cpp}` |
| **Feature Extraction** | CLIP ONNXRuntime visual encoder class | New | `clip_feature_extractor.{hpp,cpp}` |
| **Semantic Evidence** | Multi-modal evidence aggregation struct | New | `semantic_evidence.hpp` |
| **Image Quality** | IQA Python node (brightness+contrast+entropy+gradient-noise) | New | `iqa_ros.py`, `iqm.py` |
| **Dynamic Tracking** | DynaTrack BEV optical flow submodule | New | `tracking_OF/` (submodule) |
| **Zone System** | Zone creation + semantic clustering | New | `zones_node.py` |
| **Zone System** | ZoneCache C++ class | New | `zone_cache.{hpp,cpp}` |
| **Zone System** | Zone ROS message | New | `msg/Zone.msg` |
| **Loop Closure** | ICP fitness score in Loop struct | Modified | `loop_detector.hpp` |
| **Loop Closure** | CLIP cosine similarity computation | New | `loop_mapper.cpp` |
| **Loop Closure** | Object Jaccard overlap computation | New | `loop_mapper.cpp` |
| **Loop Closure** | Geometry-semantic interaction damping | New | `loop_mapper.cpp` |
| **Loop Closure** | IQA quality gate blending | New | `loop_mapper.cpp` |
| **Loop Closure** | DynaTrack dynamicity penalty factor | New | `loop_mapper.cpp` |
| **Loop Closure** | Zone coherence boost/penalty factor | New | `loop_mapper.cpp` |
| **Loop Closure** | Per-loop CSV metrics logging | New | `loop_mapper.cpp` |
| **Semantic Messages** | KeyframeSemantic ROS message | New | `msg/KeyframeSemantic.msg` |
| **Metrics** | Real-time 6-CSV metrics recorder | New | `metrics_recorder.py` |
| **Metrics** | Post-run ATE/RPE/semantic analyzer | New | `analyse_metrics.py` |
| **Testing** | Semantic test node (YOLO+CLIP standalone) | New | `semantic_test_node.cpp` |
| **Testing** | Standalone CLIP test (no ROS) | New | `test_codes/test_clip_standalone.cpp` |
| **Testing** | Synthetic keyframe publisher | New | `sim_kf_pub.py` |
| **Infrastructure** | TF filter node for bag playback | New | `tf_filter.py` |
| **Infrastructure** | CUDA 12.1 + TensorRT + ONNXRuntime Docker | New | `docker/cuda_humble/` |
| **Infrastructure** | HILTI dataset launch file | New | `s_graphs_hilti_launch.py` |
| **Infrastructure** | .gitignore | New | `.gitignore` |
| **Build System** | CUDA, TensorRT, ONNXRuntime CMake detection | Added | `CMakeLists.txt` |
| **Build System** | Conditional semantic file compilation | Added | `CMakeLists.txt` |
| **Build System** | semantic_test_node conditional build target | Added | `CMakeLists.txt` |
| **Build System** | Python scripts installation | Added | `CMakeLists.txt` |
| **Config** | Frame IDs: body → base_link | Changed | `s_graphs.yaml`, `prefiltering.yaml`, `s_graphs_launch.py` |
| **Config** | g2o solver: lm_var_cholmod → lm_pcg | Changed | `s_graphs.yaml` |
| **Config** | +7 semantic parameters in YAML | Added | `s_graphs.yaml` |
| **Bug Fix** | Null-check in floor_mapper.cpp | Fixed | `floor_mapper.cpp` |
| **Bug Fix** | #endif guard fix in loop_mapper.hpp | Fixed | `loop_mapper.hpp` |
| **Bug Fix** | TF error log level INFO→ERROR in prefiltering | Fixed | `prefiltering_node.cpp` |
| **Bug Fix** | Removed stderr suppression (`exec 2>/dev/null`) | Fixed | `s_graphs_launch.py` |
| **Ground Truth** | 6GB dataset ground truth trajectory | Added | `gt/gt6gb.txt` |
| **Documentation** | COMPLETE_WORK_DOCUMENTATION.md | New | root |
| **Documentation** | SEMANTIC_INTEGRATION_CHANGES.md | New | root |
| **Documentation** | S_GRAPHS_DEBUG_REPORT.md | New | `basement_map/` |
| **Documentation** | TECHNICAL_REFERENCE.md | New | `basement_map/` |

---

*Analysis generated from: `git diff 2be157b..a8b6bbb` across 82 changed files, 10,671 added lines, 167 deleted lines. All code paths verified against actual source.*
