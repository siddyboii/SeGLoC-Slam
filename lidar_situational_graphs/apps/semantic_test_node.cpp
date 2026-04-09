/*
 * Semantic Test Harness Node
 * 
 * Purpose: Open-ended test harness to verify camera integration with YOLO object
 * detector and CLIP feature extractor. This node subscribes to camera images,
 * runs both models on keyframes, and publishes visualization for immediate verification.
 * 
 * This is NOT a production node - it's designed for testing and validation of the
 * camera → detection/feature extraction → output data flow.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <mutex>

// Include the YOLO and CLIP extractors
#include <s_graphs/common/yolo_object_detector.hpp>
#include <s_graphs/common/clip_feature_extractor.hpp>

namespace s_graphs {

class SemanticTestNode : public rclcpp::Node {
public:
    SemanticTestNode() : Node("semantic_test_node") {
        // Declare parameters with defaults
        // Rationale: Parameters allow runtime configuration without recompilation,
        // essential for testing different model paths and thresholds
        this->declare_parameter("yolo_model_path", "");
        this->declare_parameter("clip_model_path", "");
        this->declare_parameter("confidence_threshold", 0.25);
        this->declare_parameter("image_topic", "/camera/color/image_raw");
        this->declare_parameter("process_every_n_frames", 5);  // Skip frames for real-time performance
        this->declare_parameter("enable_yolo", true);
        this->declare_parameter("enable_clip", true);
        this->declare_parameter("visualize_detections", true);

        // Load parameters
        yolo_model_path_ = this->get_parameter("yolo_model_path").as_string();
        clip_model_path_ = this->get_parameter("clip_model_path").as_string();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        image_topic_ = this->get_parameter("image_topic").as_string();
        process_every_n_frames_ = this->get_parameter("process_every_n_frames").as_int();
        enable_yolo_ = this->get_parameter("enable_yolo").as_bool();
        enable_clip_ = this->get_parameter("enable_clip").as_bool();
        visualize_detections_ = this->get_parameter("visualize_detections").as_bool();

        // Initialize models
        // Rationale: Models are initialized once at startup to avoid repeated loading overhead
        initializeModels();

        // Create subscriber for camera images
        // Rationale: Using raw subscription instead of image_transport for simplicity in test harness
        // QoS profile with BEST_EFFORT for real-time camera streams
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_,
            qos,
            std::bind(&SemanticTestNode::imageCallback, this, std::placeholders::_1));

        // Create publishers for visualization
        // Rationale: Separate topics for different outputs allow selective visualization in RViz
        detection_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "semantic_test/detection_image", 10);
        detection_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "semantic_test/detection_markers", 10);

        // Timer for status logging
        // Rationale: Periodic status logging helps verify the node is running and processing
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&SemanticTestNode::statusCallback, this));

        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "  Semantic Test Harness Initialized");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "  Image topic: %s", image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  YOLO enabled: %s", enable_yolo_ ? "YES" : "NO");
        RCLCPP_INFO(this->get_logger(), "  CLIP enabled: %s", enable_clip_ ? "YES" : "NO");
        RCLCPP_INFO(this->get_logger(), "  Process every %d frames", process_every_n_frames_);
        RCLCPP_INFO(this->get_logger(), "  Confidence threshold: %.2f", confidence_threshold_);
        RCLCPP_INFO(this->get_logger(), "========================================");
    }

    ~SemanticTestNode() {
        RCLCPP_INFO(this->get_logger(), "Semantic Test Node shutting down...");
        RCLCPP_INFO(this->get_logger(), "Total frames received: %ld", frame_count_);
        RCLCPP_INFO(this->get_logger(), "Total frames processed: %ld", processed_count_);
    }

private:
    void initializeModels() {
        // Initialize YOLO detector
        // Rationale: Wrapped in try-catch to provide clear error messages if model loading fails
        if (enable_yolo_) {
            if (yolo_model_path_.empty()) {
                RCLCPP_ERROR(this->get_logger(), 
                    "YOLO model path not set! Set 'yolo_model_path' parameter.");
                enable_yolo_ = false;
            } else {
                try {
                    RCLCPP_INFO(this->get_logger(), "Loading YOLO model from: %s", yolo_model_path_.c_str());
                    yolo_detector_ = std::make_unique<YOLOWorldTensorRT>(yolo_model_path_);
                    RCLCPP_INFO(this->get_logger(), "✓ YOLO model loaded successfully!");
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to load YOLO model: %s", e.what());
                    enable_yolo_ = false;
                }
            }
        }

        // Initialize CLIP extractor
        // Rationale: CLIP is optional - if path not set, only YOLO runs
        if (enable_clip_) {
            if (clip_model_path_.empty()) {
                RCLCPP_ERROR(this->get_logger(), 
                    "CLIP model path not set! Set 'clip_model_path' parameter.");
                enable_clip_ = false;
            } else {
                try {
                    RCLCPP_INFO(this->get_logger(), "Loading CLIP model from: %s", clip_model_path_.c_str());
                    clip_extractor_ = std::make_unique<ClipFeatureExtractor>(clip_model_path_);
                    RCLCPP_INFO(this->get_logger(), "✓ CLIP model loaded successfully!");
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to load CLIP model: %s", e.what());
                    enable_clip_ = false;
                }
            }
        }

        if (!enable_yolo_ && !enable_clip_) {
            RCLCPP_WARN(this->get_logger(), 
                "Neither YOLO nor CLIP is enabled! Node will only count frames.");
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        frame_count_++;

        // Frame skipping for real-time performance
        // Rationale: Processing every frame at 30fps would overload the GPU;
        // skipping frames maintains real-time performance while still testing the pipeline
        if (frame_count_ % process_every_n_frames_ != 0) {
            return;
        }

        // Convert ROS Image to OpenCV Mat
        // Rationale: cv_bridge handles encoding conversion and memory management
        cv::Mat cv_image;
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            cv_image = cv_ptr->image.clone();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        if (cv_image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty image!");
            return;
        }

        processed_count_++;
        auto start_time = std::chrono::high_resolution_clock::now();

        RCLCPP_INFO(this->get_logger(), "");
        RCLCPP_INFO(this->get_logger(), "=== Processing Frame %ld (Image: %dx%d) ===",
                    processed_count_, cv_image.cols, cv_image.rows);

        // Run YOLO detection
        // Rationale: YOLO provides object-level semantic information (what objects are present)
        std::vector<Detection> detections;
        if (enable_yolo_ && yolo_detector_) {
            auto yolo_start = std::chrono::high_resolution_clock::now();
            
            try {
                detections = yolo_detector_->infer(cv_image, confidence_threshold_);
                
                auto yolo_end = std::chrono::high_resolution_clock::now();
                auto yolo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    yolo_end - yolo_start).count();

                RCLCPP_INFO(this->get_logger(), "[YOLO] Inference time: %ld ms", yolo_duration);
                RCLCPP_INFO(this->get_logger(), "[YOLO] Detections found: %zu", detections.size());

                // Log each detection for verification
                for (size_t i = 0; i < detections.size(); i++) {
                    const auto& det = detections[i];
                    RCLCPP_INFO(this->get_logger(), 
                        "  [%zu] Class: %s (id=%d), Conf: %.2f, BBox: [%.1f, %.1f, %.1f, %.1f]",
                        i, det.class_name.c_str(), det.class_id, det.confidence,
                        det.x1, det.y1, det.x2, det.y2);
                }

                // Publish visualization if enabled
                if (visualize_detections_ && !detections.empty()) {
                    publishDetectionVisualization(cv_image, detections, msg->header);
                }

            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[YOLO] Inference failed: %s", e.what());
            }
        }

        // Run CLIP feature extraction
        // Rationale: CLIP provides scene-level embedding that can be used for
        // semantic similarity computation in loop closure
        if (enable_clip_ && clip_extractor_) {
            auto clip_start = std::chrono::high_resolution_clock::now();
            
            try {
                std::vector<float> embedding = clip_extractor_->extract_embedding(cv_image);
                
                auto clip_end = std::chrono::high_resolution_clock::now();
                auto clip_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    clip_end - clip_start).count();

                RCLCPP_INFO(this->get_logger(), "[CLIP] Inference time: %ld ms", clip_duration);
                
                if (!embedding.empty()) {
                    // Log embedding statistics for verification
                    // Rationale: Instead of printing the full 512-dim vector, we log statistics
                    // that verify the embedding is valid (non-zero, normalized)
                    float sum = 0.0f, min_val = embedding[0], max_val = embedding[0];
                    for (float v : embedding) {
                        sum += v * v;  // L2 norm squared
                        min_val = std::min(min_val, v);
                        max_val = std::max(max_val, v);
                    }
                    float norm = std::sqrt(sum);

                    RCLCPP_INFO(this->get_logger(), "[CLIP] Embedding dimension: %zu", embedding.size());
                    RCLCPP_INFO(this->get_logger(), "[CLIP] Embedding L2 norm: %.4f (should be ~1.0 if normalized)", norm);
                    RCLCPP_INFO(this->get_logger(), "[CLIP] Embedding range: [%.4f, %.4f]", min_val, max_val);
                    RCLCPP_INFO(this->get_logger(), "[CLIP] First 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]",
                        embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]);
                    
                    // Store for similarity comparison with next frame
                    if (!last_embedding_.empty()) {
                        float similarity = clip_extractor_->compute_similarity(last_embedding_, embedding);
                        RCLCPP_INFO(this->get_logger(), 
                            "[CLIP] Similarity with previous frame: %.4f", similarity);
                    }
                    last_embedding_ = embedding;
                    
                    RCLCPP_INFO(this->get_logger(), "[CLIP] ✓ Feature extraction SUCCESSFUL");
                } else {
                    RCLCPP_WARN(this->get_logger(), "[CLIP] Empty embedding returned!");
                }

            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[CLIP] Inference failed: %s", e.what());
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        RCLCPP_INFO(this->get_logger(), "=== Total processing time: %ld ms ===", total_duration);
    }

    void publishDetectionVisualization(const cv::Mat& image, 
                                       const std::vector<Detection>& detections,
                                       const std_msgs::msg::Header& header) {
        // Create visualization image with bounding boxes
        // Rationale: Visual feedback is essential for verifying detection quality
        cv::Mat vis_image = image.clone();

        // Color palette for different classes
        std::vector<cv::Scalar> colors = {
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(255, 0, 0),    // Blue
            cv::Scalar(0, 0, 255),    // Red
            cv::Scalar(255, 255, 0),  // Cyan
            cv::Scalar(255, 0, 255),  // Magenta
            cv::Scalar(0, 255, 255),  // Yellow
            cv::Scalar(128, 128, 0),  // Teal
            cv::Scalar(128, 0, 128),  // Purple
        };

        for (const auto& det : detections) {
            cv::Scalar color = colors[det.class_id % colors.size()];
            
            // Draw bounding box
            cv::rectangle(vis_image, 
                         cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                         cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                         color, 2);

            // Draw label background
            std::string label = det.class_name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(vis_image,
                         cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - text_size.height - 5),
                         cv::Point(static_cast<int>(det.x1) + text_size.width, static_cast<int>(det.y1)),
                         color, -1);

            // Draw label text
            cv::putText(vis_image, label,
                       cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // Add frame info overlay
        std::string info = "Frame: " + std::to_string(processed_count_) + 
                          " | Detections: " + std::to_string(detections.size());
        cv::putText(vis_image, info, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Publish detection image
        // Rationale: Publishing as ROS Image allows viewing in rqt_image_view or RViz
        auto out_msg = cv_bridge::CvImage(header, "bgr8", vis_image).toImageMsg();
        detection_image_pub_->publish(*out_msg);

        // Also publish as MarkerArray for 3D visualization in RViz
        // Rationale: Markers can be projected into 3D space if camera info is available
        publishDetectionMarkers(detections, header);
    }

    void publishDetectionMarkers(const std::vector<Detection>& detections,
                                 const std_msgs::msg::Header& header) {
        visualization_msgs::msg::MarkerArray marker_array;

        // Delete all previous markers
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header = header;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];

            visualization_msgs::msg::Marker marker;
            marker.header = header;
            marker.ns = "yolo_detections";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // Position text at center of bounding box (in image coordinates)
            // Note: These are 2D coordinates - for proper 3D visualization,
            // you'd need camera intrinsics and depth information
            marker.pose.position.x = (det.x1 + det.x2) / 2.0 / 100.0;  // Scale for visibility
            marker.pose.position.y = (det.y1 + det.y2) / 2.0 / 100.0;
            marker.pose.position.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.z = 0.3;
            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 1.0f;

            marker.text = det.class_name + " (" + 
                         std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
            
            marker.lifetime = rclcpp::Duration::from_seconds(1.0);

            marker_array.markers.push_back(marker);
        }

        detection_markers_pub_->publish(marker_array);
    }

    void statusCallback() {
        // Periodic status logging
        // Rationale: Helps identify if the node is receiving data and processing correctly
        RCLCPP_INFO(this->get_logger(), 
            "[STATUS] Frames received: %ld | Processed: %ld | Rate: %.1f fps (processed)",
            frame_count_, processed_count_,
            processed_count_ > 0 ? processed_count_ / 5.0 : 0.0);
    }

    // Parameters
    std::string yolo_model_path_;
    std::string clip_model_path_;
    std::string image_topic_;
    double confidence_threshold_;
    int process_every_n_frames_;
    bool enable_yolo_;
    bool enable_clip_;
    bool visualize_detections_;

    // Models
    std::unique_ptr<YOLOWorldTensorRT> yolo_detector_;
    std::unique_ptr<ClipFeatureExtractor> clip_extractor_;

    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_image_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detection_markers_pub_;
    rclcpp::TimerBase::SharedPtr status_timer_;

    // State
    long frame_count_ = 0;
    long processed_count_ = 0;
    std::vector<float> last_embedding_;  // For computing inter-frame similarity
};

}  // namespace s_graphs

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<s_graphs::SemanticTestNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting Semantic Test Node...");
    RCLCPP_INFO(node->get_logger(), "Waiting for images on configured topic...");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}
