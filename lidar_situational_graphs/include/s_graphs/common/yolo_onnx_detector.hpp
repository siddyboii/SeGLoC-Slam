// CPU-only YOLO detector using ONNXRuntime
// Replaces YOLOWorldTensorRT when CUDA/TensorRT is unavailable
// Same Detection struct and infer() API for drop-in replacement
#ifndef S_GRAPHS_YOLO_ONNX_DETECTOR_HPP
#define S_GRAPHS_YOLO_ONNX_DETECTOR_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <array>
#include <chrono>
#include <algorithm>
#include <string>
#include <numeric>

// Reuse the same Detection struct for API compatibility
#ifndef S_GRAPHS_DETECTION_STRUCT
#define S_GRAPHS_DETECTION_STRUCT
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;       // Detection confidence
    int class_id;          // Class index
    std::string class_name;
};
#endif

class YOLOOnnxDetector
{
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // Model metadata
    std::string input_name_;
    std::vector<std::string> output_names_;
    int batch_size_;
    int channels_;
    int height_;
    int width_;
    int num_anchors_;
    int num_classes_;

    // Image preprocessing state
    cv::Mat original_image_;
    int original_height_;
    int original_width_;
    float pad_x_;
    float pad_y_;
    float scale_;

    std::vector<std::string> class_names_ {"Door", "Cupboard", "Desk", "Monitor", "Chair", "Wire", "Pillar", "Tree", "Table", " "};

public:
    explicit YOLOOnnxDetector(const std::string& onnx_model_path);
    ~YOLOOnnxDetector() = default;

    std::vector<Detection> infer(const std::string& image_path, float confidence_threshold);
    std::vector<Detection> infer(const cv::Mat& image, float confidence_threshold);

    cv::Mat getOriginalImage() const { return original_image_; }

    // Allow setting class names (e.g. from ROS params)
    void setClassNames(const std::vector<std::string>& names) { class_names_ = names; }

private:
    std::vector<float> preprocessImage(const std::string& image_path);
    std::vector<float> preprocessImage(const cv::Mat& image);
    std::vector<float> preprocessCommon(const cv::Mat& image);
    std::vector<Detection> postprocessOutputs(const std::vector<Ort::Value>& outputs,
                                               float confidence_threshold);
    float calculateIoU(const Detection& det1, const Detection& det2);
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections,
                                     float iou_threshold);
};

#endif // S_GRAPHS_YOLO_ONNX_DETECTOR_HPP
