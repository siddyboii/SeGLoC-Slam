// CPU-only YOLO detector implementation using ONNXRuntime
// This is a drop-in replacement for YOLOWorldTensorRT that runs on CPU.
// It uses ONNXRuntime to load and run an ONNX-format YOLO model.
//
// The ONNX model is expected to have:
//   Input:  "images" -> [1, 3, H, W] (float32, normalized 0-1)
//   Output: "scores" -> [1, num_anchors, num_classes]
//           "boxes"  -> [1, num_anchors, 4] (x1,y1,x2,y2 or cx,cy,w,h)

#include <s_graphs/common/yolo_onnx_detector.hpp>
#include <rclcpp/rclcpp.hpp>

// =====================================================================
// IoU + NMS (identical to TensorRT version)
// =====================================================================

float YOLOOnnxDetector::calculateIoU(const Detection& det1, const Detection& det2) {
    float x_left   = std::max(det1.x1, det2.x1);
    float y_top    = std::max(det1.y1, det2.y1);
    float x_right  = std::min(det1.x2, det2.x2);
    float y_bottom = std::min(det1.y2, det2.y2);

    float intersection = std::max(0.0f, x_right - x_left) *
                         std::max(0.0f, y_bottom - y_top);

    float area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1);
    float area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1);
    float union_area = area1 + area2 - intersection;

    return (union_area <= 0) ? 0.0f : intersection / union_area;
}

std::vector<Detection> YOLOOnnxDetector::applyNMS(
    const std::vector<Detection>& dets, float iou_threshold) {
    std::vector<Detection> sorted_dets = dets;
    std::sort(sorted_dets.begin(), sorted_dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < sorted_dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(sorted_dets[i]);
        for (size_t j = i + 1; j < sorted_dets.size(); j++) {
            if (suppressed[j]) continue;
            if (sorted_dets[i].class_id == sorted_dets[j].class_id) {
                if (calculateIoU(sorted_dets[i], sorted_dets[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return result;
}

// =====================================================================
// Constructor — load ONNX model
// =====================================================================

YOLOOnnxDetector::YOLOOnnxDetector(const std::string& onnx_model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOOnnxDetector") {

    std::cout << "\n✓ Loading ONNX YOLO model from: " << onnx_model_path << std::endl;

    // Verify file exists
    std::ifstream f(onnx_model_path);
    if (!f.good()) {
        throw std::runtime_error("ONNX model file not found: " + onnx_model_path);
    }
    f.close();

    // Configure session for CPU-only
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(
        env_, onnx_model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;

    // ── Input info ──
    size_t num_inputs = session_->GetInputCount();
    if (num_inputs < 1) {
        throw std::runtime_error("ONNX model has no inputs");
    }
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name_ptr.get();

    auto input_shape = session_->GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
    batch_size_ = static_cast<int>(input_shape[0]);
    channels_   = static_cast<int>(input_shape[1]);
    height_     = static_cast<int>(input_shape[2]);
    width_      = static_cast<int>(input_shape[3]);

    // Handle dynamic dims (default to 640)
    if (height_ <= 0) height_ = 640;
    if (width_  <= 0) width_  = 640;
    if (batch_size_ <= 0) batch_size_ = 1;

    std::cout << "  Input: " << input_name_ << " ["
              << batch_size_ << "x" << channels_ << "x"
              << height_ << "x" << width_ << "]" << std::endl;

    // ── Output info ──
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
        std::string name = name_ptr.get();
        output_names_.push_back(name);

        auto shape = session_->GetOutputTypeInfo(i)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();

        std::cout << "  Output[" << i << "]: " << name << " [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Detect scores vs boxes by shape
        // scores: [1, num_anchors, num_classes]  (3D, last dim = num_classes)
        // boxes:  [1, num_anchors, 4]            (3D, last dim = 4)
        if (shape.size() == 3 && shape[2] != 4) {
            num_anchors_ = static_cast<int>(shape[1]);
            num_classes_ = static_cast<int>(shape[2]);
        }
    }

    if (num_anchors_ <= 0 || num_classes_ <= 0) {
        // Fallback: try to detect from the outputs in reverse order
        for (int i = static_cast<int>(num_outputs) - 1; i >= 0; i--) {
            auto shape = session_->GetOutputTypeInfo(i)
                             .GetTensorTypeAndShapeInfo()
                             .GetShape();
            if (shape.size() == 3 && shape[2] > 4) {
                num_anchors_ = static_cast<int>(shape[1]);
                num_classes_ = static_cast<int>(shape[2]);
                break;
            }
        }
    }

    std::cout << "  Anchors: " << num_anchors_
              << "  Classes: " << num_classes_ << std::endl;
    std::cout << "✓ ONNX YOLO model loaded successfully (CPU mode)!" << std::endl;
}

// =====================================================================
// Preprocessing
// =====================================================================

std::vector<float> YOLOOnnxDetector::preprocessCommon(const cv::Mat& Image) {
    original_image_  = Image.clone();
    original_height_ = Image.rows;
    original_width_  = Image.cols;

    float scale_w = static_cast<float>(width_)  / static_cast<float>(original_width_);
    float scale_h = static_cast<float>(height_) / static_cast<float>(original_height_);
    scale_ = std::min(scale_w, scale_h);

    int new_w = static_cast<int>(original_width_  * scale_);
    int new_h = static_cast<int>(original_height_ * scale_);
    pad_x_ = (width_  - new_w) / 2.0f;
    pad_y_ = (height_ - new_h) / 2.0f;

    cv::Mat resized;
    cv::resize(Image, resized, cv::Size(new_w, new_h));

    // Letterbox with gray padding
    cv::Mat padded(height_, width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(
        static_cast<int>(pad_x_), static_cast<int>(pad_y_), new_w, new_h)));

    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // Normalize [0, 1]
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    // HWC → CHW
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    int channel_size = height_ * width_;
    std::vector<float> blob(batch_size_ * channels_ * channel_size, 0.0f);
    for (int c = 0; c < channels_; c++) {
        std::memcpy(blob.data() + c * channel_size,
                    channels[c].data,
                    channel_size * sizeof(float));
    }
    return blob;
}

std::vector<float> YOLOOnnxDetector::preprocessImage(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return preprocessCommon(img);
}

std::vector<float> YOLOOnnxDetector::preprocessImage(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Failed to preprocess: empty image provided");
    }
    return preprocessCommon(image);
}

// =====================================================================
// Postprocess
// =====================================================================

std::vector<Detection> YOLOOnnxDetector::postprocessOutputs(
    const std::vector<Ort::Value>& outputs, float confidence_threshold) {

    // Identify which output is scores and which is boxes
    const float* scores_ptr = nullptr;
    const float* boxes_ptr  = nullptr;
    int scores_last_dim = 0;
    int boxes_last_dim  = 0;
    int anchors_from_scores = 0;
    int anchors_from_boxes  = 0;

    for (size_t i = 0; i < outputs.size(); i++) {
        auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        const float* data = outputs[i].GetTensorData<float>();

        if (shape.size() == 3) {
            int last_dim = static_cast<int>(shape[2]);
            if (last_dim == 4) {
                boxes_ptr = data;
                boxes_last_dim = last_dim;
                anchors_from_boxes = static_cast<int>(shape[1]);
            } else {
                scores_ptr = data;
                scores_last_dim = last_dim;
                anchors_from_scores = static_cast<int>(shape[1]);
            }
        }
    }

    if (!scores_ptr || !boxes_ptr) {
        RCLCPP_WARN(rclcpp::get_logger("yolo_onnx"),
                    "Could not identify scores/boxes outputs");
        return {};
    }

    int n_anchors = anchors_from_scores;
    int n_classes = scores_last_dim;

    std::vector<Detection> detections;

    for (int i = 0; i < n_anchors; i++) {
        float max_conf = 0.0f;
        int best_class = -1;

        // Skip last padding class if present
        int classes_to_check = (n_classes > 1 && n_classes == static_cast<int>(class_names_.size()))
                                   ? n_classes
                                   : n_classes - 1;
        if (classes_to_check <= 0) classes_to_check = n_classes;

        for (int c = 0; c < classes_to_check; c++) {
            float conf = scores_ptr[i * n_classes + c];
            if (conf > max_conf) {
                max_conf = conf;
                best_class = c;
            }
        }

        if (max_conf < confidence_threshold) continue;

        int box_offset = i * 4;
        float bx0 = boxes_ptr[box_offset + 0];
        float by0 = boxes_ptr[box_offset + 1];
        float bx1 = boxes_ptr[box_offset + 2];
        float by1 = boxes_ptr[box_offset + 3];

        // Auto-detect format: if (bx1 < bx0) it's likely cx,cy,w,h
        // YOLO-World ONNX exports can use either format.
        float x1_model, y1_model, x2_model, y2_model;

        // Heuristic: if 3rd value < 1st value for many boxes, it's cx,cy,w,h
        // For safety, check if values make sense as x1y1x2y2
        if (bx1 > bx0 && by1 > by0) {
            // x1, y1, x2, y2 format
            x1_model = bx0;
            y1_model = by0;
            x2_model = bx1;
            y2_model = by1;
        } else {
            // cx, cy, w, h format
            float cx = bx0, cy = by0, w = bx1, h = by1;
            x1_model = cx - w / 2.0f;
            y1_model = cy - h / 2.0f;
            x2_model = cx + w / 2.0f;
            y2_model = cy + h / 2.0f;
        }

        // Remove letterbox padding and scale back to original coords
        float x1 = (x1_model - pad_x_) / scale_;
        float y1 = (y1_model - pad_y_) / scale_;
        float x2 = (x2_model - pad_x_) / scale_;
        float y2 = (y2_model - pad_y_) / scale_;

        // Clamp to image boundaries
        x1 = std::max(0.0f, std::min(static_cast<float>(original_width_), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(original_height_), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(original_width_), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(original_height_), y2));

        Detection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.confidence = max_conf;
        det.class_id = best_class;
        det.class_name = (best_class >= 0 &&
                          best_class < static_cast<int>(class_names_.size()))
                             ? class_names_[best_class]
                             : "unknown";
        detections.push_back(det);
    }

    return detections;
}

// =====================================================================
// Inference (file path)
// =====================================================================

std::vector<Detection> YOLOOnnxDetector::infer(
    const std::string& image_path, float confidence_threshold) {

    auto start = std::chrono::high_resolution_clock::now();

    auto blob = preprocessImage(image_path);

    // Create input tensor
    std::vector<int64_t> input_shape = {batch_size_, channels_, height_, width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.data(), blob.size(),
        input_shape.data(), input_shape.size());

    // Prepare output names as const char*
    std::vector<const char*> output_name_ptrs;
    for (auto& n : output_names_) output_name_ptrs.push_back(n.c_str());
    const char* input_name_cstr = input_name_.c_str();

    // Run
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name_cstr, &input_tensor, 1,
        output_name_ptrs.data(), output_name_ptrs.size());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "✓ ONNX inference completed in " << ms << " ms" << std::endl;

    auto detections = postprocessOutputs(outputs, confidence_threshold);
    std::cout << "✓ Before NMS: " << detections.size() << " detections" << std::endl;
    detections = applyNMS(detections, 0.45f);
    std::cout << "✓ After NMS: " << detections.size() << " detections" << std::endl;

    return detections;
}

// =====================================================================
// Inference (cv::Mat)
// =====================================================================

std::vector<Detection> YOLOOnnxDetector::infer(
    const cv::Mat& image, float confidence_threshold) {

    auto start = std::chrono::high_resolution_clock::now();

    auto blob = preprocessImage(image);

    std::vector<int64_t> input_shape = {batch_size_, channels_, height_, width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.data(), blob.size(),
        input_shape.data(), input_shape.size());

    std::vector<const char*> output_name_ptrs;
    for (auto& n : output_names_) output_name_ptrs.push_back(n.c_str());
    const char* input_name_cstr = input_name_.c_str();

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name_cstr, &input_tensor, 1,
        output_name_ptrs.data(), output_name_ptrs.size());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    RCLCPP_DEBUG(rclcpp::get_logger("yolo_onnx"),
                 "ONNX inference completed in %.2f ms", ms);

    auto detections = postprocessOutputs(outputs, confidence_threshold);
    detections = applyNMS(detections, 0.45f);

    return detections;
}
