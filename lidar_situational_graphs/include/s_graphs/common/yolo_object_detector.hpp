// Header guard added for proper include protection
#ifndef S_GRAPHS_YOLO_OBJECT_DETECTOR_HPP
#define S_GRAPHS_YOLO_OBJECT_DETECTOR_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <array>
#include <chrono>
#include <algorithm>


// Logger for TensorRT

#ifndef S_GRAPHS_DETECTION_STRUCT
#define S_GRAPHS_DETECTION_STRUCT
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;       // Detection confidence
    int class_id;          // Class index
    std::string class_name;
};
#endif

class YOLOWorldTensorRT
{
    private: 
    // std::unique_ptr is a smart pointer that enforces single ownership and automatically frees memory when it goes out of scope
        std::unique_ptr<nvinfer1::IRuntime> runtime; // TensorRT runtime for managing the inference engine
        std::unique_ptr<nvinfer1::ICudaEngine> engine; // TensorRT engine for executing the model
        std::unique_ptr<nvinfer1::IExecutionContext> context; // Execution context for running inference

        /*
        gpu_buffers[0] = input_gpu_ptr
        gpu_buffers[1] = boxes_gpu_ptr
        gpu_buffers[2] = scores_gpu_ptr
        gpu_buffers[3] = classes_gpu_ptr
        */
        std::vector<void*> gpu_buffers; // GPU memory buffers for input and output tensors
        std::vector<void*> cpu_buffers; // CPU memory buffers for input and output tensors
        std::vector<size_t> buffer_sizes; // Sizes of the buffers

        int input_index; // Index of the input tensor
        int output_scores_index; // Index of the output scores tensor
        int output_boxes_index; // Index of the output boxes tensor

        int batch_size; // Batch size for inference
        int channels; // Number of channels in the input tensor
        int height; // Height of the input tensor
        int width; // Width of the input tensor
        int num_anchors; // Number of anchors in the output tensor
        int num_classes; // Number of classes in the output tensor

        cv::Mat original_image; // Original input image
        int original_height; // Original height of the input image
        int original_width; // Original width of the input image
        float pad_x; // Padding in x direction
        float pad_y; // Padding in y direction
        float scale; // Scale factor for resizing

        std::vector<std::string> class_names {  "Persons", "Windows", "Lights", "Door", " "}; // Class names for detected objects

    public:
        explicit YOLOWorldTensorRT(const std::string& trt_model_path); //in public: Constructor
        ~YOLOWorldTensorRT(); // Destructor
        
        std::vector<Detection> infer(const std::string& image_path, float confidence_threshold);
        
        // Overload for cv::Mat input - allows direct inference from ROS image messages
        // without saving to disk. Essential for real-time ROS integration.
        std::vector<Detection> infer(const cv::Mat& image, float confidence_threshold);
        
        // Getter for original image after preprocessing - needed for visualization
        // of detection boxes on the correctly scaled image
        cv::Mat getOriginalImage() const { return original_image; }

    private:
        void preprocessImage(const std::string& image_path); // Preprocess input image from file path
        void preprocessImage(const cv::Mat& image);          // Preprocess input cv::Mat directly
        std::vector<Detection> postprocessOutputs(float confidence_threshold); // Postprocess detections
        void allocateBuffers(); // Allocate GPU and CPU buffers
        float calculateIoU(const Detection& det1, const Detection& det2); // Calculate Intersection over Union (IoU) between two detections
        std::vector<Detection> applyNMS(const std::vector<Detection>& detections, float iou_threshold); // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
};

#endif // S_GRAPHS_YOLO_OBJECT_DETECTOR_HPP