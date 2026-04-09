// Include the header from the proper path
// Rationale: Using the s_graphs/common path ensures CMake finds the header correctly
// when building as part of the ROS package
#include <s_graphs/common/yolo_object_detector.hpp>
#include <rclcpp/rclcpp.hpp>  // Added for RCLCPP_DEBUG logging in cv::Mat infer overload

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << msg << std::endl;
        }
    }
};

static Logger logger;

YOLOWorldTensorRT::~YOLOWorldTensorRT() {
}

/*
Image coordinate system

(0,0)----------------- >x
|
|
|
|
|
y

*/

float YOLOWorldTensorRT::calculateIoU(const Detection& det1, const Detection& det2) {

  float x_left = std::max(det1.x1,det2.x1);
  float y_top = std::max(det1.y1,det2.y1);
  float x_right = std::min(det1.x2,det2.x2);
  float y_bottom = std::min(det1.y2,det2.y2);

  float intersection_area = std::max(0.0f, (x_right - x_left)) * std::max(0.0f, (y_bottom - y_top));

  float det1_area = (det1.x2 - det1.x1) * (det1.y2 - det1.y1);
  float det2_area = (det2.x2 - det2.x1) * (det2.y2 - det2.y1);
  float union_area = det1_area + det2_area - intersection_area; // A U B = A + B - A ∩ B

  return (union_area <= 0) ? 0.0f : intersection_area / union_area;

}

std::vector<Detection> YOLOWorldTensorRT::applyNMS (const std::vector<Detection>& dets, float iou_threshold)
{
  std::vector<Detection> result;
  std::vector<Detection> sorted_dets = dets; // Make a copy to sort

  std::sort(sorted_dets.begin(), sorted_dets.end(),
            [] (const Detection&a, const Detection&b){ return a.confidence > b.confidence; });

  std::vector<bool> supressed(sorted_dets.size(), false);
  for (size_t i = 0; i < sorted_dets.size(); i++) {
    if (supressed[i]) continue;
    result.push_back(sorted_dets[i]);
    //class aware NMS
    for (size_t j = i + 1; j < sorted_dets.size(); j++) {
      if (supressed[j]) continue;
      if (sorted_dets[i].class_id == sorted_dets[j].class_id) {
        if (calculateIoU(sorted_dets[i], sorted_dets[j]) > iou_threshold) {
          supressed[j] = true;
        }
      }
    }
  }

  return result;
}

YOLOWorldTensorRT::YOLOWorldTensorRT(const std::string& trt_model_path) {
  std::cout<<"path to trt model: "<<trt_model_path<<std::endl;
  // Load TensorRT engine from file
  std::cout << "\n✓ Loading TensorRT engine from: " << trt_model_path << std::endl;
  std::ifstream engine_file(trt_model_path, std::ios::binary);
  if (!engine_file.is_open()) {
      throw std::runtime_error("Failed to open TensorRT engine file");
  }
  engine_file.seekg(0, engine_file.end); // Move to end of file
  size_t model_size = engine_file.tellg(); // Get current position in stream (file size) (take to end and see last memeory location)
  engine_file.seekg(0, engine_file.beg); // Move back to beginning of file

  std::vector<char> model_data(model_size); //main baatcheet yahan se shuru hoti hai
  engine_file.read(model_data.data(), model_size); // Read the entire file into the vector
  engine_file.close();

  std::cout << "  File size: " << (model_size / 1024 / 1024) << " MB" << std::endl;

  //Initilizing TRT runtime and engine
  std::cout << "✓ Initializing TensorRT runtime..." << std::endl;
  runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
      throw std::runtime_error("Failed to create TensorRT runtime");
  }
  engine.reset(runtime->deserializeCudaEngine(model_data.data(), model_size));
  if (!engine) {
      throw std::runtime_error("Failed to deserialize CUDA engine");
    }

  std::cout << "✓ Creating execution context..." << std::endl;
  context.reset(engine->createExecutionContext());
  if (!context) {
      throw std::runtime_error("Failed to create execution context");
  }

  std::cout << "\n✓ Model Information:" << std::endl;
  std::cout << "  Number of IO tensors: " << engine->getNbIOTensors() << std::endl; //total number of input and output tensors

  input_index = -1;
  output_scores_index = -1;
  output_boxes_index = -1;

  for (int i = 0; i < engine->getNbIOTensors(); i++)
  {
    const char* name = engine->getIOTensorName(i);
    if (name) {
      std::cout << "  Tensor " << i << " name: " << name << std::endl;
    }
    nvinfer1::Dims dims = engine->getTensorShape(name);
    bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT; // Check if tensor is input or output
    std::cout << "  [" << i << "] " << name 
              << " - " << (is_input ? "INPUT" : "OUTPUT")
              << " - Shape: [";
              for (int j = 0; j < dims.nbDims; j++) {
                  std::cout << dims.d[j];
                  if (j < dims.nbDims - 1) std::cout << ", ";
              }
              std::cout << "]" << std::endl;
    if (is_input) {
        input_index = i;
        batch_size = dims.d[0];
        channels = dims.d[1];
        height = dims.d[2];
        width = dims.d[3];
    } else if (std::string(name).find("scores") != std::string::npos) {
        output_scores_index = i;
        num_anchors = dims.d[1];
        num_classes = dims.d[2];
    } else {
        output_boxes_index = i;
    }
  }

  if (input_index == -1 || output_scores_index == -1 || output_boxes_index == -1) {
      throw std::runtime_error("Could not find all required tensors in model");
  }

  std::cout << "✓ Model loaded successfully!" << std::endl;
  std::cout << "  Input shape: " << batch_size << "x" << channels 
                  << "x" << height << "x" << width << std::endl;
  std::cout << "  Output anchors: " << num_anchors << std::endl;
  std::cout << "  Output classes: " << num_classes << std::endl;

  allocateBuffers();
}

void YOLOWorldTensorRT::allocateBuffers() {
  std::cout << "\n✓ Allocating GPU/CPU buffers..." << std::endl;
  for (int i = 0; i < engine->getNbIOTensors(); i++) {
    const char* name = engine->getIOTensorName(i);
    nvinfer1::Dims dims = engine->getTensorShape(name); // Get tensor shape
    // Calculate buffer size
    size_t volume = 1; //total number of elements in the tensor
    for (int j = 0; j < dims.nbDims; j++) {
        volume *= dims.d[j];  // Multiply each dimension to get total volume
      }
    size_t size_bytes = volume * sizeof(float); // Assuming float data type (4 bytes)
    buffer_sizes.push_back(size_bytes); // Store buffer size
    void* gpu_buffer; // Allocate GPU memory
    cudaMalloc(&gpu_buffer, size_bytes); // Allocate GPU memory
    gpu_buffers.push_back(gpu_buffer); // Store GPU buffer pointer
    void* cpu_buffer = malloc(size_bytes); // Allocate CPU memory
    cpu_buffers.push_back(cpu_buffer); // Store CPU buffer pointer
    std::cout << "  [" << name << "] " << (size_bytes / 1024) << " KB" << std::endl;
  }
}

void YOLOWorldTensorRT::preprocessImage(const std::string& image_path)  //& means pass by reference go to the memory location directly
{
  std::cout << "\n✓ Preprocessing image: " << image_path << std::endl;
  cv::Mat Image = cv::imread(image_path); // Read image using OpenCV
  if (Image.empty()) {
      throw std::runtime_error("Failed to load image: " + image_path);
  }
  original_image = Image.clone(); // Store original image for later use
  original_height = Image.rows; // Store original height
  original_width = Image.cols; // Store original width

  std::cout << "original size of image is : " << original_width << " x " << original_height << std::endl;

  //Resize it to shape accoring to model (here is the mail modification which will happended in preprocess function)

  float scale_w = static_cast<float>(width) / static_cast<float>(original_width);
  float scale_h = static_cast<float>(height) / static_cast<float>(original_height); //static cast is used to convert int to float 

  scale = std::min(scale_w, scale_h); // Get the minimum scale to maintain aspect ratio

  //padding
  pad_x = (width - static_cast<int>(original_width * scale)) / 2;
  pad_y = (height - static_cast<int>(original_height * scale)) / 2;

  int new_w = static_cast<int>(original_width * scale);
  int new_h = static_cast<int>(original_height * scale);

  cv::Mat resized;
  cv::resize(Image, resized, cv::Size(new_w, new_h));

  //Create a new image with padding
  cv::Mat padded(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));
  std::cout << "Resized image to: " << width << " x " << height << " with letterbox padding" << std::endl;

  //Convert BGR to RGB
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

  //Normalize to [0, 1]
  rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

  //Convert HWC to CHW format
  std::vector<cv::Mat> channels;
  cv::split(rgb, channels); // Split into R, G, B channels
  //Copy data to input buffer

  float* input_ptr = static_cast<float*>(cpu_buffers[input_index]);
  int channel_size = height * width;

  //copy each channel data to input buffer in NCHW format
  for (int c = 0; c < channels.size(); c++) {
      std::memcpy(input_ptr + c * channel_size, //destination pointer
                  channels[c].data, //source pointer 
                  channel_size * sizeof(float)); //number of bytes to copy
  }

  std::cout << "Image preprocessed and copied to input buffer" << std::endl;
  //basically we have converted the image into the format which is required by the model and copied it to the input buffer
  //input buffer is the cpu memory buffer which we have created earlier,
  //float* input_ptr is makeing the generic void pointer to the specific float pointer
  //this is done because we know that the model input is of float type
  //so in memeory it is stored as [rrrrrrrrrrrr][gggggggggggg][bbbbbbbbbbbb] for each channel
  //3 iteration for 3 channels is N = 1, C = 3, H = height, W = width
  //if N=more than 1 then we have to do for each image in the batch

}

// Overloaded preprocessImage for cv::Mat input
// Rationale: This method is essential for ROS integration - it accepts cv::Mat directly
// from cv_bridge without requiring disk I/O. The preprocessing logic (letterbox padding,
// normalization, CHW conversion) is identical to the file-based version.
void YOLOWorldTensorRT::preprocessImage(const cv::Mat& Image)
{
  if (Image.empty()) {
      throw std::runtime_error("Failed to preprocess: empty image provided");
  }
  original_image = Image.clone(); // Store original image for later use
  original_height = Image.rows; // Store original height
  original_width = Image.cols; // Store original width

  // Calculate scale to maintain aspect ratio during resize
  float scale_w = static_cast<float>(width) / static_cast<float>(original_width);
  float scale_h = static_cast<float>(height) / static_cast<float>(original_height);
  scale = std::min(scale_w, scale_h);

  // Calculate letterbox padding to center the image
  pad_x = (width - static_cast<int>(original_width * scale)) / 2;
  pad_y = (height - static_cast<int>(original_height * scale)) / 2;

  int new_w = static_cast<int>(original_width * scale);
  int new_h = static_cast<int>(original_height * scale);

  cv::Mat resized;
  cv::resize(Image, resized, cv::Size(new_w, new_h));

  // Create letterboxed image with gray padding (114 is YOLO standard)
  cv::Mat padded(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

  // Convert BGR to RGB (YOLO models expect RGB input)
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

  // Normalize pixel values to [0, 1] range
  rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

  // Convert HWC to CHW format (channels-first for TensorRT)
  std::vector<cv::Mat> channels;
  cv::split(rgb, channels);

  float* input_ptr = static_cast<float*>(cpu_buffers[input_index]);
  int channel_size = height * width;

  // Copy each channel data to input buffer in NCHW format
  for (size_t c = 0; c < channels.size(); c++) {
      std::memcpy(input_ptr + c * channel_size,
                  channels[c].data,
                  channel_size * sizeof(float));
  }
}

std::vector<Detection> YOLOWorldTensorRT::infer(const std::string& image_path, float confidence_threshold) 
{
  auto start_time = std::chrono::high_resolution_clock::now(); //start time for measuring inference time
  //Preprocess image
  preprocessImage(image_path);
  //Copy input to GPU
  std::cout << "\n✓ Running inference..." << std::endl;
  cudaMemcpy(gpu_buffers[input_index], //destination pointer
              cpu_buffers[input_index], //source pointer
              buffer_sizes[input_index], //number of bytes to copy
              cudaMemcpyHostToDevice); //copy from host(cpu) to device(gpu)

  //Execute inference
  std::vector<void*> bindings(gpu_buffers.begin(), gpu_buffers.end());
  bool success = context->executeV2(bindings.data());
  if (!success) {
      throw std::runtime_error("Failed to execute inference");

  }
  cudaDeviceSynchronize(); //wait for the GPU to finish
  //Copy outputs from GPU
  cudaMemcpy(cpu_buffers[output_scores_index], //destination pointer
              gpu_buffers[output_scores_index], //source pointer
              buffer_sizes[output_scores_index], //number of bytes to copy
              cudaMemcpyDeviceToHost); //copy from device(gpu) to host(cpu)
  cudaMemcpy(cpu_buffers[output_boxes_index],
              gpu_buffers[output_boxes_index],
              buffer_sizes[output_boxes_index],
              cudaMemcpyDeviceToHost);
  auto end_time = std::chrono::high_resolution_clock::now(); //end time for measuring inference time
  std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
  std::cout << "✓ Inference completed in " << inference_time.count() << " ms" << std::endl;

  std::vector<Detection> detections = postprocessOutputs(confidence_threshold);
  std::cout << "✓ Before NMS: " << detections.size() << " detections" << std::endl;
  detections = applyNMS(detections, 0.45f); // Apply NMS with IoU threshold of 0.45
  std::cout << "✓ After NMS: " << detections.size() << " detections" << std::endl;

  return detections;
}

// Overloaded infer method for cv::Mat input
// Rationale: This overload is critical for ROS integration - it accepts cv::Mat
// directly from cv_bridge, avoiding file I/O latency. The inference pipeline
// (GPU transfer, execution, output copy) is identical to the file-based version.
std::vector<Detection> YOLOWorldTensorRT::infer(const cv::Mat& image, float confidence_threshold) 
{
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Preprocess cv::Mat directly
  preprocessImage(image);
  
  // Copy input to GPU
  cudaMemcpy(gpu_buffers[input_index],
              cpu_buffers[input_index],
              buffer_sizes[input_index],
              cudaMemcpyHostToDevice);

  // Execute inference
  std::vector<void*> bindings(gpu_buffers.begin(), gpu_buffers.end());
  bool success = context->executeV2(bindings.data());
  if (!success) {
      throw std::runtime_error("Failed to execute inference");
  }
  cudaDeviceSynchronize();
  
  // Copy outputs from GPU
  cudaMemcpy(cpu_buffers[output_scores_index],
              gpu_buffers[output_scores_index],
              buffer_sizes[output_scores_index],
              cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_buffers[output_boxes_index],
              gpu_buffers[output_boxes_index],
              buffer_sizes[output_boxes_index],
              cudaMemcpyDeviceToHost);
              
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
  
  // Reduced logging for real-time performance - only log timing
  RCLCPP_DEBUG(rclcpp::get_logger("yolo_detector"), 
               "Inference completed in %.2f ms", inference_time.count());

  std::vector<Detection> detections = postprocessOutputs(confidence_threshold);
  detections = applyNMS(detections, 0.45f);

  return detections;
}

std::vector<Detection> YOLOWorldTensorRT::postprocessOutputs(float confidence_threshold)
{
  std::cout << "\n Post -processing outputs..." << std::endl;
  float* scores_ptr = static_cast<float*>(cpu_buffers[output_scores_index]); //take from cpu buffer the output scores
  float* boxes_ptr = static_cast<float*>(cpu_buffers[output_boxes_index]); //take from cpu buffer the output boxes

  std::vector<Detection> detections;

  for (int i = 0; i < num_anchors; i++) {
            // Get the maximum confidence across all classes (skip last padding class)
            float max_conf = 0.0f;
            int best_class = -1;
            
            for (int c = 0; c < num_classes - 1; c++) {  // Skip padding class
                float conf = scores_ptr[i * num_classes + c];
                if (conf > max_conf) {
                    max_conf = conf;
                    best_class = c;
                }
            }
            
            // Filter by confidence threshold
            if (max_conf < confidence_threshold) {
                continue;
            }
            
            // Get bounding box - ONNX outputs (x1, y1, x2, y2) format in 640x640 model space
            int box_offset = i * 4;
            float x1_model = boxes_ptr[box_offset + 0];
            float y1_model = boxes_ptr[box_offset + 1];
            float x2_model = boxes_ptr[box_offset + 2];
            float y2_model = boxes_ptr[box_offset + 3];
            
            // Remove letterbox padding and convert to original image coordinates
            // 1. Subtract padding offset
            float x1_unpad = x1_model - pad_x;
            float y1_unpad = y1_model - pad_y;
            float x2_unpad = x2_model - pad_x;
            float y2_unpad = y2_model - pad_y;
            
            // 2. Scale back to original image size (divide by scale)
            float x1 = x1_unpad / scale;
            float y1 = y1_unpad / scale;
            float x2 = x2_unpad / scale;
            float y2 = y2_unpad / scale;
            
            // 3. Clamp to original image boundaries
            x1 = std::max(0.0f, std::min(static_cast<float>(original_width), x1));
            y1 = std::max(0.0f, std::min(static_cast<float>(original_height), y1));
            x2 = std::max(0.0f, std::min(static_cast<float>(original_width), x2));
            y2 = std::max(0.0f, std::min(static_cast<float>(original_height), y2));
            
            Detection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.confidence = max_conf;
            det.class_id = best_class;
            det.class_name = (best_class >= 0 && best_class < class_names.size()) 
                           ? class_names[best_class] 
                           : "unknown";
            
            detections.push_back(det);
        }
        
        std::cout << "  Found " << detections.size() << " detections" << std::endl;
        
        return detections;
    }