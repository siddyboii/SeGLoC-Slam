#include <s_graphs/common/clip_feature_extractor.hpp>

ClipFeatureExtractor::ClipFeatureExtractor(const std::string& weights):
    weights_(weights),
    ort_env_(ORT_LOGGING_LEVEL_WARNING, "CLIP"),
    ort_sess_opts_()
{
    std::cout << "CLIP extraction started" << std::endl;
    load_ort_params();
}

ClipFeatureExtractor::~ClipFeatureExtractor() {
}

std::vector<float> ClipFeatureExtractor::extract_embedding(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "[CLIP] Error: Empty image provided" << std::endl;
        return std::vector<float>();
    }
    
    return ortForward(image);
}

void ClipFeatureExtractor::makeInputTensor(const cv::Mat& bgr,
                       std::vector<float>& chw,
                       std::array<int64_t,4>& dims) {
    cv::Mat resized; cv::resize(bgr, resized, cv::Size(inp_w_, inp_h_), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb; cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);

    chw.resize(3 * inp_h_ * inp_w_);
    const size_t plane = static_cast<size_t>(inp_h_) * static_cast<size_t>(inp_w_);
    std::vector<cv::Mat> ch(3);
    for (int c = 0; c < 3; ++c) {
      ch[c] = cv::Mat(inp_h_, inp_w_, CV_32F, chw.data() + c * plane);
    }
    cv::split(rgb, ch);  // fills chw

    dims = {1, 3, inp_h_, inp_w_};
}

void ClipFeatureExtractor::init_input_size_safe_() {
  // default if anything fails
  inp_h_ = 224;
  inp_w_ = 224;

  // log the actually loaded ORT runtime
  const char* ort_ver = OrtGetApiBase()->GetVersionString();

  try {
    Ort::TypeInfo ti = ort_sess_->GetInputTypeInfo(0);

    // Ensure input 0 is a tensor
    if (ti.GetONNXType() != ONNX_TYPE_TENSOR) {
      return;
    }

    // NOTE: in newer ORT this is ConstTensorTypeAndShapeInfo
    auto tshape = ti.GetTensorTypeAndShapeInfo();

    // Shape may contain -1 for dynamic dims
    std::vector<int64_t> ishape = tshape.GetShape();
    std::cout << "[ORT] input shape raw: ["
              << (ishape.size() ? std::to_string(ishape[0]) : "?") << ", "
              << (ishape.size() > 1 ? std::to_string(ishape[1]) : "?") << ", "
              << (ishape.size() > 2 ? std::to_string(ishape[2]) : "?") << ", "
              << (ishape.size() > 3 ? std::to_string(ishape[3]) : "?") << "]"
              << std::endl;

    if (ishape.size()==4 && ishape[2] > 0 && ishape[3] > 0) {
      inp_h_ = static_cast<int>(ishape[2]);
      inp_w_ = static_cast<int>(ishape[3]);
    } else {
      std::cout<<"[ORT] Dynamic/unknown input size. Using default 640x640."<<std::endl;
    }
  } catch (const Ort::Exception& e) {
    std::cout<<"[ORT] Failed to read input shape. Using default 224x224."<<std::endl;
  } catch (...) {
    std::cout<<"[ORT] Unknown error while reading input shape. Using default 640x640."<<std::endl;
  }

  std::cout << "[ORT] using input size "
          << inp_w_ << "x" << inp_h_
          << std::endl;
}

void ClipFeatureExtractor::load_ort_params() {
    ort_sess_opts_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    try {
      ort_sess_ = std::make_unique<Ort::Session>(ort_env_, weights_.c_str(), ort_sess_opts_);

      // Cache input/output names (we own std::string; keep parallel const char* arrays for Run)
      Ort::AllocatorWithDefaultOptions alloc;
      const size_t num_in  = ort_sess_->GetInputCount();
      const size_t num_out = ort_sess_->GetOutputCount();

      input_names_.reserve(num_in);
      input_cstrs_.reserve(num_in);
      for (size_t i = 0; i < num_in; ++i) {
        auto n = ort_sess_->GetInputNameAllocated(i, alloc);
        input_names_.emplace_back(n.get());
        input_cstrs_.push_back(input_names_.back().c_str());
      }

      output_names_.reserve(num_out);
      output_cstrs_.reserve(num_out);

      for (size_t i = 0; i < num_out; ++i) {
        auto n = ort_sess_->GetOutputNameAllocated(i, alloc);
        output_names_.emplace_back(n.get());
        output_cstrs_.push_back(output_names_.back().c_str());
      }

      auto tinfo  = ort_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
      //Verify the input size

      init_input_size_safe_();  // may contain -1
      
      std::cout << "[ORT] model=" << weights_
          << " input=" << input_names_.front()
          << " outputs=" << output_names_.size()
          << " size=" << inp_w_ << "x" << inp_h_
          << std::endl;



    } catch (const Ort::Exception& e) {
      std::cerr << "ONNX Runtime failed to load model: " << e.what() << std::endl;
      throw;
    }

}

std::vector<float> ClipFeatureExtractor::ortForward(const cv::Mat& bgr) {
    std::vector<float> input_data;
    std::array<int64_t,4> in_dims;
    makeInputTensor(bgr, input_data, in_dims);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value in_t = Ort::Value::CreateTensor<float>(
        mem, input_data.data(), input_data.size(), in_dims.data(), in_dims.size());

    auto outputs = ort_sess_->Run(
        Ort::RunOptions{nullptr},
        input_cstrs_.data(), &in_t, 1,
        output_cstrs_.data(), output_cstrs_.size());
    float* embedding = outputs.front().GetTensorMutableData<float>();
    auto shape = outputs.front()
              .GetTensorTypeAndShapeInfo()
              .GetShape();
    size_t embedding_size = shape[1];

    std::vector<float> result(embedding, embedding + embedding_size);
    float norm = 0.0f;
    for (float val : result) norm += val * val;
    norm = std::sqrt(norm);
    
    // Add epsilon check to prevent division by zero
    if (norm > 1e-6f) {
        for (float& val : result) val /= norm;
    }

    return result;
}

float ClipFeatureExtractor::compute_similarity(
    const std::vector<float>& emb1,
    const std::vector<float>& emb2) {
  
  float dot_product = 0.0f;
  for (size_t i = 0; i < emb1.size(); i++) {
    dot_product += emb1[i] * emb2[i];
  }
  
  // Both embeddings are normalized, so cosine similarity = dot product
  return std::clamp(dot_product, 0.0f, 1.0f);
}
