#ifndef CLIP_FEATURE_EXTRACTOR_H
#define CLIP_FEATURE_EXTRACTOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <iostream>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <exception>


class ClipFeatureExtractor {
    public:

    //Constructor
    explicit ClipFeatureExtractor(const std::string& weights);
    
    //Destructor
    ~ClipFeatureExtractor(); 

    std::vector<float> extract_embedding(const cv::Mat& image);

    float compute_similarity(const std::vector<float>& emb1, const std::vector<float>& emb2);

    
    private:

        int inp_w_ = 224, inp_h_ = 224;
        std::string weights_;

        //ORT variables    
        Ort::Env ort_env_;
        Ort::SessionOptions ort_sess_opts_;
        std::unique_ptr<Ort::Session> ort_sess_;
        std::vector<std::string> input_names_, output_names_;
        std::vector<const char*> input_cstrs_, output_cstrs_;


        // Read Parameters
        void load_ort_params();

        //ORT Functions
        std::vector<float> ortForward(const cv::Mat& bgr);
        void makeInputTensor(const cv::Mat& bgr,
                        std::vector<float>& chw,  
                        std::array<int64_t,4>& dims);

        void init_input_size_safe_(); 
        
};

#endif // CLIP_FEATURE_EXTRACTOR_H