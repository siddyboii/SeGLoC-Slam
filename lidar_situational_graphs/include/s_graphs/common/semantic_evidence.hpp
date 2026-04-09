#pragma once

#include<vector>
#include<string>
#include<map>
#include <boost/optional.hpp>

namespace s_graphs {

    struct SemanticEvidence
    {
          // ===== LIDAR-BASED SEMANTICS =====
  
        // Plane geometric stability
        double plane_stability_score;              // [0, 1]
        std::vector<double> plane_normals_variance;
        std::vector<double> plane_distance_variance;
        int num_detected_planes;
            
        // Spatial structure metrics
        double spatial_coverage;                   // [0, 1]
        double geometric_distinctiveness;          // [0, 1]
            
        // Point cloud quality
        size_t point_count;
        double point_density;                      // points per m³
        double point_distribution_uniformity;      // [0, 1]
            
        // ===== CAMERA-BASED SEMANTICS =====
            
        // Visual/CLIP features
        boost::optional<std::vector<float>> clip_embedding;     // 512-dim
        float clip_embedding_magnitude;                         // Norm
            
        // Object detection
        std::vector<std::string> detected_objects;
        std::map<std::string, float> object_confidences;        // conf per object
        int num_high_confidence_objects;                        // count > 0.7
        float object_detection_confidence_mean;                 // average
            
        // Scene understanding
        boost::optional<std::string> scene_type;                // label
        float scene_distinctiveness;                            // [0, 1]
        std::map<std::string, float> scene_class_probabilities; // alternatives
            
        // Image quality
        float image_brightness;                    // [0, 255]
        float image_sharpness;                     // [0, 1]
        float image_motion_blur;                   // [0, 1]
        float image_saturation;                    // [0, 1]
            
        // ===== VISIBILITY & CONSISTENCY =====
            
        double appearance_consistency;             // [0, 1]
        double temporal_stability;                 // [0, 1]
        double visibility_confidence;              // [0, 1]
        bool is_well_observed_lidar;
        bool is_well_observed_camera;
            
        // ===== MULTI-MODAL FUSION =====
            
        double combined_distinctiveness;           // [0, 1]
        double lidar_distinctiveness_weight;       // [0, 1]
        double camera_distinctiveness_weight;      // [0, 1]
            
        float modality_agreement_score;            // [0, 1]
        bool modalities_complementary;
            
        float overall_semantic_confidence;         // [0, 1] FINAL

    }; 
    
} // namespace s_graphs



