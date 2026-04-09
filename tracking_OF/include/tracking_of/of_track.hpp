#ifndef TRACKING_OF__OF_TRACK_HPP_
#define TRACKING_OF__OF_TRACK_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <situational_graphs_msgs/msg/dynamic_objects.hpp>
#include <vector>
#include <cmath>
#include <mutex>

using PointT = pcl::PointXYZ;

struct Cell {
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
};

// Represents a detected dynamic object cluster
struct TrackedCluster {
    cv::Point2f mean_position;    // Mean (row, col) in image coords
    cv::Point2f mean_velocity;    // Mean (vx, vy) from optical flow
    float mean_omega;             // Mean angular velocity
    int pixel_count;              // Number of pixels in the cluster
};

class DynaTrack : public rclcpp::Node {
public:
    DynaTrack();

private:
    // --- ROS interfaces ---
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr track_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr masked_flow_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr compensated_flow_publisher_;
    rclcpp::Publisher<situational_graphs_msgs::msg::DynamicObjects>::SharedPtr dynamic_objects_publisher_;

    // --- Grid configuration ---
    double x_min, x_max;
    double y_min, y_max;
    double resolution;
    double a, b, h_max;

    // --- Ego-motion from odometry ---
    // Latest robot velocities from /odom (body frame)
    double ego_vx_;      // Linear velocity along robot X (forward)
    double ego_vy_;      // Linear velocity along robot Y (left)
    double ego_omega_;   // Angular velocity around Z (yaw rate)
    bool odom_received_; // Whether we've received at least one odom msg
    std::mutex odom_mutex_;  // Thread safety for odom data

    // --- Temporal data for optical flow (Section IV-C.1) ---
    cv::Mat prev_gray_img_;       // Previous grayscale BEV frame
    cv::Mat prev_flow_;           // Previous linear velocity field V(t-1)  [CV_32FC2]
    cv::Mat prev_omega_;          // Previous angular velocity field ω(t-1) [CV_32FC1]
    double dt_;                   // Time increment between frames (s)

    // --- Masking thresholds ---
    double tau_p_;                // Propagation mask threshold
    double tau_div_;              // Divergence (linear continuity) threshold
    double tau_omega_;            // Angular continuity threshold (Laplacian of ω)
    double min_flow_magnitude_;   // Minimum flow magnitude to consider a cell "moving"

    // --- Clustering parameters ---
    double cluster_distance_;     // Max Euclidean distance for clustering (pixels)
    int min_cluster_size_;        // Minimum pixels to form a valid cluster

    // --- Core pipeline ---
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
    void compute_optical_flow(const cv::Mat& current_gray);

    // --- Ego-motion compensation ---
    cv::Mat compute_ego_flow(int rows, int cols);

    // --- Masking & Filtering (Section IV-C) ---
    cv::Mat compute_propagation_mask(const cv::Mat& flow, const cv::Mat& omega);
    cv::Mat compute_continuity_mask(const cv::Mat& flow, const cv::Mat& omega);
    cv::Mat apply_masks(const cv::Mat& flow, const cv::Mat& omega,
                        const cv::Mat& mask_p, const cv::Mat& mask_c);

    // --- Clustering (Section IV-C.5) ---
    std::vector<TrackedCluster> cluster_vectors(const cv::Mat& masked_flow,
                                                const cv::Mat& masked_omega,
                                                const cv::Mat& final_mask);
};

#endif  // TRACKING_OF__OF_TRACK_HPP_
