#include "tracking_of/of_track.hpp"
#include <std_msgs/msg/header.hpp>
#include <situational_graphs_msgs/msg/dynamic_objects.hpp>
#include <cmath>

DynaTrack::DynaTrack()
    : Node("DynaTrack"),
      ego_vx_(0.0), ego_vy_(0.0), ego_omega_(0.0),
      odom_received_(false) {
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", 10,
        std::bind(&DynaTrack::point_cloud_callback, this,
                  std::placeholders::_1));

    odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&DynaTrack::odom_callback, this,
                  std::placeholders::_1));

    image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/bev_grid_map", 10);
    track_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/track_map", 10);
    masked_flow_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/masked_flow_map", 10);
    compensated_flow_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/compensated_flow_map", 10);
    dynamic_objects_publisher_ = this->create_publisher<situational_graphs_msgs::msg::DynamicObjects>(
        "dynamic_objects", 10);

    // Grid config
    x_min = -50.0; x_max = 50.0;
    y_min = -50.0; y_max = 50.0;
    resolution = 0.17; 

    // Paper constants
    a = 1.0;
    b = 1.0;
    h_max = 5.0; // Fixed normalization height 

    // Temporal config (Section IV-C.1)
    dt_ = 0.1;  // 10 Hz LiDAR -> 0.1s between frames

    // Masking thresholds (Section IV-C.2 & IV-C.3)
    tau_p_   = 2.0;    // Propagation mask threshold (pixels/frame)
    tau_div_ = 0.5;    // Divergence threshold for linear continuity
    tau_omega_ = 0.3;  // Angular continuity threshold (Laplacian of omega)
    min_flow_magnitude_ = 1.5; // Minimum flow to consider cell as moving (increased to reduce noise)

    // Clustering parameters (Section IV-C.5)
    cluster_distance_ = 5.0;   // Max Euclidean distance (pixels) for grouping
    min_cluster_size_ = 15;    // Minimum pixels per valid cluster (increased to reduce noise)

    RCLCPP_INFO(this->get_logger(), "DynaTrack node initialized (with ego-motion compensation)");
}

// =============================================================================
// Odometry Callback
// =============================================================================
// Caches the latest robot body-frame velocities from odometry.
// These are used to compute the expected ego-motion flow in the BEV image.
// =============================================================================
void DynaTrack::odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    ego_vx_    = odom_msg->twist.twist.linear.x;   // forward velocity (m/s)
    ego_vy_    = odom_msg->twist.twist.linear.y;    // lateral velocity (m/s)
    ego_omega_ = odom_msg->twist.twist.angular.z;   // yaw rate (rad/s)
    odom_received_ = true;
}

// =============================================================================
// Ego-Motion Flow Field Computation
// =============================================================================
// Given the robot's body-frame velocities (vx, vy, ω), computes the expected
// optical flow at every BEV pixel due to the robot's own motion.
//
// BEV coordinate mapping (from point_cloud_callback):
//   row_idx = (size_x - 1) - i,  where i = (x - x_min) / resolution
//   col_idx = (size_y - 1) - j,  where j = (y - y_min) / resolution
//
// So for pixel (row, col):
//   world_x = x_max - row * resolution   (row 0 = x_max = forward)
//   world_y = y_max - col * resolution   (col 0 = y_max = left)
//
// The ego-motion at a world point (px, py) relative to the robot at the
// origin is a combination of:
//   - Translation: every point appears to move opposite to robot velocity
//   - Rotation:    every point appears to rotate opposite to robot yaw rate
//
// Apparent velocity of a STATIC point at (px, py) in world frame:
//   apparent_vx = -(ego_vx - ego_omega * py)
//   apparent_vy = -(ego_vy + ego_omega * px)
//
// We then convert this world-velocity into pixel-shift per frame (flow):
//   flow_col = -apparent_vy / resolution * dt  (y-axis → col direction)
//   flow_row = -apparent_vx / resolution * dt  (x-axis → row direction, flipped)
// =============================================================================
cv::Mat DynaTrack::compute_ego_flow(int rows, int cols) {
    cv::Mat ego_flow(rows, cols, CV_32FC2, cv::Scalar(0, 0));

    double vx_robot, vy_robot, omega_robot;
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        vx_robot    = ego_vx_;
        vy_robot    = ego_vy_;
        omega_robot = ego_omega_;
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Convert pixel (r, c) back to world coordinates
            // row 0 = x_max (forward), row increases = x decreases
            double world_x = x_max - r * resolution;
            // col 0 = y_max (left),    col increases = y decreases
            double world_y = y_max - c * resolution;

            // Apparent velocity of a STATIC point at (world_x, world_y)
            // due to robot moving with (vx_robot, vy_robot, omega_robot):
            //
            // In world frame, a static point appears to move as:
            //   app_x = -(vx_robot - omega_robot * world_y)
            //   app_y = -(vy_robot + omega_robot * world_x)
            double app_x = -(vx_robot - omega_robot * world_y);
            double app_y = -(vy_robot + omega_robot * world_x);

            // Convert world velocity to BEV pixel flow (pixels per frame)
            // Row direction: row decreases as x increases, so flow_row = -app_x / res * dt
            // Col direction: col decreases as y increases, so flow_col = -app_y / res * dt
            float flow_col = static_cast<float>(-app_y / resolution * dt_);
            float flow_row = static_cast<float>(-app_x / resolution * dt_);

            // OpenCV flow convention: flow[0] = horizontal (col), flow[1] = vertical (row)
            ego_flow.at<cv::Vec2f>(r, c) = cv::Vec2f(flow_col, flow_row);
        }
    }

    return ego_flow;
}

void DynaTrack::compute_optical_flow(const cv::Mat& current_gray) {
    // If this is the first frame, just store it
    if (prev_gray_img_.empty()) {
        prev_gray_img_ = current_gray.clone();
        return;
    }

    // =========================================================
    // Step 1: Compute dense optical flow using Farneback method
    // =========================================================
    cv::Mat flow(prev_gray_img_.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(
        prev_gray_img_,
        current_gray,
        flow,
        0.5,      // pyr_scale: pyramid scale
        3,        // levels: number of pyramid levels
        15,       // winsize: averaging window size
        3,        // iterations: number of iterations
        5,        // poly_n: size of pixel neighborhood
        1.2,      // poly_sigma: standard deviation of Gaussian
        cv::OPTFLOW_FARNEBACK_GAUSSIAN);

    // Split raw flow into vx, vy components
    cv::Mat raw_flow_parts[2];
    cv::split(flow, raw_flow_parts);
    cv::Mat raw_vx = raw_flow_parts[0];
    cv::Mat raw_vy = raw_flow_parts[1];

    // =========================================================
    // Step 1.5: Ego-Motion Compensation
    // =========================================================
    // Compute the expected flow field from robot's own motion and subtract
    // it from the raw optical flow. After this, static objects will have
    // ~zero residual flow, and only truly dynamic objects will remain.
    cv::Mat ego_flow = compute_ego_flow(flow.rows, flow.cols);
    cv::Mat compensated_flow;
    cv::subtract(flow, ego_flow, compensated_flow);

    if (!odom_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "No odometry received yet — ego-motion compensation disabled");
        compensated_flow = flow.clone();
    }

    // Use compensated flow for all downstream processing
    cv::Mat comp_parts[2];
    cv::split(compensated_flow, comp_parts);
    cv::Mat vx = comp_parts[0];  // ego-compensated horizontal velocity
    cv::Mat vy = comp_parts[1];  // ego-compensated vertical velocity

    // Compute angular velocity on compensated flow: omega = 0.5 * (dvy/dx - dvx/dy)
    cv::Mat dvx_dy, dvy_dx;
    cv::Sobel(vx, dvx_dy, CV_32F, 0, 1, 3);
    cv::Sobel(vy, dvy_dx, CV_32F, 1, 0, 3);
    cv::Mat omega = 0.5f * (dvy_dx - dvx_dy);

    // =========================================================
    // Step 2: Masking and Filtering (Section IV-C)
    // =========================================================
    cv::Mat mask_p = compute_propagation_mask(compensated_flow, omega);
    cv::Mat mask_c = compute_continuity_mask(compensated_flow, omega);
    cv::Mat masked_flow = apply_masks(compensated_flow, omega, mask_p, mask_c);

    // Compute the final combined mask for clustering
    // Note: apply_masks already computes the final mask including the magnitude threshold.
    // We should re-compute it here the exact same way, or better yet, just use the magnitude
    // of the masked_flow to determine the final mask.
    cv::Mat final_mask = cv::Mat::zeros(masked_flow.size(), CV_8UC1);
    cv::Mat masked_parts_tmp[2];
    cv::split(masked_flow, masked_parts_tmp);
    cv::Mat masked_mag_tmp;
    cv::magnitude(masked_parts_tmp[0], masked_parts_tmp[1], masked_mag_tmp);
    for (int r = 0; r < masked_mag_tmp.rows; r++) {
        for (int c = 0; c < masked_mag_tmp.cols; c++) {
            if (masked_mag_tmp.at<float>(r, c) > 0) {
                final_mask.at<uchar>(r, c) = 255;
            }
        }
    }

    // Also compute masked omega for clustering
    cv::Mat masked_omega;
    omega.copyTo(masked_omega, final_mask);

    // =========================================================
    // Step 3: Cluster remaining vectors (Section IV-C.5)
    // =========================================================
    std::vector<TrackedCluster> clusters = cluster_vectors(masked_flow, masked_omega, final_mask);

    RCLCPP_INFO(this->get_logger(), "Detected %zu dynamic clusters", clusters.size());
    for (size_t k = 0; k < clusters.size(); k++) {
        RCLCPP_DEBUG(this->get_logger(),
            "  Cluster %zu: pos=(%.1f,%.1f) vel=(%.2f,%.2f) omega=%.3f pixels=%d",
            k, clusters[k].mean_position.x, clusters[k].mean_position.y,
            clusters[k].mean_velocity.x, clusters[k].mean_velocity.y,
            clusters[k].mean_omega, clusters[k].pixel_count);
    }

    // =========================================================
    // Step 3.5: Publish DynamicObjects message
    // =========================================================
    {
        situational_graphs_msgs::msg::DynamicObjects dyn_msg;
        dyn_msg.header.stamp = this->now();
        dyn_msg.header.frame_id = "base_link";
        dyn_msg.num_dynamic_clusters = static_cast<int32_t>(clusters.size());

        int total_dyn_pixels = 0;
        for (const auto& cluster : clusters) {
            dyn_msg.cluster_pos_x.push_back(cluster.mean_position.x);
            dyn_msg.cluster_pos_y.push_back(cluster.mean_position.y);
            dyn_msg.cluster_vel_x.push_back(cluster.mean_velocity.x);
            dyn_msg.cluster_vel_y.push_back(cluster.mean_velocity.y);
            dyn_msg.cluster_omega.push_back(cluster.mean_omega);
            dyn_msg.cluster_pixel_count.push_back(cluster.pixel_count);
            total_dyn_pixels += cluster.pixel_count;
        }

        // Count total occupied pixels in the BEV image
        int total_occupied = cv::countNonZero(current_gray);

        dyn_msg.total_dynamic_pixels = total_dyn_pixels;
        dyn_msg.total_occupied_pixels = total_occupied;

        // Scene dynamicity score: ratio of dynamic pixels to occupied pixels
        // Clamped to [0, 1]
        if (total_occupied > 0) {
            // Scale the ratio to make it more sensitive, as dynamic pixels are usually a small fraction
            float ratio = static_cast<float>(total_dyn_pixels) / static_cast<float>(total_occupied);
            dyn_msg.scene_dynamicity = std::min(1.0f, ratio * 5.0f); // 20% dynamic pixels = 1.0 dynamicity
        } else {
            dyn_msg.scene_dynamicity = 0.0f;
        }

        dynamic_objects_publisher_->publish(dyn_msg);
        RCLCPP_INFO(this->get_logger(),
            "Published DynamicObjects: %d clusters, dynamicity=%.3f (%d/%d pixels)",
            dyn_msg.num_dynamic_clusters, dyn_msg.scene_dynamicity,
            total_dyn_pixels, total_occupied);
    }

    // =========================================================
    // Step 4: Visualization & Publishing
    // =========================================================
    // --- HSV visualization of raw optical flow (before ego-compensation) ---
    cv::Mat raw_magnitude, raw_angle, raw_magn_norm;
    cv::cartToPolar(raw_vx, raw_vy, raw_magnitude, raw_angle, true);
    cv::normalize(raw_magnitude, raw_magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    cv::Mat hsv_parts[3], hsv, hsv8;
    hsv_parts[0] = raw_angle * (180.0f / 360.0f);  // H: [0,180] for OpenCV
    hsv_parts[1] = cv::Mat::ones(raw_angle.size(), CV_32F);
    hsv_parts[2] = raw_magn_norm;
    cv::merge(hsv_parts, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::Mat bgr_flow;
    cv::cvtColor(hsv8, bgr_flow, cv::COLOR_HSV2BGR);

    auto flow_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr_flow).toImageMsg();
    track_publisher_->publish(*flow_msg);

    // --- HSV visualization of ego-compensated flow ---
    cv::Mat comp_magnitude, comp_angle, comp_magn_norm;
    cv::cartToPolar(vx, vy, comp_magnitude, comp_angle, true);
    cv::normalize(comp_magnitude, comp_magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    cv::Mat chsv_parts[3], chsv, chsv8;
    chsv_parts[0] = comp_angle * (180.0f / 360.0f);
    chsv_parts[1] = cv::Mat::ones(comp_angle.size(), CV_32F);
    chsv_parts[2] = comp_magn_norm;
    cv::merge(chsv_parts, 3, chsv);
    chsv.convertTo(chsv8, CV_8U, 255.0);
    cv::Mat bgr_comp;
    cv::cvtColor(chsv8, bgr_comp, cv::COLOR_HSV2BGR);

    auto comp_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr_comp).toImageMsg();
    compensated_flow_publisher_->publish(*comp_msg);

    // --- Visualization of masked flow (filtered dynamic regions) ---
    cv::Mat masked_parts[2];
    cv::split(masked_flow, masked_parts);
    cv::Mat masked_mag, masked_ang, masked_mag_norm;
    cv::cartToPolar(masked_parts[0], masked_parts[1], masked_mag, masked_ang, true);
    cv::normalize(masked_mag, masked_mag_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    cv::Mat mhsv_parts[3], mhsv, mhsv8;
    mhsv_parts[0] = masked_ang * (180.0f / 360.0f);
    mhsv_parts[1] = cv::Mat::ones(masked_ang.size(), CV_32F);
    mhsv_parts[2] = masked_mag_norm;
    cv::merge(mhsv_parts, 3, mhsv);
    mhsv.convertTo(mhsv8, CV_8U, 255.0);
    cv::Mat bgr_masked;
    cv::cvtColor(mhsv8, bgr_masked, cv::COLOR_HSV2BGR);

    // Draw cluster centroids on the masked visualization
    for (const auto& cluster : clusters) {
        cv::circle(bgr_masked,
                   cv::Point(static_cast<int>(cluster.mean_position.y),
                             static_cast<int>(cluster.mean_position.x)),
                   8, cv::Scalar(0, 255, 0), 2);
    }

    auto masked_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr_masked).toImageMsg();
    masked_flow_publisher_->publish(*masked_msg);

    // =========================================================
    // Step 5: Store current data for next iteration
    // =========================================================
    // Store the COMPENSATED flow for the propagation mask (next frame
    // should predict based on true object motion, not ego-contaminated flow)
    prev_gray_img_ = current_gray.clone();
    prev_flow_ = compensated_flow.clone();
    prev_omega_ = omega.clone();

    double min_val, max_val;
    cv::minMaxLoc(comp_magnitude, &min_val, &max_val);
    RCLCPP_DEBUG(this->get_logger(),
                 "Optical flow computed: max_magnitude=%.3f (after ego compensation)", max_val);
}

// =============================================================================
// Section IV-C.2: Vector Field Propagation Mask (M_p)
// =============================================================================
// Warps the previous velocity field forward using its own velocities to predict
// what the current flow "should" look like. Cells where the actual flow differs
// significantly from the prediction are masked out (false positives from noise).
// =============================================================================
cv::Mat DynaTrack::compute_propagation_mask(const cv::Mat& flow,
                                             const cv::Mat& omega) {
    cv::Mat mask = cv::Mat::ones(flow.size(), CV_8UC1);  // default: all pass

    // Need at least one previous flow field to propagate
    if (prev_flow_.empty()) {
        return mask;
    }

    const int rows = flow.rows;
    const int cols = flow.cols;

    cv::Mat prev_parts[2], curr_parts[2];
    cv::split(prev_flow_, prev_parts);  // prev vx, vy
    cv::split(flow, curr_parts);        // current vx, vy

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float prev_vx = prev_parts[0].at<float>(r, c);
            float prev_vy = prev_parts[1].at<float>(r, c);

            // Predicted source position: where did this cell come from?
            // x_pred = x + vx_prev * dt, y_pred = y + vy_prev * dt
            float src_r = r + prev_vy * static_cast<float>(dt_);
            float src_c = c + prev_vx * static_cast<float>(dt_);

            // Boundary check for the predicted source
            int sr = static_cast<int>(std::round(src_r));
            int sc = static_cast<int>(std::round(src_c));

            if (sr < 0 || sr >= rows || sc < 0 || sc >= cols) {
                mask.at<uchar>(r, c) = 0;  // Out of bounds -> mask out
                continue;
            }

            // Propagated (predicted) velocity at current position
            float prop_vx = prev_parts[0].at<float>(sr, sc);
            float prop_vy = prev_parts[1].at<float>(sr, sc);

            // Actual velocity at current position
            float actual_vx = curr_parts[0].at<float>(r, c);
            float actual_vy = curr_parts[1].at<float>(r, c);

            // Difference magnitude
            float diff = std::sqrt((actual_vx - prop_vx) * (actual_vx - prop_vx) +
                                   (actual_vy - prop_vy) * (actual_vy - prop_vy));

            // Apply threshold tau_p: keep cell only if difference is small
            if (diff > tau_p_) {
                mask.at<uchar>(r, c) = 0;
            }
        }
    }

    return mask;
}

// =============================================================================
// Section IV-C.3: Rigid-Body Continuity Mask (M_c)
// =============================================================================
// Enforces rigid-body motion constraints:
//   - Linear continuity: div(V) ≈ 0  (no tearing / imploding)
//   - Angular continuity: Laplacian(ω) ≈ 0  (uniform rotation across object)
// Cells violating these constraints are masked out.
// =============================================================================
cv::Mat DynaTrack::compute_continuity_mask(const cv::Mat& flow,
                                            const cv::Mat& omega) {
    const int rows = flow.rows;
    const int cols = flow.cols;

    cv::Mat mask = cv::Mat::ones(flow.size(), CV_8UC1);

    // Split flow into components
    cv::Mat parts[2];
    cv::split(flow, parts);
    cv::Mat vx = parts[0];
    cv::Mat vy = parts[1];

    // --- Linear continuity: divergence = dvx/dx + dvy/dy ---
    cv::Mat dvx_dx, dvy_dy;
    cv::Sobel(vx, dvx_dx, CV_32F, 1, 0, 3);  // partial vx / partial x
    cv::Sobel(vy, dvy_dy, CV_32F, 0, 1, 3);   // partial vy / partial y
    cv::Mat divergence = cv::abs(dvx_dx + dvy_dy);

    // --- Angular continuity: Laplacian of omega ---
    // All points on a rigid body must rotate with the same ω.
    // Laplacian(ω) ≈ 0 for a uniform angular velocity field.
    cv::Mat laplacian_omega;
    cv::Laplacian(omega, laplacian_omega, CV_32F, 3);
    laplacian_omega = cv::abs(laplacian_omega);

    // Apply thresholds to create the continuity mask
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float div_val = divergence.at<float>(r, c);
            float lap_omega_val = laplacian_omega.at<float>(r, c);

            // Reject cells where divergence is too high (not rigid body)
            if (div_val > tau_div_) {
                mask.at<uchar>(r, c) = 0;
                continue;
            }
            // Reject cells where angular velocity is not spatially uniform
            if (lap_omega_val > tau_omega_) {
                mask.at<uchar>(r, c) = 0;
            }
        }
    }

    return mask;
}

// =============================================================================
// Section IV-C.4: Mask Integration and Application
// =============================================================================
// Combines M_p and M_c into a final mask, then zeroes out flow vectors in
// cells that are either static noise or false positives.
// =============================================================================
cv::Mat DynaTrack::apply_masks(const cv::Mat& flow, const cv::Mat& omega,
                                const cv::Mat& mask_p, const cv::Mat& mask_c) {
    // Final mask = M_p AND M_c
    cv::Mat final_mask;
    cv::bitwise_and(mask_p, mask_c, final_mask);

    // Also filter out cells with negligible flow magnitude (static regions)
    cv::Mat parts[2];
    cv::split(flow, parts);
    cv::Mat magnitude;
    cv::magnitude(parts[0], parts[1], magnitude);

    for (int r = 0; r < flow.rows; r++) {
        for (int c = 0; c < flow.cols; c++) {
            if (magnitude.at<float>(r, c) < min_flow_magnitude_) {
                final_mask.at<uchar>(r, c) = 0;
            }
        }
    }

    // Apply the mask: zero out rejected cells
    cv::Mat masked_flow = cv::Mat::zeros(flow.size(), flow.type());
    flow.copyTo(masked_flow, final_mask);

    return masked_flow;
}

// =============================================================================
// Section IV-C.5: Euclidean Distance Clustering
// =============================================================================
// Groups the remaining valid flow vectors into distinct dynamic objects using
// connected-component analysis on the binary mask. For each cluster, computes
// mean position, mean 2D velocity, and mean angular velocity as measurements
// for downstream EKF tracking.
// =============================================================================
std::vector<TrackedCluster> DynaTrack::cluster_vectors(
    const cv::Mat& masked_flow,
    const cv::Mat& masked_omega,
    const cv::Mat& final_mask) {

    std::vector<TrackedCluster> clusters;

    // Create a binary image of all valid (non-zero mask) cells
    cv::Mat binary_mask;
    final_mask.convertTo(binary_mask, CV_8UC1);
    // Threshold to ensure proper binary: any non-zero -> 255
    cv::threshold(binary_mask, binary_mask, 0, 255, cv::THRESH_BINARY);

    // Dilate slightly to bridge small gaps between adjacent moving cells
    int dilate_size = static_cast<int>(std::ceil(cluster_distance_ / 2.0));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size(dilate_size, dilate_size));
    cv::Mat dilated;
    cv::dilate(binary_mask, dilated, kernel);

    // Connected component labeling on dilated mask
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(dilated, labels, stats, centroids, 8);

    // Split masked flow for velocity extraction
    cv::Mat flow_parts[2];
    cv::split(masked_flow, flow_parts);

    // Iterate over each connected component (label 0 = background, skip it)
    for (int label = 1; label < num_labels; label++) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_cluster_size_) {
            continue;  // Skip clusters that are too small
        }

        TrackedCluster cluster;
        cluster.pixel_count = 0;
        float sum_r = 0, sum_c = 0;
        float sum_vx = 0, sum_vy = 0;
        float sum_omega = 0;

        // Accumulate only pixels that are in the original (non-dilated) mask
        for (int r = 0; r < labels.rows; r++) {
            for (int c = 0; c < labels.cols; c++) {
                if (labels.at<int>(r, c) == label &&
                    final_mask.at<uchar>(r, c) > 0) {
                    sum_r += r;
                    sum_c += c;
                    sum_vx += flow_parts[0].at<float>(r, c);
                    sum_vy += flow_parts[1].at<float>(r, c);
                    sum_omega += masked_omega.at<float>(r, c);
                    cluster.pixel_count++;
                }
            }
        }

        if (cluster.pixel_count < min_cluster_size_) {
            continue;  // After filtering with original mask, cluster too small
        }

        float n = static_cast<float>(cluster.pixel_count);
        cluster.mean_position = cv::Point2f(sum_r / n, sum_c / n);
        cluster.mean_velocity = cv::Point2f(sum_vx / n, sum_vy / n);
        cluster.mean_omega = sum_omega / n;

        clusters.push_back(cluster);
    }

    return clusters;
}

void DynaTrack::point_cloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    int size_x = std::ceil((x_max - x_min) / resolution);
    int size_y = std::ceil((y_max - y_min) / resolution);

    std::vector<std::vector<Cell>> grid(
        size_x, std::vector<Cell>(size_y));

    // 1. Accumulate points into grid cells
    for (const auto& point : cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) 
            continue;

        int i = (point.x - x_min) / resolution;
        int j = (point.y - y_min) / resolution;

        if (i >= 0 && i < size_x && j >= 0 && j < size_y) {
            grid[i][j].sum += point.z;
            grid[i][j].sum_sq += point.z * point.z;
            grid[i][j].count++;
        }
    }

    cv::Mat gray_img(size_x, size_y, CV_8UC1, cv::Scalar(0));

    // 2. Compute statistics and fill image (Combined step)
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            
            // COORDINATE MAPPING:
            // 1. Forward (X+) -> Up (Row 0)
            int row_idx = (size_x - 1) - i;
            
            // 2. Left (Y+) -> Left (Col 0)
            // Since j increases as Y goes Left (from -50 to +50),
            // and Image Col increases to the Right, we must flip j.
            int col_idx = (size_y - 1) - j; 

            if (grid[i][j].count == 0) {
                gray_img.at<uchar>(row_idx, col_idx) = 0;
                continue;
            }

            double mean = grid[i][j].sum / grid[i][j].count;
            double variance = (grid[i][j].sum_sq / grid[i][j].count) - (mean * mean);
            double stddev = std::sqrt(std::max(variance, 0.0));

            double normalized_val = (a * mean + b * stddev) / h_max;

            // Clamping to avoid overflow/wrap-around
            if (normalized_val > 1.0) normalized_val = 1.0;
            if (normalized_val < 0.0) normalized_val = 0.0;

            gray_img.at<uchar>(row_idx, col_idx) = static_cast<uchar>(normalized_val * 255.0); //unsigned charhter
        }
    }

    // Compute and publish optical flow tracking
    compute_optical_flow(gray_img);

    // Publish BEV grid map
    auto msg = cv_bridge::CvImage(cloud_msg->header, "mono8", gray_img).toImageMsg();
    image_publisher_->publish(*msg);

    RCLCPP_DEBUG(this->get_logger(),
                 "Processed grid: %dx%d", size_x, size_y);
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynaTrack>());
    rclcpp::shutdown();
    return 0;
}