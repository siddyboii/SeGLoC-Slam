# Dynamic object detection summary from BEV optical flow tracking
# Published by DynaTrack node, consumed by s_graphs_node

std_msgs/Header header

# Number of detected dynamic clusters in this frame
int32 num_dynamic_clusters

# Per-cluster data (parallel arrays)
float32[] cluster_pos_x        # Mean position X in BEV (pixels)
float32[] cluster_pos_y        # Mean position Y in BEV (pixels)
float32[] cluster_vel_x        # Mean velocity X (pixels/frame)
float32[] cluster_vel_y        # Mean velocity Y (pixels/frame)
float32[] cluster_omega        # Mean angular velocity (rad/frame)
int32[]   cluster_pixel_count  # Number of pixels per cluster

# Scene-level dynamicity summary
float32 scene_dynamicity       # Overall scene dynamicity score [0, 1]
                               # 0 = fully static, 1 = highly dynamic
int32 total_dynamic_pixels     # Total pixels classified as dynamic
int32 total_occupied_pixels    # Total occupied (non-zero) pixels in BEV
