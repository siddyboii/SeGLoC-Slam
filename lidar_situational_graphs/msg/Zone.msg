# Zone.msg
std_msgs/Header header

int32 zone_id
int32 floor_id
geometry_msgs/Pose centroid
geometry_msgs/Polygon polygon

int32[] room_ids
int32[] keyframe_ids

# semantic labels inferred (optional)
string[] top_labels
float32[] top_label_confidences
float32 confidence

# structural support for later g2o integration
situational_graphs_msgs/PlaneData[] supporting_planes
int32[] supporting_wall_ids

uint8 action    # 0=CREATE,1=UPDATE,2=DELETE
uint64 version
