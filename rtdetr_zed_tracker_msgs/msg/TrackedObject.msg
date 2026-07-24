# One tracked object: stable ID + class + depth-derived 3D position.

string track_id            # display id, stable for the track's life, e.g. "red_buoy_2"
string class_name          # frozen class label (frozen at confirmation)
int32 class_index          # frozen class index
float32 confidence         # latest detection score

# Distances. NEVER a bare "distance" field: forward Z and Euclidean range diverge
# by 30%+ for off-axis objects in the ZED's wide FOV. Z is normally what obstacle
# logic wants. Both are NaN when depth_valid is false (do not substitute 0).
float32 distance_z_m       # forward along the optical axis (Z), metres
float32 distance_range_m   # sqrt(X^2 + Y^2 + Z^2), metres

geometry_msgs/Point position   # (X,Y,Z) in the camera optical frame, metres; NaN if invalid

float32 depth_valid_ratio  # fraction of valid depth pixels in the sampling ROI, [0..1]
uint32 age_frames          # frames since the track was created
bool depth_valid           # true iff a usable depth measurement was obtained this cycle
