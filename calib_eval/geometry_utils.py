###############################################
# GEOMETRY UTILITIES FOR PIPELINE
###############################################

import numpy as np
import cv2


###############################################
# QUATERNION AND TRANSFORM UTILITIES
###############################################

def quaternion_to_rotation_matrix(q):
    """
    Converts quaternion q = [qx, qy, qz, qw] into a 3x3 rotation matrix.

    This function normalizes the quaternion to avoid numerical issues.
    """
    qx, qy, qz, qw = q

    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-12:
        # Degenerate quaternion; fall back to identity rotation
        return np.eye(3, dtype=np.float32)

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float32)

    return R


def invert_extrinsics_cam_to_lidar_to_lidar_to_cam(t_cam_lidar, q_cam_lidar_xyzw):
    """
    Invert a camera->lidar rigid transform into lidar->camera.

    Pipeline context:
      - /eval/ref_extrinsics and /eval/estimated_extrinsics are published as:
            header.frame_id = camera
            child_frame_id  = lidar
        i.e. the transform represents camera -> lidar.
      - Projection utilities apply:
            p_cam = R * p_lidar + t
        therefore they expect lidar -> camera.

    Math:
      Given T_cl (camera -> lidar):  p_lidar = R_cl * p_cam + t_cl
      The inverse T_lc (lidar -> camera) is:
            R_lc = R_cl^T
            t_lc = -R_cl^T * t_cl
      For a unit quaternion in xyzw, inverse rotation is the conjugate:
            q_lc = [-x, -y, -z, w]

    Args:
        t_cam_lidar: (3,) translation (camera -> lidar)
        q_cam_lidar_xyzw: (4,) quaternion [x,y,z,w] (camera -> lidar)

    Returns:
        t_lidar_cam: (3,) translation (lidar -> camera)
        q_lidar_cam_xyzw: (4,) quaternion [x,y,z,w] (lidar -> camera)
    """
    t = np.asarray(t_cam_lidar, dtype=np.float32).reshape(3,)
    q = np.asarray(q_cam_lidar_xyzw, dtype=np.float32).reshape(4,)

    # Normalize defensively (should already be unit, but avoid malformed inputs).
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q = q / n
    else:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    R_cl = quaternion_to_rotation_matrix(q)
    R_lc = R_cl.T
    t_lidar_cam = -(R_lc @ t)

    q_lidar_cam = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    return t_lidar_cam.astype(np.float32), q_lidar_cam.astype(np.float32)


def transform_points(points, translation, quaternion):
    """
    Applies rigid transform to points.

    Args:
        points: Nx3 array in LiDAR frame
        translation: [tx, ty, tz]
        quaternion: [qx, qy, qz, qw]

    Returns:
        points_cam: Nx3 array in camera frame
    """
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    R = quaternion_to_rotation_matrix(quaternion)
    t = np.array(translation, dtype=np.float32).reshape(3, 1)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3).T  # 3xN
    points_cam = (R @ points) + t
    return points_cam.T  # Nx3


###############################################
# CAMERA PROJECTION UTILITIES
###############################################

def project_points_to_image(points_cam, intrinsics):
    """
    Projects 3D points in camera frame into pixel coordinates.

    intrinsics dict format:
        fx, fy, cx, cy

    Returns:
        uv: Nx2 pixel coords
        valid: boolean mask for z>0
    """
    fx = float(intrinsics['fx'])
    fy = float(intrinsics['fy'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])

    uv = np.zeros((len(points_cam), 2), dtype=np.float32)
    valid = np.zeros((len(points_cam),), dtype=bool)

    for i, (x, y, z) in enumerate(points_cam):
        if z <= 0:
            continue

        u = fx * x / z + cx
        v = fy * y / z + cy

        uv[i] = [u, v]
        valid[i] = True

    return uv, valid


def check_image_bounds(uv, image_shape):
    """
    Returns boolean mask for whether points are inside image bounds.
    """
    height, width = image_shape[:2]
    inside = np.zeros((len(uv),), dtype=bool)

    for i, (u, v) in enumerate(uv):
        inside[i] = (0 <= int(u) < width) and (0 <= int(v) < height)

    return inside


def project_lidar_to_image(points_lidar, translation, quaternion, intrinsics, image_shape):
    """
    Convenience wrapper used by evaluators.

    Returns:
        visible_uv: Mx2 (u,v) for points that are in front of camera AND inside image
        visible_cam_points: Mx3 corresponding camera-frame points
    """
    points_cam = transform_points(points_lidar, translation, quaternion)
    uv, valid_depth = project_points_to_image(points_cam, intrinsics)
    inside = check_image_bounds(uv, image_shape)

    valid = valid_depth & inside
    return uv[valid], points_cam[valid]


###############################################
# PIXEL-ERROR UTILITIES
###############################################

def compute_edge_distance_transform(image_bgr, canny_low=100, canny_high=200):
    """
    Computes a distance transform to nearest edge pixel.

    Steps:
      1) Canny edge detection -> binary edge map
      2) Invert edge map (distance transform expects non-zero foreground)
      3) Distance transform -> for each pixel, distance to nearest edge

    Returns:
      edges: uint8 edge image (0 or 255)
      dist: float32 distance transform (pixels)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Distance transform computes distances to zero pixels, so invert:
    # edges==255 -> becomes 0 (edge locations)
    inv = cv2.bitwise_not(edges)

    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    return edges, dist


def sample_edge_distances_px(dist_transform, projected_uv):
    """
    Samples distance-transform values at projected point locations.

    This keeps all distance-based edge sampling logic in one place so both
    evaluators use the same definition.

    Args:
        dist_transform: HxW float32 distance transform
        projected_uv: Mx2 array of (u, v) float pixel coordinates

    Returns:
        sampled_distances_px: N float32 distances for points inside image bounds
        num_used: int
    """
    if projected_uv is None or len(projected_uv) == 0:
        return np.zeros((0,), dtype=np.float32), 0

    h, w = dist_transform.shape[:2]
    dists = []

    for u, v in projected_uv:
        ui = int(u)
        vi = int(v)

        if 0 <= ui < w and 0 <= vi < h:
            dists.append(float(dist_transform[vi, ui]))

    if len(dists) == 0:
        return np.zeros((0,), dtype=np.float32), 0

    return np.asarray(dists, dtype=np.float32), len(dists)


def mean_edge_distance_px(dist_transform, projected_uv):
    """
    Computes mean pixel distance of projected points to nearest edge.

    Args:
        dist_transform: HxW float32
        projected_uv: Mx2 array of (u,v) float pixel coordinates

    Returns:
        mean_distance_px: float (pixels)
        num_used: int
    """
    dists, num_used = sample_edge_distances_px(dist_transform, projected_uv)

    if num_used == 0:
        return float('inf'), 0

    return float(np.mean(dists)), num_used


def edge_hit_ratio_with_threshold(dist_transform, projected_uv, hit_threshold_px):
    """
    Computes a threshold-based edge hit ratio.

    A projected point counts as a hit if its nearest-edge distance is less than
    or equal to hit_threshold_px. This is more stable than requiring exact
    overlap with a single Canny edge pixel.

    Args:
        dist_transform: HxW float32 distance transform
        projected_uv: Mx2 array of (u, v) float pixel coordinates
        hit_threshold_px: float threshold in pixels

    Returns:
        hit_ratio: float in [0, 1]
        num_hits: int
        num_used: int
    """
    dists, num_used = sample_edge_distances_px(dist_transform, projected_uv)

    if num_used == 0:
        return 0.0, 0, 0

    threshold = float(hit_threshold_px)
    num_hits = int(np.sum(dists <= threshold))
    hit_ratio = float(num_hits) / float(num_used)

    return hit_ratio, num_hits, num_used


###############################################
# EXTRINSICS COMPARISON UTILITIES
###############################################

def quat_normalize_xyzw(q):
    """
    Normalizes quaternion q = [qx, qy, qz, qw].

    If the norm is ~0, returns identity quaternion.

    Note:
      This duplicates the normalization logic used inside
      quaternion_to_rotation_matrix(), but is provided here as a standalone
      utility for evaluators that operate directly on quaternions.
    """
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_geodesic_angle_deg_xyzw(q1, q2):
    """
    Computes the geodesic angle between two unit quaternions in degrees.

    Using abs(dot) makes it invariant to sign flip (q and -q represent
    the same rotation).

    Args:
        q1, q2: quaternions [qx, qy, qz, qw]

    Returns:
        angle_deg: float
    """
    q1 = quat_normalize_xyzw(q1)
    q2 = quat_normalize_xyzw(q2)

    dot = float(abs(np.dot(q1, q2)))
    dot = max(min(dot, 1.0), -1.0)

    return float(2.0 * np.arccos(dot) * 180.0 / np.pi)


def welford_update(n, mean, m2, x):
    """
    One-step Welford update for running mean/variance.

    Args:
        n: current sample count (int)
        mean: current running mean (float)
        m2: current running sum of squares of differences (float)
        x: new sample (float)

    Returns:
        (n_new, mean_new, m2_new)
    """
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    delta2 = x - mean_new
    m2_new = m2 + delta * delta2
    return n_new, mean_new, m2_new


def welford_std(n, m2):
    """
    Returns sample standard deviation if n >= 2, else 0.0.
    """
    if n < 2:
        return 0.0
    var = m2 / (n - 1)
    return float(np.sqrt(max(var, 0.0)))


###############################################
# LASERSCAN (2D) -> PSEUDO REPRESENTATIONS
###############################################

def laserscan_to_points_xy_plane(ranges, angle_min, angle_increment, range_min=0.0, range_max=1e9):
    """
    Converts a 2D LaserScan (ranges + angles) into an Nx3 point set in the LiDAR frame, on the XY plane.

    Output convention:
      - x = r * cos(theta)
      - y = r * sin(theta)
      - z = 0.0

    Filtering:
      - ignores NaN / inf
      - ignores values outside [range_min, range_max]

    This is used for the dynamic rig to create a "pseudo point cloud" so the rest of the pipeline
    can stay PointCloud2-based (wrappers + evaluators).
    """
    if ranges is None:
        return np.zeros((0, 3), dtype=np.float32)

    r = np.asarray(ranges, dtype=np.float32).reshape(-1,)
    if r.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    valid = np.isfinite(r)
    valid = valid & (r >= float(range_min)) & (r <= float(range_max))

    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    theta = float(angle_min) + idx.astype(np.float32) * float(angle_increment)
    rr = r[idx]

    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    z = np.zeros_like(x, dtype=np.float32)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def laserscan_to_depth_like_image(ranges, out_width=None, out_height=64, range_min=0.0, range_max=50.0):
    """
    Builds a "depth-like" single-channel image from a 2D LaserScan range vector.

    IMPORTANT LIMITATION:
      A 2D LaserScan cannot be projected into a true camera-aligned depth image without knowing
      camera-lidar extrinsics (which is what we estimate). Therefore we generate a stable "scan image"
      that preserves the range signal in an image-shaped tensor.

    Representation:
      - Take the 1D ranges vector (N beams)
      - Optionally resample to out_width
      - Repeat vertically to out_height (HxW)
      - Output dtype float32 with range values in meters

    Args:
        ranges: list/array of N float ranges (meters)
        out_width: int or None. If None, uses N beams as width.
        out_height: int. Vertical repetition factor.
        range_min/range_max: for clipping invalid/too-large values.

    Returns:
        depth_img: (H, W) float32 image, meters.
    """
    if ranges is None:
        return np.zeros((int(out_height), 0), dtype=np.float32)

    r = np.asarray(ranges, dtype=np.float32).reshape(-1,)
    if r.size == 0:
        return np.zeros((int(out_height), 0), dtype=np.float32)

    # Replace invalid values with 0, then clip.
    valid = np.isfinite(r)
    r_clean = np.where(valid, r, 0.0).astype(np.float32)
    r_clean = np.clip(r_clean, float(range_min), float(range_max))

    # Optional resample in width.
    if out_width is None:
        r_resampled = r_clean
    else:
        W = int(out_width)
        if W <= 0:
            W = r_clean.size
        x_old = np.linspace(0.0, 1.0, num=r_clean.size, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=W, dtype=np.float32)
        r_resampled = np.interp(x_new, x_old, r_clean).astype(np.float32)

    H = int(out_height)
    if H <= 0:
        H = 1

    return np.repeat(r_resampled.reshape(1, -1), repeats=H, axis=0).astype(np.float32)