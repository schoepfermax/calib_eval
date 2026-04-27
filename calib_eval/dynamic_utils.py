###############################################
# 2D SCAN-CAMERA CALIBRATION UTILITIES
###############################################

import math
import yaml
import numpy as np
import cv2


###############################################
# BASIC INTRINSICS HELPERS
###############################################

def intrinsics_dict_from_camera_info_or_loader_dict(intrinsics):
    """
    Normalize intrinsics into the common dict used across the pipeline:
        {
            'width': int,
            'height': int,
            'fx': float,
            'fy': float,
            'cx': float,
            'cy': float,
        }

    Supported input styles:
      1) Loader-style dict already containing fx/fy/cx/cy
      2) Extractor-style dict containing K or k (flat 3x3 row-major)
      3) CameraInfo-like dict containing width/height and k/K
    """
    if not isinstance(intrinsics, dict):
        raise RuntimeError(f"intrinsics must be dict, got {type(intrinsics)}")

    if all(k in intrinsics for k in ["width", "height", "fx", "fy", "cx", "cy"]):
        return {
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
        }

    K = None
    if "K" in intrinsics:
        K = intrinsics["K"]
    elif "k" in intrinsics:
        K = intrinsics["k"]
    elif "camera_matrix" in intrinsics:
        K = intrinsics["camera_matrix"]

    if K is None:
        raise RuntimeError(
            "Could not normalize intrinsics. Expected either fx/fy/cx/cy or K/k/camera_matrix."
        )

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    width = int(intrinsics.get("width", 0))
    height = int(intrinsics.get("height", 0))

    return {
        "width": width,
        "height": height,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


###############################################
# SIMPLE RIGID-BODY MATH
###############################################

def normalize_quaternion_xyzw(q):
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quaternion_to_rotation_matrix_xyzw(q):
    qx, qy, qz, qw = normalize_quaternion_xyzw(q)
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float32)


def rotation_matrix_to_quaternion_xyzw(R):
    """
    Convert 3x3 rotation matrix to quaternion [x, y, z, w].
    Simple explicit implementation to avoid extra dependencies.
    """
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return normalize_quaternion_xyzw([qx, qy, qz, qw])


def rpy_to_rotation_matrix(roll, pitch, yaw):
    cr = math.cos(float(roll))
    sr = math.sin(float(roll))
    cp = math.cos(float(pitch))
    sp = math.sin(float(pitch))
    cy = math.cos(float(yaw))
    sy = math.sin(float(yaw))

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def rotation_matrix_to_rpy(R):
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float32)


def rpy_to_quaternion_xyzw(roll, pitch, yaw):
    R = rpy_to_rotation_matrix(roll, pitch, yaw)
    return rotation_matrix_to_quaternion_xyzw(R)


def quaternion_xyzw_to_rpy(q):
    R = quaternion_to_rotation_matrix_xyzw(q)
    return rotation_matrix_to_rpy(R)


def transform_points(points_xyz, translation_xyz, quaternion_xyzw):
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    R = quaternion_to_rotation_matrix_xyzw(quaternion_xyzw)
    t = np.asarray(translation_xyz, dtype=np.float32).reshape(1, 3)
    return (pts @ R.T + t).astype(np.float32)


###############################################
# SCAN CONVERSION AND PROJECTION
###############################################

def scan_dict_to_points_lidar_frame(scan_dict, vertical_offset=0.0):
    """
    Convert stored raw 2D scan into Nx3 points in the LiDAR frame using the
    dynamic-rig projection convention.

    IMPORTANT PROJECTION NOTE
    -------------------------
    The dynamic 2D pipeline publishes extrinsics as camera->lidar and later
    projects LiDAR points into the camera image. Camera projection expects
    camera-frame depth on +Z. For this rig, the raw 2D scan must therefore be
    embedded in a camera-compatible horizontal plane before projection.

    The dynamic rig i.e. Model Car ID = 6 also has a fixed physical LiDAR yaw mounting offset of
    -45 deg. We keep dynamic_reference.yaml unchanged and compensate that
    convention here in the dynamic-only scan embedding path.

    The scan is therefore embedded after applying a fixed angle correction:
      a_corr = a - 45 deg

    Then:
      x = -r * sin(a_corr)
      y = vertical_offset
      z =  r * cos(a_corr)

    This keeps the dynamic scan geometry consistent with the rest of the 2D
    camera-projection path without changing the shared evaluator nodes or the
    reference YAML used by other experiments.
    """
    if scan_dict is None:
        return np.zeros((0, 3), dtype=np.float32)

    ranges = np.asarray(scan_dict.get("ranges", []), dtype=np.float32).reshape(-1,)
    angle_min = float(scan_dict.get("angle_min", 0.0))
    angle_increment = float(scan_dict.get("angle_increment", 0.0))
    range_min = float(scan_dict.get("range_min", 0.0))
    range_max = float(scan_dict.get("range_max", 1e9))

    if ranges.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    angles = angle_min + np.arange(ranges.shape[0], dtype=np.float32) * angle_increment
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    r = ranges[valid]
    a = angles[valid]

    # Dynamic rig fixed LiDAR mounting yaw convention correction.
    # Car 6 had -45 deg yaw.
    a_corr = a + np.deg2rad(-45.0)

    x = -r * np.sin(a_corr)
    y = np.full_like(r, float(vertical_offset), dtype=np.float32)
    z = r * np.cos(a_corr)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def project_points_camera_to_image(points_cam, intrinsics):
    intr = intrinsics_dict_from_camera_info_or_loader_dict(intrinsics)

    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    valid = z > 1e-6
    uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = fx * x[valid] / z[valid] + cx
    uv[valid, 1] = fy * y[valid] / z[valid] + cy
    return uv, valid


def filter_projected_points_inside_image(uv, image_shape):
    h, w = image_shape[:2]
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if uv.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    inside = (
        (uv[:, 0] >= 0.0) & (uv[:, 0] < float(w)) &
        (uv[:, 1] >= 0.0) & (uv[:, 1] < float(h))
    )
    return inside


def invert_transform(translation_xyz, quaternion_xyzw):
    """
    Invert a camera->lidar rigid transform into lidar->camera.

    The dynamic 2D pipeline publishes and optimizes camera->lidar extrinsics,
    but image projection requires LiDAR points to be transformed into the
    camera frame. This inversion is therefore mandatory before projection.
    """
    t = np.asarray(translation_xyz, dtype=np.float32).reshape(3,)
    q = normalize_quaternion_xyzw(quaternion_xyzw)

    R = quaternion_to_rotation_matrix_xyzw(q)
    R_inv = R.T
    t_inv = -R_inv @ t
    q_inv = rotation_matrix_to_quaternion_xyzw(R_inv)
    return t_inv.astype(np.float32), q_inv.astype(np.float32)


def project_scan_to_image(scan_dict, translation_xyz, quaternion_xyzw, intrinsics, image_shape):
    """
    Project a dynamic-rig raw scan into the image.

    INPUT EXTRINSIC CONVENTION
    --------------------------
    translation_xyz / quaternion_xyzw are expected to represent the published
    camera->lidar transform, because that is the convention used by the dynamic
    reference publisher and the 2D calibration nodes.

    PROJECTION CONVENTION
    ---------------------
    LiDAR points must be transformed into the camera frame before perspective
    projection. Therefore we first invert camera->lidar into lidar->camera.
    """
    pts_lidar = scan_dict_to_points_lidar_frame(scan_dict)
    t_lidar_cam, q_lidar_cam = invert_transform(translation_xyz, quaternion_xyzw)
    pts_cam = transform_points(pts_lidar, t_lidar_cam, q_lidar_cam)
    uv, valid_depth = project_points_camera_to_image(pts_cam, intrinsics)
    inside = filter_projected_points_inside_image(uv, image_shape)
    valid = valid_depth & inside
    return uv[valid], pts_cam[valid], pts_lidar[valid]


###############################################
# IMAGE EDGE PROCESSING
###############################################

def compute_edge_map_and_distance_transform(image_bgr, canny_low=100, canny_high=200, blur_kernel=3):
    img = np.asarray(image_bgr)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8)

    k = int(blur_kernel)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    inv = cv2.bitwise_not(edges)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size) if edges.size > 0 else 0.0
    mean_grad = float(np.mean(cv2.Laplacian(gray, cv2.CV_32F) ** 2))
    return gray, edges, dist, edge_ratio, mean_grad


def projected_edge_distances(dist_transform, projected_uv):
    uv = np.asarray(projected_uv, dtype=np.float32).reshape(-1, 2)
    if uv.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    h, w = dist_transform.shape[:2]
    dists = []
    for u, v in uv:
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if 0 <= ui < w and 0 <= vi < h:
            dists.append(float(dist_transform[vi, ui]))
    return np.asarray(dists, dtype=np.float32)


def robust_mean_distance(distances_px, clip_px=30.0):
    d = np.asarray(distances_px, dtype=np.float32).reshape(-1,)
    if d.size == 0:
        return float("inf")
    d = np.clip(d, 0.0, float(clip_px))
    return float(np.mean(d))


def compute_visibility_penalty(num_projected, min_projected_points=20, penalty_value=100.0):
    if int(num_projected) >= int(min_projected_points):
        return 0.0
    shortage = float(int(min_projected_points) - int(num_projected))
    return float(penalty_value) * shortage / max(1.0, float(min_projected_points))


def draw_scan_overlay(image_bgr, projected_uv, radius=2, color=(0, 255, 0)):
    img = image_bgr.copy()
    uv = np.asarray(projected_uv, dtype=np.float32).reshape(-1, 2)
    for u, v in uv:
        cv2.circle(img, (int(round(float(u))), int(round(float(v)))), int(radius), color, -1)
    return img


###############################################
# OFFLINE FRAME SCORING AND SELECTION
###############################################

def metadata_quality_score(sample_meta, default_edge_ratio=0.0, default_grad=0.0, default_scan_count=0):
    """
    Rank frames using extractor metadata when available.
    This keeps selection cheap and deterministic before optimization.
    """
    if sample_meta is None:
        edge_ratio = float(default_edge_ratio)
        mean_grad = float(default_grad)
        valid_scan_count = float(default_scan_count)
    else:
        edge_ratio = float(sample_meta.get("edge_ratio", default_edge_ratio))
        mean_grad = float(sample_meta.get("mean_gradient", default_grad))
        valid_scan_count = float(sample_meta.get("valid_scan_count", default_scan_count))

    # Simple explicit weighted score. We keep it interpretable.
    return 10.0 * edge_ratio + 0.001 * mean_grad + 0.01 * valid_scan_count


def select_top_k_diverse_samples(sample_entries, top_k=50, min_index_gap=3):
    """
    Each entry is expected to contain at least:
        {
            'dataset_index': int,
            'quality_score': float,
            ...
        }

    We sort by descending quality and keep a small index gap to avoid selecting
    many near-duplicate neighboring frames.
    """
    if top_k <= 0:
        return []

    entries = sorted(sample_entries, key=lambda x: float(x.get("quality_score", 0.0)), reverse=True)
    selected = []
    selected_indices = []

    for e in entries:
        idx = int(e.get("dataset_index", -10**9))
        keep = True
        for prev_idx in selected_indices:
            if abs(idx - prev_idx) < int(min_index_gap):
                keep = False
                break
        if keep:
            selected.append(e)
            selected_indices.append(idx)
        if len(selected) >= int(top_k):
            break

    return selected


###############################################
# EXTRINSIC PARAMETERIZATION FOR 2D CALIBRATION
###############################################

def extrinsic_dict_to_translation_quaternion(extrinsic_dict):
    if not isinstance(extrinsic_dict, dict):
        raise RuntimeError(f"extrinsic_dict must be dict, got {type(extrinsic_dict)}")

    t = extrinsic_dict.get("translation", {}) or {}
    q = extrinsic_dict.get("rotation_quat", {}) or {}

    translation = np.array([
        float(t.get("x", 0.0)),
        float(t.get("y", 0.0)),
        float(t.get("z", 0.0)),
    ], dtype=np.float32)

    quaternion = np.array([
        float(q.get("x", 0.0)),
        float(q.get("y", 0.0)),
        float(q.get("z", 0.0)),
        float(q.get("w", 1.0)),
    ], dtype=np.float32)

    quaternion = normalize_quaternion_xyzw(quaternion)
    return translation, quaternion


def translation_quaternion_to_reference_yaml_dict(
    translation_xyz,
    quaternion_xyzw,
    parent_frame="camera",
    child_frame="lidar",
    source="offline2d",
    units="meters_radians",
    repeatability=None,
):
    t = np.asarray(translation_xyz, dtype=np.float32).reshape(3,)
    q = normalize_quaternion_xyzw(quaternion_xyzw)

    out = {
        "parent_frame": str(parent_frame),
        "child_frame": str(child_frame),
        "units": str(units),
        "source": str(source),
        "translation": {
            "x": float(t[0]),
            "y": float(t[1]),
            "z": float(t[2]),
        },
        "rotation_quat": {
            "x": float(q[0]),
            "y": float(q[1]),
            "z": float(q[2]),
            "w": float(q[3]),
        },
    }
    if repeatability is not None:
        out["repeatability"] = repeatability
    return out


def load_reference_yaml(reference_yaml_path):
    with open(reference_yaml_path, "r") as f:
        d = yaml.safe_load(f) or {}
    return d


def save_reference_yaml(reference_yaml_path, yaml_dict):
    with open(reference_yaml_path, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)


def make_parameter_vector_from_translation_quaternion(translation_xyz, quaternion_xyzw):
    rpy = quaternion_xyzw_to_rpy(quaternion_xyzw)
    t = np.asarray(translation_xyz, dtype=np.float32).reshape(3,)
    return np.array([t[0], t[1], t[2], rpy[0], rpy[1], rpy[2]], dtype=np.float32)


def make_translation_quaternion_from_parameter_vector(param_vec):
    p = np.asarray(param_vec, dtype=np.float32).reshape(6,)
    translation = p[:3].astype(np.float32)
    quaternion = rpy_to_quaternion_xyzw(p[3], p[4], p[5])
    return translation, quaternion


def clamp_parameter_vector(param_vec, translation_bounds_xyz, rotation_bounds_rpy):
    p = np.asarray(param_vec, dtype=np.float32).reshape(6,).copy()

    for i in range(3):
        lo, hi = translation_bounds_xyz[i]
        p[i] = np.clip(p[i], float(lo), float(hi))

    for i in range(3):
        lo, hi = rotation_bounds_rpy[i]
        p[3 + i] = np.clip(p[3 + i], float(lo), float(hi))

    return p.astype(np.float32)


def build_coarse_candidate_grid(base_param_vec, translation_steps_xyz, rotation_steps_rpy, stage_mask=None):
    """
    Build a small explicit candidate set around the current estimate.

    stage_mask is a boolean iterable of length 6.
    If stage_mask[i] is False, that parameter stays fixed at the base value.
    """
    base = np.asarray(base_param_vec, dtype=np.float32).reshape(6,)
    if stage_mask is None:
        stage_mask = [True, True, True, True, True, True]

    tx_steps = translation_steps_xyz[0] if stage_mask[0] else [0.0]
    ty_steps = translation_steps_xyz[1] if stage_mask[1] else [0.0]
    tz_steps = translation_steps_xyz[2] if stage_mask[2] else [0.0]
    rr_steps = rotation_steps_rpy[0] if stage_mask[3] else [0.0]
    rp_steps = rotation_steps_rpy[1] if stage_mask[4] else [0.0]
    ry_steps = rotation_steps_rpy[2] if stage_mask[5] else [0.0]

    out = []
    for dtx in tx_steps:
        for dty in ty_steps:
            for dtz in tz_steps:
                for drr in rr_steps:
                    for drp in rp_steps:
                        for dry in ry_steps:
                            p = base.copy()
                            p[0] += float(dtx)
                            p[1] += float(dty)
                            p[2] += float(dtz)
                            p[3] += float(drr)
                            p[4] += float(drp)
                            p[5] += float(dry)
                            out.append(p.astype(np.float32))
    return out


###############################################
# OFFLINE COST EVALUATION
###############################################

def evaluate_single_frame_edge_cost(
    image_bgr,
    scan_dict,
    intrinsics,
    translation_xyz,
    quaternion_xyzw,
    canny_low=100,
    canny_high=200,
    blur_kernel=3,
    clip_px=30.0,
    min_projected_points=20,
    visibility_penalty_value=100.0,
):
    """
    Compute a single-frame edge-alignment cost for one candidate extrinsic.
    Lower is better.
    """
    _, _, dist, edge_ratio, mean_grad = compute_edge_map_and_distance_transform(
        image_bgr=image_bgr,
        canny_low=canny_low,
        canny_high=canny_high,
        blur_kernel=blur_kernel,
    )

    projected_uv, _, _ = project_scan_to_image(
        scan_dict=scan_dict,
        translation_xyz=translation_xyz,
        quaternion_xyzw=quaternion_xyzw,
        intrinsics=intrinsics,
        image_shape=image_bgr.shape,
    )

    dists = projected_edge_distances(dist, projected_uv)
    edge_cost = robust_mean_distance(dists, clip_px=clip_px)
    visibility_penalty = compute_visibility_penalty(
        num_projected=len(projected_uv),
        min_projected_points=min_projected_points,
        penalty_value=visibility_penalty_value,
    )

    total_cost = edge_cost + visibility_penalty

    return {
        "total_cost": float(total_cost),
        "edge_cost": float(edge_cost),
        "visibility_penalty": float(visibility_penalty),
        "num_projected": int(len(projected_uv)),
        "edge_ratio": float(edge_ratio),
        "mean_gradient": float(mean_grad),
        "projected_uv": projected_uv,
    }


def evaluate_multiframe_edge_cost(sample_bundle_list, translation_xyz, quaternion_xyzw, **kwargs):
    """
    sample_bundle_list elements are expected to contain:
        {
            'image': ...,
            'scan': ...,
            'intrinsics': ...,
            'weight': optional float,
        }
    """
    if len(sample_bundle_list) == 0:
        return {
            "mean_total_cost": float("inf"),
            "mean_edge_cost": float("inf"),
            "mean_visibility_penalty": float("inf"),
            "mean_num_projected": 0.0,
            "frame_results": [],
        }

    total_weight = 0.0
    sum_total = 0.0
    sum_edge = 0.0
    sum_vis = 0.0
    sum_num = 0.0
    frame_results = []

    for bundle in sample_bundle_list:
        w = float(bundle.get("weight", 1.0))
        res = evaluate_single_frame_edge_cost(
            image_bgr=bundle["image"],
            scan_dict=bundle["scan"],
            intrinsics=bundle["intrinsics"],
            translation_xyz=translation_xyz,
            quaternion_xyzw=quaternion_xyzw,
            **kwargs,
        )
        frame_results.append(res)
        total_weight += w
        sum_total += w * float(res["total_cost"])
        sum_edge += w * float(res["edge_cost"])
        sum_vis += w * float(res["visibility_penalty"])
        sum_num += w * float(res["num_projected"])

    total_weight = max(total_weight, 1e-12)
    return {
        "mean_total_cost": float(sum_total / total_weight),
        "mean_edge_cost": float(sum_edge / total_weight),
        "mean_visibility_penalty": float(sum_vis / total_weight),
        "mean_num_projected": float(sum_num / total_weight),
        "frame_results": frame_results,
    }


def coarse_to_fine_search(
    sample_bundle_list,
    init_param_vec,
    translation_bounds_xyz,
    rotation_bounds_rpy,
    stage_a_mask,
    stage_a_translation_steps_xyz,
    stage_a_rotation_steps_rpy,
    stage_b_translation_steps_xyz,
    stage_b_rotation_steps_rpy,
    evaluation_kwargs,
):
    """
    Explicit two-stage search:
      - Stage A: constrained coarse candidate search
      - Stage B: full local refinement around the best Stage A candidate
    """
    base = np.asarray(init_param_vec, dtype=np.float32).reshape(6,)

    best_param = clamp_parameter_vector(base, translation_bounds_xyz, rotation_bounds_rpy)
    best_t, best_q = make_translation_quaternion_from_parameter_vector(best_param)
    best_eval = evaluate_multiframe_edge_cost(sample_bundle_list, best_t, best_q, **evaluation_kwargs)

    # Stage A
    candidates_a = build_coarse_candidate_grid(
        base_param_vec=best_param,
        translation_steps_xyz=stage_a_translation_steps_xyz,
        rotation_steps_rpy=stage_a_rotation_steps_rpy,
        stage_mask=stage_a_mask,
    )

    for cand in candidates_a:
        cand = clamp_parameter_vector(cand, translation_bounds_xyz, rotation_bounds_rpy)
        t, q = make_translation_quaternion_from_parameter_vector(cand)
        ev = evaluate_multiframe_edge_cost(sample_bundle_list, t, q, **evaluation_kwargs)
        if float(ev["mean_total_cost"]) < float(best_eval["mean_total_cost"]):
            best_param = cand
            best_eval = ev

    # Stage B
    candidates_b = build_coarse_candidate_grid(
        base_param_vec=best_param,
        translation_steps_xyz=stage_b_translation_steps_xyz,
        rotation_steps_rpy=stage_b_rotation_steps_rpy,
        stage_mask=[True, True, True, True, True, True],
    )

    for cand in candidates_b:
        cand = clamp_parameter_vector(cand, translation_bounds_xyz, rotation_bounds_rpy)
        t, q = make_translation_quaternion_from_parameter_vector(cand)
        ev = evaluate_multiframe_edge_cost(sample_bundle_list, t, q, **evaluation_kwargs)
        if float(ev["mean_total_cost"]) < float(best_eval["mean_total_cost"]):
            best_param = cand
            best_eval = ev

    best_t, best_q = make_translation_quaternion_from_parameter_vector(best_param)
    return {
        "best_param_vec": best_param,
        "best_translation": best_t,
        "best_quaternion": best_q,
        "best_eval": best_eval,
    }


###############################################
# ONLINE-STYLE REFINEMENT HELPERS
###############################################

def quaternion_angle_deg(q1, q2):
    qa = normalize_quaternion_xyzw(q1)
    qb = normalize_quaternion_xyzw(q2)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * math.acos(dot)))


def translation_distance(t1, t2):
    a = np.asarray(t1, dtype=np.float32).reshape(3,)
    b = np.asarray(t2, dtype=np.float32).reshape(3,)
    return float(np.linalg.norm(a - b))


def pose_dict_to_translation_quaternion(odom_pose_dict):
    pos = odom_pose_dict.get("position", {}) or {}
    ori = odom_pose_dict.get("orientation", {}) or {}
    t = np.array([
        float(pos.get("x", 0.0)),
        float(pos.get("y", 0.0)),
        float(pos.get("z", 0.0)),
    ], dtype=np.float32)
    q = np.array([
        float(ori.get("x", 0.0)),
        float(ori.get("y", 0.0)),
        float(ori.get("z", 0.0)),
        float(ori.get("w", 1.0)),
    ], dtype=np.float32)
    return t, normalize_quaternion_xyzw(q)


def odom_window_motion_metrics(odom_sequence):
    """
    Compute motion observability metrics from an odom window.
    """
    if odom_sequence is None or len(odom_sequence) < 2:
        return {
            "translation_total": 0.0,
            "rotation_total_deg": 0.0,
            "is_motion_usable": False,
        }

    t0, q0 = pose_dict_to_translation_quaternion(odom_sequence[0])
    t1, q1 = pose_dict_to_translation_quaternion(odom_sequence[-1])

    trans = translation_distance(t0, t1)
    rot_deg = quaternion_angle_deg(q0, q1)

    return {
        "translation_total": float(trans),
        "rotation_total_deg": float(rot_deg),
        "is_motion_usable": bool((trans > 0.01) or (rot_deg > 1.0)),
    }


def refinement_should_accept_update(
    previous_cost,
    candidate_cost,
    translation_step,
    rotation_step_deg,
    max_translation_step=0.20,
    max_rotation_step_deg=10.0,
    min_improvement=1e-4,
):
    """
    Acceptance logic for online-style refinement.
    """
    if translation_step > float(max_translation_step):
        return False
    if rotation_step_deg > float(max_rotation_step_deg):
        return False
    if candidate_cost >= previous_cost - float(min_improvement):
        return False
    return True


def deterioration_detected(best_cost, recent_costs, patience=3, tolerance=1e-4):
    if recent_costs is None or len(recent_costs) < int(patience):
        return False

    tail = recent_costs[-int(patience):]
    for c in tail:
        if float(c) <= float(best_cost) + float(tolerance):
            return False
    return True
