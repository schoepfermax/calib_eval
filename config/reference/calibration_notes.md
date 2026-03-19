# Camera–LiDAR Calibration Notes (Static Rig h1)

## Date

Calibration finalized: 2026-03-15

## Sensors

Camera: Basler camera (ROS2 pylon driver)
LiDAR: 2D LiDAR (static mount)

Rig configuration: **static_mount_h1**

Approximate physical measurements:

* Base mounting plate width: **23.5 cm**
* Approximate camera–LiDAR center separation: **~11.5 cm**
* Relative rotation between sensors: **approximately zero (visually aligned)**

---

# Initial Calibration Attempt

Extrinsic calibration was first attempted using the **Abdul Haq calibration tool**, which estimates the transform using:

* ArUco marker detection in camera images
* manually selected LiDAR points corresponding to marker corners
* solvePnPRansac optimization.

## Observed Problems

Multiple calibration runs produced unstable results:

| Run          | Estimated Translation Magnitude |
| ------------ | ------------------------------- |
| Example runs | 1 m – 13 m                      |

Despite low reprojection errors, the estimated transforms were **physically impossible** for the sensor rig.

Likely causes:

* planar calibration geometry
* sparse correspondences
* sensitivity of PnP to noise
* correspondence ordering ambiguity

Because of this instability, the results from the Abdul Haq tool were deemed **unreliable for this rig configuration**.

---

# Alternative Calibration Strategy

A **direct rigid alignment approach using SVD (Umeyama method)** was implemented.

This method estimates the rigid transform between two corresponding 3D point sets.

---

# Correspondence Data

Each capture saved **5 correspondences**:

* 4 marker corner points
* 1 marker center point

For calibration, **only the 4 corner points were used**.

Total captures collected:

```
7 captures × 4 corners = 28 correspondences
```

Files used:

```
img_points.txt
lidar_points.txt
```

Each correspondence pair consisted of:

Image coordinates:

```
(u, v)
```

LiDAR coordinates:

```
(x, y, z)
```

---

# Camera Back-Projection

Using camera intrinsics:

```
fx = 2000
fy = 2000
cx = 1300
cy = 1064
```

image points were converted into **unit rays in the camera frame**.

---

# Approximate Camera 3D Points

Each ray was scaled using the Euclidean distance of the corresponding LiDAR point:

```
p_cam = d_lidar * ray
```

This produces approximate camera-frame 3D coordinates.

---

# Rigid Transform Estimation

The rigid transform between the LiDAR points and camera points was estimated using **SVD-based least squares alignment**:

```
R, t = argmin Σ || R p_i + t − q_i ||
```

where:

```
p_i = LiDAR points
q_i = scaled camera points
```

---

# Translation Normalization

Because ray scaling introduces scale ambiguity, the translation vector magnitude was normalized using the measured rig baseline:

```
||t|| ≈ 0.115 m
```

---

# Validation Procedure

The resulting extrinsic transform was validated by projecting LiDAR points into the camera image.

Validation script:

```
validate_extrinsic_projection.py
```

Result:

```
points projected inside image: 28
total points: 28
ratio: 1.0
```

All correspondences projected within image bounds, confirming that the transformation is **geometrically consistent**.

---

# Final Reference Extrinsic

The final transformation is stored in:

```
static_mount_h1_reference.yaml
```

This file replaces the earlier calibration produced by the Abdul Haq tool.

Important:

This transformation is treated as a **reference extrinsic**, not strict ground truth.

The evaluation pipeline measures **temporal stability and calibration correction performance relative to this baseline**.

---

# Lessons Learned

1. PnP-based calibration is unstable for sparse planar correspondences.
2. Direct rigid alignment can produce more stable results when combined with physical constraints.
3. Rig measurements provide useful scale constraints.
4. Projection validation is a simple but effective sanity check.
