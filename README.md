# calib_eval

ROS 2 framework for evaluation and deployment of camera–LiDAR extrinsic calibration methods for mobile sensor setups.

This repository contains the final implementation developed as part of a Master’s thesis in Autonomous Driving.

---

## Overview

`calib_eval` is designed as both:

* an **evaluation framework** for comparing calibration methods, and
* a **practical calibration tool**, capable of saving estimated extrinsics at the end of a run.

The system supports multiple calibration paradigms and operates under realistic constraints where true ground-truth extrinsics are unavailable.

---

## Key Features

* Unified ROS 2 pipeline for calibration evaluation
* Static **3D calibration** (camera + point cloud)
* Dynamic **2D calibration** (camera + LaserScan)
* Integration of deep learning-based calibration models:

  * **LCCNet**
  * **BEVCalib**
* Native **2D calibration pipeline**:

  * `offline2d` (coarse estimation)
  * `online2d` (motion-based refinement)
* Evaluation metrics:

  * Translation / rotation deviation (primary)
  * Reprojection error (diagnostic)
  * Edge alignment (diagnostic)
* Final transform saving for deployment use
* Visualization of projected LiDAR on camera images

---

## Important Note on Ground Truth

No true ground-truth extrinsics are available for the custom static or dynamic rigs used in this project.

All evaluations are therefore **reference-relative**, meaning:

* Metrics represent deviation from a **baseline/reference extrinsic calibration**
* Absolute accuracy cannot be guaranteed
* Temporal stability and consistency are key evaluation criteria

---

## System Architecture

Typical pipeline:

```
Dataset → Preprocessing → 3D Inference / 2D Solver → Evaluators → Aggregation → Output
```

### Main Nodes

* `data_preprocessor_node`
* `supervised_model_node` (LCCNet)
* `calib_model_node` (BEVCalib)
* `offline2d`
* `online2d`
* `reprojection_evaluator_node`
* `edge_alignment_evaluator_node`
* `baseline_deviation_evaluator_node`
* `evaluation_pipeline_node`
* `reference_extrinsics_publisher`
* `extrinsic_projection_visualizer`
* `save_final_transform`

---

## Transform Convention

All estimated extrinsics are published as:

```
camera → lidar
```

Topic:

```
/eval/estimated_extrinsics
```

Reference extrinsics follow the same convention.

---

## Supported Modes

### 1. 3D Calibration (Model-Based)

* Input: Image + Point Cloud
* Models:

  * LCCNet
  * BEVCalib
* Used for:

  * Static rigs (h1, h2)
  * Cross-evaluation
  * Dynamic rig via pseudo point cloud

---

### 2. 2D Calibration (Dynamic Rig)

* Input: Image + LaserScan (+ Odometry)
* Pipeline:

  * `offline2d` → coarse estimate
  * `online2d` → refinement
* Used for:

  * Dynamic mobile setups
  * Real-time feasibility

---

## Repository Contents

This repository includes:

* ROS 2 package (`calib_eval`)
* Launch files for:

  * static 3D evaluation
  * dynamic 2D evaluation
  * visualization
* Configuration files
* Reference extrinsics
* Evaluation pipeline and metrics
* Model integration wrappers
* Third-party model code for:

  * LCCNet
  * BEVCalib

Datasets and checkpoints are **not stored in this repository**.

---

## Dependencies

### System Requirements

* Ubuntu 22.04
* ROS 2 Humble
* Python 3.10

### ROS Dependencies

Declared in `package.xml`:

* rclpy
* sensor_msgs
* geometry_msgs
* std_msgs
* tf2_ros
* cv_bridge
* numpy
* OpenCV

Install via:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

---

### Python / Model Dependencies

```bash
pip install torch torchvision numpy opencv-python scipy pyyaml
```

Notes:

* CUDA is optional but recommended for model inference
* PyTorch version must match CUDA setup

---

## Build Instructions

From workspace root:

```bash
cd ~/calib_ws
colcon build --packages-select calib_eval
source install/setup.bash
```

---

## Dataset Extraction

The pipeline expects datasets in a standardized **run-based structure**.  
Two dataset extractors are provided to generate this structure directly from ROS topics.

---

### Overview

The extractors:

* Record synchronized sensor data  
* Apply basic quality and synchronization checks  
* Save data in a format directly compatible with the evaluation pipeline  
* Automatically organize data into `run_XXX` folders  

Each run is saved as:

```
run_XXX/
  images/
  lidar/        # point cloud OR pseudo-3D
  scans/        # raw LaserScan (dynamic only)
  odom/         # odometry (dynamic only)
  intrinsics/
  index/
  meta/
```

---

## Static Rig Extraction (3D)

Used for:

* camera + 3D LiDAR datasets  

### Inputs

* Image (`sensor_msgs/Image`)  
* Point cloud (`sensor_msgs/PointCloud2`)  
* Camera intrinsics (`CameraInfo`)  

### Command

```bash
ros2 run calib_eval extract_static_rig_dataset --ros-args \
  -p camera_topic:=/basler/camera/image_raw \
  -p points_topic:=/ouster/points \
  -p camera_info_topic:=/basler/camera/camera_info \
  -p output_root:=/home/$USER/dataset_root/static_mount_h1
```

---

## Dynamic Rig Extraction (2D)

Used for:

* camera + LaserScan datasets  

### Inputs

* Image (`sensor_msgs/Image`)  
* LaserScan (`sensor_msgs/LaserScan`)  
* Odometry (`nav_msgs/Odometry`)  
* Camera intrinsics (`CameraInfo`)  

### Command

```bash
ros2 run calib_eval extract_dynamic_rig_dataset --ros-args \
  -p camera_topic:=/camera/image_raw \
  -p scan_topic:=/lidar/scan \
  -p camera_info_topic:=/camera/camera_info \
  -p odom_topic:=/odom \
  -p output_root:=/home/$USER/dataset_root/dynamic_rig
```


### Saved Outputs

Primary (2D pipeline):

* Raw LaserScan → `.npz` (in `scans/`)  

Compatibility (3D pipeline):

* Pseudo point cloud → `.pcd` (in `lidar/`)  

Additional:

* Images (`.png`)  
* Odometry (`.yaml`)  
* Per-sample metadata (`meta/`)  
* Run summary (`run_summary.yaml`)  

---

## Launch Usage

The framework is controlled through ROS 2 launch parameters, enabling flexible configuration without modifying code.

---

### Static 3D Evaluation

```bash
ros2 launch calib_eval offline_eval_static.launch.py
```

### Key Parameters (Static 3D Evaluation)

- `dataset_root` → Path to dataset root directory  
- `split` → Dataset split (`train`, `val`, `test`)  
- `rig_config_id` → Reference configuration (`static_mount_h1`, `static_mount_h2`, `dynamic`)  
- `use_dynamic_rig` → Enable dynamic-rig conventions (`true` / `false`)  
- `dynamic_representation` → Representation for dynamic rig:
  - `pseudo_points` → use stored pseudo point cloud (for 3D models)
  - `raw_scan` → use raw LaserScan (not typical for 3D models)

---

### Model Selection

- `model_node_executable` → Select calibration model wrapper:
  - `supervised_model_node` → LCCNet  
  - `calib_model_node` → BEVCalib  

- `checkpoint_path` → Path to model weights  

- `device` → Inference device:
  - `cpu`
  - `cuda`

---

### LCCNet Runtime Parameters

- `lccnet_mode` → Execution mode:
  - `single_pass` → one-shot prediction  
  - `iterative` → multi-stage refinement  

- `lccnet_iterative_steps` → Number of refinement iterations (1-5, used only in `iterative` mode)
---

#### Example: LCCNet

```bash
ros2 launch calib_eval offline_eval_static.launch.py \
  dataset_root:=/path/to/static_mount_h1 \
  split:=val \
  rig_config_id:=static_mount_h1 \
  model_node_executable:=supervised_model_node \
  checkpoint_path:=/path/to/lccnet.tar \
  device:=cuda
```

---

#### Example: BEVCalib

```bash
ros2 launch calib_eval offline_eval_static.launch.py \
  dataset_root:=/path/to/static_mount_h1 \
  split:=val \
  rig_config_id:=static_mount_h1 \
  model_node_executable:=calib_model_node \
  checkpoint_path:=/path/to/bevcalib.pth \
  device:=cuda
```

---

### Dynamic 2D Evaluation

```bash
ros2 launch calib_eval offline_eval_dynamic.launch.py
```

### Key Parameters (Dynamic 2D Evaluation)

- `dataset_root` → Path to dynamic rig dataset  
- `split` → Dataset split (`train`, `val`, `test`)  
- `rig_config_id` → Typically `dynamic`  
- `use_dynamic_rig` → Should be `true` for 2D pipeline  

---

### Online2D Refinement Parameters

- `online_window_size` → Rolling window size for refinement  
- `online_min_window_size` → Minimum valid window size before updates  
- `online_evaluation_stride` → Frame stride for evaluation  
- `online_min_improvement` → Minimum improvement required to accept an update  
- `online_deterioration_patience` → Number of worsening windows tolerated  
- `online_deterioration_tolerance` → Threshold for deterioration detection  

---

### Search Parameters

- `online_rotation_step_candidates_deg` → Candidate rotation steps (degrees)  
- `online_translation_step_candidates` → Candidate translation steps (meters)

Example:

```bash
ros2 launch calib_eval offline_eval_dynamic.launch.py \
  dataset_root:=/path/to/dynamic_rig \
  split:=train \
  rig_config_id:=dynamic \
  use_dynamic_rig:=true
```

---

### Visualizer

```bash
ros2 launch calib_eval visualizer.launch.py
```

Modes:

* `live`
* `archive_reconstruct`

Output topics:

```
/viz/reference_overlay
/viz/estimated_overlay
```

---

## Checkpoints

Model checkpoints are not included due to size constraints.

They are archived separately:

```
[Zenodo DOI will be added here]
```

The archive contains all checkpoints used in the thesis experiments.

### Restore Structure

```
calib_eval/models/checkpoints/
calib_eval/models/third_party/LCCNet/checkpoints/
calib_eval/models/third_party/BEVCalib/logs/
```

A list of recommended checkpoints will be added separately.

---

## Dataset

Datasets are external.

Structure:

```
dataset_root/
  static_mount_h1/
  static_mount_h2/
  static_mount_h1_h2/
  dynamic_rig/
```

Each run:

```
run_XXX/
  images/
  lidar/
  scans/
  odom/
  intrinsics/
  index/
  splits/
  meta/
```

---

## Results

Evaluation metrics are written to a rolling CSV file.

Workflow:

1. Run evaluation
2. Inspect results
3. Archive CSV

---

## Limitations

* No true ground-truth extrinsics
* Evaluation is reference-relative
* Limited model coverage (LCCNet, BEVCalib)
* Reprojection metrics are unreliable without GT

---

## Thesis Context

This repository represents the final implementation used for:

* calibration method evaluation
* comparative experiments (h1, h2, h1+h2, dynamic)
* analysis of supervised vs BEV-based approaches

---

