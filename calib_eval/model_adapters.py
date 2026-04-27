import numpy as np
import cv2

from calib_eval.geometry_utils import (
    laserscan_to_depth_like_image,
)


class LCCNetAdapter:
    """
    Adapter for LCCNet-style models.

    Converts sparse LiDAR points into a dense depth image
    aligned with the RGB camera frame.

    NOTE (dynamic rig):
      - Dynamic rig provides a 2D LaserScan.
      - We cannot create a true camera-aligned depth map without extrinsics.
      - Therefore we generate a stable "depth-like scan image" from the scan ranges
        (see laserscan_to_depth_like_image in geometry_utils).
    """

    @staticmethod
    def adapt(sample):
        image = sample['image']
        intr = sample['intrinsics']

        # ------------------------------------------------------------
        # Preferred dynamic-rig path:
        # If scan exists, provide a depth-like scan image.
        # ------------------------------------------------------------
        if sample.get('scan', None) is not None:
            scan = sample['scan']

            # Depth-like image is NOT camera-aligned depth; it's a scan-image tensor.
            # Keep it simple and deterministic.
            depth = laserscan_to_depth_like_image(
                ranges=scan.get('ranges', []),
                out_width=None,
                out_height=64,
                range_min=float(scan.get('range_min', 0.0)) if 'range_min' in scan else 0.0,
                range_max=float(scan.get('range_max', 50.0)) if 'range_max' in scan else 50.0,
            )

            return {
                'rgb': image,
                'depth': depth.astype(np.float32),
                'gt_extrinsics': (
                    sample.get('gt_translation', None),
                    sample.get('gt_rotation_quat', None),
                )
            }

        # ------------------------------------------------------------
        # Static-rig (or pointcloud-provided) path:
        # project points directly to depth image.
        # WARNING:
        #   This assumes points are already in the camera frame (x,y,z).
        #   If points are in LiDAR frame, this must be replaced with
        #   GT/estimated extrinsics transform before projection.
        # ------------------------------------------------------------
        points = sample['points']

        depth = np.zeros(image.shape[:2], dtype=np.float32)

        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']

        for x, y, z in points:
            if z <= 0:
                continue

            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)

            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                depth[v, u] = z

        return {
            'rgb': image,
            'depth': depth,
            'gt_extrinsics': (
                sample.get('gt_translation', None),
                sample.get('gt_rotation_quat', None),
            )
        }


class BEVCalibAdapter:
    """
    Adapter for BEV-based calibration models.

    Converts LiDAR points into a bird’s-eye-view occupancy grid.
    """

    @staticmethod
    def adapt(sample, grid_size=0.1, grid_range=20.0):
        points = sample['points']

        grid_dim = int((2 * grid_range) / grid_size)
        bev = np.zeros((grid_dim, grid_dim), dtype=np.uint8)

        for x, y, _ in points:
            ix = int((x + grid_range) / grid_size)
            iy = int((y + grid_range) / grid_size)

            if 0 <= ix < grid_dim and 0 <= iy < grid_dim:
                bev[iy, ix] = 255

        return {
            'bev': bev,
            'rgb': sample['image'],
            'gt_extrinsics': (
                sample.get('gt_translation', None),
                sample.get('gt_rotation_quat', None),
            )
        }


class CalibNetAdapter(LCCNetAdapter):
    """
    CalibNet uses the same RGB + depth formulation as LCCNet.

    Inheriting avoids duplication while keeping intent explicit.
    """
    pass


    @staticmethod
    def adapt(sample):
        return BEVCalibAdapter.adapt(sample)