###############################################
# KITTI INGESTION
###############################################

import os
from typing import Dict, List


def discover_kitti_odometry_sequences(kitti_root: str) -> List[str]:
    """
    Returns a sorted list of KITTI odometry sequence IDs found under kitti_root.

    Expected common structure (varies by download):
      kitti_root/
        dataset/
          sequences/
            00/
              image_2/
              velodyne/
              calib.txt
            01/
            ...
    """
    kitti_root = os.path.expanduser(kitti_root)

    # Common KITTI odometry layout
    seq_root = os.path.join(kitti_root, "dataset", "sequences")
    if not os.path.isdir(seq_root):
        # Fallback
        seq_root = os.path.join(kitti_root, "sequences")

    if not os.path.isdir(seq_root):
        return []

    seqs = []
    for name in os.listdir(seq_root):
        full = os.path.join(seq_root, name)
        if os.path.isdir(full) and name.isdigit():
            seqs.append(name)

    seqs.sort(key=lambda s: int(s))
    return seqs


def kitti_sequence_paths(kitti_root: str, seq_id: str) -> Dict[str, str]:
    """
    Returns important paths for a given KITTI sequence ID.

    Keys:
      - sequence_dir
      - image_dir
      - velodyne_dir
      - calib_path
    """
    kitti_root = os.path.expanduser(kitti_root)

    seq_root = os.path.join(kitti_root, "dataset", "sequences")
    if not os.path.isdir(seq_root):
        seq_root = os.path.join(kitti_root, "sequences")

    sequence_dir = os.path.join(seq_root, seq_id)

    return {
        "sequence_dir": sequence_dir,
        "image_dir": os.path.join(sequence_dir, "image_2"),
        "velodyne_dir": os.path.join(sequence_dir, "velodyne"),
        "calib_path": os.path.join(sequence_dir, "calib.txt"),
    }
