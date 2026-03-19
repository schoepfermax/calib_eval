###############################################
# Deterministic model registry (workspace-safe)
# Supports:
#   - BEVCalib
#   - LCCNet
###############################################

import os
import sys
import random
import numpy as np


def _models_root():
    """
    Resolve the package-local models folder directly from this file location.

    This avoids assuming a specific workspace name such as 'calib_ws'.
    """
    return os.path.dirname(os.path.abspath(__file__))


def _checkpoint_path():
    """
    Default BEVCalib checkpoint path inside this package.
    """
    models_root = _models_root()
    return os.path.join(models_root, "checkpoints", "BEVCalib", "ckpt.pth")


def _third_party_bevcalib_path():
    """
    BEVCalib Python package root containing bev_calib.py.
    """
    models_root = _models_root()
    return os.path.join(models_root, "third_party", "BEVCalib", "kitti-bev-calib")


def _third_party_lccnet_path():
    models_root = _models_root()
    return os.path.join(models_root, "third_party", "LCCNet")


def _lcc_checkpoint_default():
    models_root = _models_root()
    return os.path.join(models_root, "checkpoints", "LCCNet", "kitti_iter1.tar")


def set_global_determinism(seed=42, force_determinism=True, logger=None):
    """
    Keep runtime behavior deterministic where practical.

    This is intentionally lightweight and safe to call from ROS2 nodes.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if force_determinism:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            try:
                import torch.backends.cudnn as cudnn
                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass

        if logger is not None:
            logger.info(
                f"Determinism configured: seed={seed}, force_determinism={force_determinism}"
            )

    except Exception as e:
        if logger is not None:
            logger.warn(f"Could not fully configure torch determinism: {e}")


class _BEVWrapper:
    """
    Thin wrapper to normalize BEVCalib I/O to a stable interface for the ROS2 pipeline.

    The BEV node performs resize / intrinsics scaling itself.
    This wrapper constructs the remaining tensors required by the third-party
    model forward() signature and returns a full 4x4 transform estimate.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_T(self, img_bchw, pc_bnx3, K_bx3x3, init_T_bx4x4):
        import torch

        if init_T_bx4x4 is None:
            B = int(img_bchw.shape[0])
            init_T_bx4x4 = torch.eye(
                4, dtype=img_bchw.dtype, device=img_bchw.device
            ).unsqueeze(0).repeat(B, 1, 1)

        # During inference we do not have GT extrinsics.
        # BEVCalib's forward() still expects gt_T_to_camera because the training
        # code computes a loss internally. Passing identity is sufficient here
        # because we only use the returned predicted transform.
        B = int(img_bchw.shape[0])
        gt_T_to_camera = torch.eye(
            4, dtype=img_bchw.dtype, device=img_bchw.device
        ).unsqueeze(0).repeat(B, 1, 1)

        post_cam2ego_T = torch.eye(
            4, dtype=img_bchw.dtype, device=img_bchw.device
        ).unsqueeze(0).repeat(B, 1, 1)

        with torch.no_grad():
            pred_T, _, _ = self.model(
                img=img_bchw,
                pc=pc_bnx3,
                gt_T_to_camera=gt_T_to_camera,
                init_T_to_camera=init_T_bx4x4,
                post_cam2ego_T=post_cam2ego_T,
                cam_intrinsic=K_bx3x3,
                masks=None,
                out_init_loss=False,
            )

        return pred_T


class _LCCNetWrapper:
    """
    Minimal wrapper to keep parity with the BEV wrapper style.

    LCCNet forward returns:
        translation: (B,3)
        rotation:    (B,4) quaternion

    IMPORTANT (quaternion convention):
      The upstream LCCNet codebase (and forks) have been observed with both:
        - xyzw  (x,y,z,w)
        - wxyz  (w,x,y,z)

      Our downstream ROS2 pipeline expects xyzw when converting to/from geometry transforms.
      To avoid systematic 180° flips caused by convention mismatch, we:

        1) auto-detect the convention using a documented magnitude heuristic
        2) convert to xyzw if needed
        3) explicitly normalize the quaternion
    """

    def __init__(self, model, device, logger=None):
        self.model = model
        self.device = device
        self.logger = logger
        self._warned_quat_order = False
        self._logged_xyzw = False

    def predict_delta(self, rgb_bchw, depth_b1hw):
        import torch

        with torch.no_grad():
            transl, rot = self.model(rgb_bchw, depth_b1hw)

        rot_xyzw = rot

        if rot is not None and hasattr(rot, "shape") and rot.shape[-1] == 4:
            w_last_mean = torch.mean(torch.abs(rot[..., 3])).item()
            w_first_mean = torch.mean(torch.abs(rot[..., 0])).item()

            if (w_last_mean < 0.50) and (w_first_mean > 0.50):
                rot_xyzw = torch.stack(
                    [rot[..., 1], rot[..., 2], rot[..., 3], rot[..., 0]],
                    dim=-1,
                )

                if (self.logger is not None) and (not self._warned_quat_order):
                    self.logger.warn(
                        "LCCNet quaternion appears to be wxyz; converting to xyzw "
                        f"(mean|w_last|={w_last_mean:.3f}, mean|w_first|={w_first_mean:.3f})."
                    )
                    self._warned_quat_order = True
            else:
                if (self.logger is not None) and (not self._logged_xyzw):
                    self.logger.info(
                        "LCCNet quaternion appears to be xyzw "
                        f"(mean|w_last|={w_last_mean:.3f}, mean|w_first|={w_first_mean:.3f})."
                    )
                    self._logged_xyzw = True

            eps = 1e-12
            rot_norm = torch.linalg.norm(rot_xyzw, dim=-1, keepdim=True).clamp_min(eps)
            rot_xyzw = rot_xyzw / rot_norm

        return transl, rot_xyzw


def create_model_from_registry(model_name, checkpoint_path="", device="cpu", logger=None):
    name = (model_name or "").lower().strip()

    import torch

    torch_device = torch.device(device if device in ["cpu", "cuda"] else "cpu")

    # ------------------------------------------------------
    # BEVCalib
    # ------------------------------------------------------
    if name in ["bevcalib", "bev_calib", "bev"]:
        bev_repo_path = _third_party_bevcalib_path()
        if not os.path.isdir(bev_repo_path):
            raise RuntimeError(f"BEVCalib repo not found at: {bev_repo_path}")

        if bev_repo_path not in sys.path:
            sys.path.insert(0, bev_repo_path)

        try:
            from bev_calib import BEVCalib  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import BEVCalib from {bev_repo_path}: {e}")

        ckpt = checkpoint_path.strip() if checkpoint_path else _checkpoint_path()

        if not os.path.isfile(ckpt):
            raise RuntimeError(f"Checkpoint not found at: {ckpt}")

        model = BEVCalib(
            deformable=False,
            bev_encoder=True,
        ).to(torch_device)

        state = torch.load(ckpt, map_location=torch_device)

        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        model.eval()

        if logger:
            logger.info(f"Loaded BEVCalib from {ckpt} on device {torch_device}")

        return _BEVWrapper(model=model, device=str(torch_device))

    # ------------------------------------------------------
    # LCCNet
    # ------------------------------------------------------
    if name in ["lccnet", "lcc"]:
        lcc_repo_path = _third_party_lccnet_path()
        if not os.path.isdir(lcc_repo_path):
            raise RuntimeError(f"LCCNet repo not found at: {lcc_repo_path}")

        if lcc_repo_path not in sys.path:
            sys.path.insert(0, lcc_repo_path)

        try:
            from models.LCCNet import LCCNet  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import LCCNet: {e}")

        ckpt = checkpoint_path.strip() if checkpoint_path else _lcc_checkpoint_default()
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"LCCNet checkpoint not found at: {ckpt}")

        # Derived from fc1.weight shape [512, 67712] => expected image_size (H,W)=(256,512)
        image_size = (256, 512)

        model = LCCNet(
            image_size=image_size,
            use_feat_from=1,
        ).to(torch_device)

        state = torch.load(ckpt, map_location=torch_device)

        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

        model.eval()

        if logger:
            logger.info(f"Loaded LCCNet from {ckpt} on device {torch_device}")

        return _LCCNetWrapper(model=model, device=str(torch_device), logger=logger)

    raise RuntimeError(
        f"Unknown model_name '{model_name}'. Supported: BEVCalib, LCCNet."
    )