###############################################
# BEV TRAINING ENTRY FOR THESIS PROJECT
###############################################

import os
import sys
import cv2
import yaml
import shutil
import argparse
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


def _current_file_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _project_root() -> str:
    return os.path.dirname(_current_file_dir())


def _third_party_bev_root() -> str:
    return os.path.join(
        _current_file_dir(),
        "models",
        "third_party",
        "BEVCalib",
        "kitti-bev-calib",
    )


def _configure_bev_python_paths():
    bev_root = _third_party_bev_root()
    bev_img_branch = os.path.join(bev_root, "img_branch")

    for p in [bev_root, bev_img_branch]:
        if p not in sys.path:
            sys.path.insert(0, p)


_configure_bev_python_paths()

from bev_calib import BEVCalib
from tools import generate_single_perturbation_from_T

from calib_eval.thesis_bev_dataset import ThesisBEVDataset
from calib_eval.models.registry import set_global_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="Train BEVCalib on thesis dataset")

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--combined_index", type=int, default=0)

    parser.add_argument("--static_mount_h1_reference_yaml", type=str, default="")
    parser.add_argument("--static_mount_h2_reference_yaml", type=str, default="")
    parser.add_argument("--dynamic_reference_yaml", type=str, default="")

    parser.add_argument("--log_dir", type=str, default="./logs/thesis_bev")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--copy_source_snapshot", type=int, default=1)

    parser.add_argument("--save_ckpt_per_epoches", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_epoches", type=int, default=1)

    parser.add_argument("--angle_range_deg", type=float, default=20.0)
    parser.add_argument("--trans_range", type=float, default=1.5)
    parser.add_argument("--eval_angle_range_deg", type=float, default=None)
    parser.add_argument("--eval_trans_range", type=float, default=None)

    parser.add_argument("--deformable", type=int, default=0)
    parser.add_argument("--bev_encoder", type=int, default=1)
    parser.add_argument("--xyz_only", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--scheduler", type=int, default=-1)

    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_determinism", type=int, default=0)

    parser.add_argument("--hf_cache", type=str, default="")

    return parser.parse_args()


def maybe_set_hf_cache(hf_cache: str):
    cache = str(hf_cache).strip()
    if not cache:
        return
    os.environ["HF_HOME"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache


def crop_and_resize_bgr(image_bgr, size, intrinsics, crop=False):
    """
    Upstream train_kitti.py assumes RGB/PIL input and converts to BGR.
    Our loader already returns BGR (OpenCV convention), so we keep the same
    intrinsics-resize logic but skip the color conversion.
    """
    img = np.asarray(image_bgr, dtype=np.uint8)
    h, w = img.shape[:2]

    if crop:
        mid_width = w // 2
        start_x = (w - mid_width) // 2
        cropped = img[:, start_x:start_x + mid_width]
        resized = cv2.resize(cropped, size)
    else:
        resized = cv2.resize(img, size)

    if crop:
        new_cx = float(intrinsics[0, 2]) - float(start_x)
        scale_x = float(size[0]) / float(mid_width)
    else:
        new_cx = float(intrinsics[0, 2])
        scale_x = float(size[0]) / float(w)

    scale_y = float(size[1]) / float(h)

    new_intrinsics = np.array(
        [
            [float(intrinsics[0, 0]) * scale_x, 0.0, new_cx * scale_x],
            [0.0, float(intrinsics[1, 1]) * scale_y, float(intrinsics[1, 2]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return resized.astype(np.uint8), new_intrinsics.astype(np.float32)


def collate_fn(batch):
    target_size = (704, 256)

    processed_data = [
        crop_and_resize_bgr(item[0], target_size, item[3], crop=False)
        for item in batch
    ]

    imgs = [item[0] for item in processed_data]
    intrinsics = [item[1] for item in processed_data]

    gt_T_to_camera = [item[2] for item in batch]

    pcs = []
    masks = []
    max_num_points = 0

    for item in batch:
        max_num_points = max(max_num_points, int(item[1].shape[0]))

    for item in batch:
        pc = np.asarray(item[1], dtype=np.float32)
        valid_count = int(pc.shape[0])

        masks.append(
            np.concatenate(
                [
                    np.ones(valid_count, dtype=np.float32),
                    np.zeros(max_num_points - valid_count, dtype=np.float32),
                ],
                axis=0,
            )
        )

        if valid_count < max_num_points:
            pad = np.full(
                (max_num_points - valid_count, pc.shape[1]),
                999999.0,
                dtype=np.float32,
            )
            pc = np.concatenate([pc, pad], axis=0)

        pcs.append(pc.astype(np.float32))

    return imgs, pcs, masks, gt_T_to_camera, intrinsics


def _load_pretrain_weights(model, ckpt_path: str, device):
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)


def _save_yaml(path: str, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _prepare_log_dir(base_log_dir: str, label: str = None) -> str:
    if label is not None and str(label).strip():
        out = os.path.join(base_log_dir, str(label).strip())
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = os.path.join(base_log_dir, current_time)

    os.makedirs(out, exist_ok=True)
    return out


def _copy_training_snapshot(log_dir: str):
    """
    Save the exact training-side code used for reproducibility.
    """
    try:
        shutil.copy2(__file__, os.path.join(log_dir, "train_bev.py"))
    except Exception as e:
        print(f"WARNING: Could not snapshot train_bev.py: {e}")

    try:
        src_dataset = os.path.join(_current_file_dir(), "thesis_bev_dataset.py")
        shutil.copy2(src_dataset, os.path.join(log_dir, "thesis_bev_dataset.py"))
    except Exception as e:
        print(f"WARNING: Could not snapshot thesis_bev_dataset.py: {e}")

    try:
        bev_root = _third_party_bev_root()
        dst = os.path.join(log_dir, "kitti-bev-calib")
        if os.path.isdir(bev_root) and not os.path.exists(dst):
            shutil.copytree(bev_root, dst)
    except Exception as e:
        print(f"WARNING: Could not snapshot BEV third-party source tree: {e}")


def main():
    args = parse_args()
    print(args)

    maybe_set_hf_cache(args.hf_cache)

    set_global_determinism(
        seed=int(args.seed),
        force_determinism=bool(args.force_determinism),
        logger=None,
    )

    if bool(args.force_determinism) and torch.cuda.is_available():
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            print(
                "WARNING: force_determinism=1 but CUBLAS_WORKSPACE_CONFIG is not set.\n"
                "For CUDA deterministic behavior, export for example:\n"
                "  CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )

    log_dir = _prepare_log_dir(args.log_dir, args.label)
    model_save_dir = os.path.join(log_dir, "model")
    ckpt_save_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    if int(args.copy_source_snapshot) > 0:
        _copy_training_snapshot(log_dir)

    _save_yaml(
        os.path.join(log_dir, "train_args.yaml"),
        vars(args),
    )

    writer = SummaryWriter(log_dir)

    train_dataset = ThesisBEVDataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        combined_index=bool(args.combined_index),
        static_mount_h1_reference_yaml=args.static_mount_h1_reference_yaml,
        static_mount_h2_reference_yaml=args.static_mount_h2_reference_yaml,
        dynamic_reference_yaml=args.dynamic_reference_yaml,
    )

    val_dataset = ThesisBEVDataset(
        dataset_root=args.dataset_root,
        split=args.val_split,
        combined_index=bool(args.combined_index),
        static_mount_h1_reference_yaml=args.static_mount_h1_reference_yaml,
        static_mount_h2_reference_yaml=args.static_mount_h2_reference_yaml,
        dynamic_reference_yaml=args.dynamic_reference_yaml,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    deformable_choice = int(args.deformable) > 0
    bev_encoder_choice = int(args.bev_encoder) > 0
    xyz_only_choice = int(args.xyz_only) > 0

    requested_device = str(args.device).strip().lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = BEVCalib(
        deformable=deformable_choice,
        bev_encoder=bev_encoder_choice,
    ).to(device)

    if args.pretrain_ckpt is not None and str(args.pretrain_ckpt).strip():
        _load_pretrain_weights(model, str(args.pretrain_ckpt).strip(), device)
        print(f"Loaded pretrain model from {args.pretrain_ckpt}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    scheduler_choice = int(args.scheduler) > 0
    if scheduler_choice:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    train_noise = {
        "angle_range_deg": float(args.angle_range_deg),
        "trans_range": float(args.trans_range),
    }

    eval_noise = {
        "angle_range_deg": (
            float(args.eval_angle_range_deg)
            if args.eval_angle_range_deg is not None
            else train_noise["angle_range_deg"]
        ),
        "trans_range": (
            float(args.eval_trans_range)
            if args.eval_trans_range is not None
            else train_noise["trans_range"]
        ),
    }

    for epoch in range(int(args.num_epochs)):
        model.train()
        train_loss = {}
        out_init_loss_choice = epoch < 5

        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(train_loader):
            gt_T_to_camera = np.asarray(gt_T_to_camera, dtype=np.float32)

            init_T_to_camera, _, _ = generate_single_perturbation_from_T(
                gt_T_to_camera,
                angle_range_deg=train_noise["angle_range_deg"],
                trans_range=train_noise["trans_range"],
            )

            resize_imgs = torch.from_numpy(np.asarray(imgs)).permute(0, 3, 1, 2).float().to(device)

            pcs_np = np.asarray(pcs, dtype=np.float32)
            if xyz_only_choice:
                pcs_np = pcs_np[:, :, :3]
            pcs_t = torch.from_numpy(pcs_np).float().to(device)

            gt_T_to_camera_t = torch.from_numpy(gt_T_to_camera).float().to(device)
            init_T_to_camera_t = torch.from_numpy(init_T_to_camera).float().to(device)

            post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera_t.shape[0], 1, 1).float().to(device)
            intrinsic_matrix = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).float().to(device)

            optimizer.zero_grad()

            _, init_loss, loss = model(
                resize_imgs,
                pcs_t,
                gt_T_to_camera_t,
                init_T_to_camera_t,
                post_cam2ego_T,
                intrinsic_matrix,
                masks=masks,
                out_init_loss=out_init_loss_choice,
            )

            total_loss = loss["total_loss"]
            total_loss.backward()
            optimizer.step()

            for key in loss.keys():
                train_loss[key] = train_loss.get(key, 0.0) + float(loss[key].item())

            if init_loss is not None:
                for key in init_loss.keys():
                    train_key = f"init_{key}"
                    train_loss[train_key] = train_loss.get(train_key, 0.0) + float(init_loss[key].item())

            if batch_index % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
                    f"Step [{batch_index + 1}/{len(train_loader)}], "
                    f"Loss: {float(total_loss.item()):.4f}"
                )

        if scheduler_choice:
            scheduler.step()

        for key in train_loss.keys():
            train_loss[key] /= max(1, len(train_loader))
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Train Loss {key}: {train_loss[key]:.4f}")
            writer.add_scalar(f"Loss/train/{key}", train_loss[key], epoch)

        if epoch == int(args.num_epochs) - 1 or (
            int(args.save_ckpt_per_epoches) > 0
            and (epoch + 1) % int(args.save_ckpt_per_epoches) == 0
        ):
            ckpt_path = os.path.join(ckpt_save_dir, f"ckpt_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_noise": train_noise,
                    "eval_noise": eval_noise,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

        if epoch % int(args.eval_epoches) == 0:
            model.eval()
            val_loss = {}

            with torch.no_grad():
                for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
                    gt_T_to_camera = np.asarray(gt_T_to_camera, dtype=np.float32)

                    init_T_to_camera, _, _ = generate_single_perturbation_from_T(
                        gt_T_to_camera,
                        angle_range_deg=eval_noise["angle_range_deg"],
                        trans_range=eval_noise["trans_range"],
                    )

                    resize_imgs = torch.from_numpy(np.asarray(imgs)).permute(0, 3, 1, 2).float().to(device)

                    pcs_np = np.asarray(pcs, dtype=np.float32)
                    if xyz_only_choice:
                        pcs_np = pcs_np[:, :, :3]
                    pcs_t = torch.from_numpy(pcs_np).float().to(device)

                    gt_T_to_camera_t = torch.from_numpy(gt_T_to_camera).float().to(device)
                    init_T_to_camera_t = torch.from_numpy(init_T_to_camera).float().to(device)

                    post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera_t.shape[0], 1, 1).float().to(device)
                    intrinsic_matrix = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).float().to(device)

                    _, init_loss, loss = model(
                        resize_imgs,
                        pcs_t,
                        gt_T_to_camera_t,
                        init_T_to_camera_t,
                        post_cam2ego_T,
                        intrinsic_matrix,
                        masks=masks,
                        out_init_loss=False,
                    )

                    for key in loss.keys():
                        val_loss[key] = val_loss.get(key, 0.0) + float(loss[key].item())

                    if init_loss is not None:
                        for key in init_loss.keys():
                            val_key = f"init_{key}"
                            val_loss[val_key] = val_loss.get(val_key, 0.0) + float(init_loss[key].item())

            for key in val_loss.keys():
                val_loss[key] /= max(1, len(val_loader))
                print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
                    f"{eval_noise['angle_range_deg']}_{eval_noise['trans_range']} "
                    f"Validation Loss {key}: {val_loss[key]:.4f}"
                )
                writer.add_scalar(f"Loss/val/{key}", val_loss[key], epoch)

    writer.close()
    print(f"Logs are saved at {log_dir}")


if __name__ == "__main__":
    main()