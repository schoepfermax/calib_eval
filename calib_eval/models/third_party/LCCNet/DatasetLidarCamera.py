# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/DatasetVisibilityKitti.py

import csv
import os
from pathlib import Path
from math import radians
import cv2
import yaml

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, quaternion_from_matrix, read_calib_file
from pykitti import odometry
import pykitti



def _normalize_stem_token(token):
    token = str(token).strip()
    if not token:
        return token
    token = os.path.basename(token)
    token = os.path.splitext(token)[0]
    return token


def _load_camera_intrinsics_matrix(intrinsics_path):
    with open(intrinsics_path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid intrinsics yaml: {intrinsics_path}")

    if 'camera' in data and isinstance(data['camera'], dict):
        camera = data['camera']
        fx = float(camera['fx'])
        fy = float(camera['fy'])
        cx = float(camera['cx'])
        cy = float(camera['cy'])
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    if 'camera_matrix' in data:
        camera_matrix = data['camera_matrix']
        if isinstance(camera_matrix, dict) and 'data' in camera_matrix:
            values = camera_matrix['data']
            return np.array(values, dtype=np.float32).reshape(3, 3)

    if 'K' in data:
        return np.array(data['K'], dtype=np.float32).reshape(3, 3)

    if 'k' in data:
        return np.array(data['k'], dtype=np.float32).reshape(3, 3)

    raise KeyError(f"Could not find intrinsics matrix in {intrinsics_path}")


def _quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rot = np.array([
        [ww + xx - yy - zz, 2.0 * (xy - zw),     2.0 * (xz + yw)],
        [2.0 * (xy + zw),   ww - xx + yy - zz,   2.0 * (yz - xw)],
        [2.0 * (xz - yw),   2.0 * (yz + xw),     ww - xx - yy + zz],
    ], dtype=np.float32)
    return rot


def _load_reference_extrinsic_matrix(reference_yaml_path):
    with open(reference_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    translation = np.array(data['translation'], dtype=np.float32)
    rotation_quat = np.array(data['rotation_quat'], dtype=np.float32)
    rotation_matrix = _quat_xyzw_to_rotmat(rotation_quat)

    rt = np.eye(4, dtype=np.float32)
    rt[:3, :3] = rotation_matrix
    rt[:3, 3] = translation
    return rt


def _read_ascii_pcd_xyz1(pcd_path):
    points = []
    with open(pcd_path, 'r') as f:
        header_done = False
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not header_done:
                if line.upper().startswith('DATA'):
                    if 'ascii' not in line.lower():
                        raise ValueError(f"Only ASCII PCD is supported: {pcd_path}")
                    header_done = True
                continue

            values = line.split()
            if len(values) < 3:
                continue
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            points.append([x, y, z, 1.0])

    if not points:
        raise ValueError(f"No points found in PCD file: {pcd_path}")

    return np.array(points, dtype=np.float32)


class DatasetLidarCameraThesisStatic(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png',
                 reference_path=None, split_file=None, max_samples=None, val_perturb_file=None):
        super(DatasetLidarCameraThesisStatic, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.reference_path = reference_path
        self.split_file = split_file
        self.max_samples = max_samples
        self.val_perturb_file = val_perturb_file
        self.suf = suf

        if self.reference_path is None:
            raise ValueError("reference_path must be provided for DatasetLidarCameraThesisStatic")

        # Training/evaluation can operate on a single static rig root (h1 or h2)
        # or on a merged h1+h2 root built from symlinked run directories.
        # In the merged case, one global reference extrinsic is not sufficient,
        # because h1 and h2 are different rig configurations. Therefore we keep
        # a default reference RT from the provided reference_path, and also try
        # to load both static rig references from the same reference directory.
        # At sample time we select the correct RT based on the real run path.
        self.reference_RT_default = _load_reference_extrinsic_matrix(self.reference_path)

        reference_dir = Path(self.reference_path).resolve().parent
        self.reference_path_h1 = reference_dir / 'static_mount_h1_reference.yaml'
        self.reference_path_h2 = reference_dir / 'static_mount_h2_reference.yaml'

        self.reference_RT_h1 = None
        self.reference_RT_h2 = None

        if self.reference_path_h1.exists():
            self.reference_RT_h1 = _load_reference_extrinsic_matrix(str(self.reference_path_h1))
        if self.reference_path_h2.exists():
            self.reference_RT_h2 = _load_reference_extrinsic_matrix(str(self.reference_path_h2))

        self.all_files = []
        self.K = {}

        run_dirs = [p for p in Path(self.root_dir).glob('run_*') if p.is_dir()]
        run_dirs = sorted(run_dirs)

        for run_dir in run_dirs:
            intrinsics_path = run_dir / 'intrinsics' / 'camera_intrinsics.yaml'
            if not intrinsics_path.exists():
                continue

            split_path = None
            if self.split_file is not None:
                split_candidate = Path(self.split_file)
                if not split_candidate.is_absolute():
                    split_candidate = run_dir / split_candidate
                split_path = split_candidate
            else:
                split_path = run_dir / 'splits' / f'{self.split}.txt'
                if not split_path.exists():
                    split_path = run_dir / 'index' / f'{self.split}.txt'

            if split_path is None or not split_path.exists():
                continue

            try:
                calib = _load_camera_intrinsics_matrix(str(intrinsics_path))
            except Exception as exc:
                print(f"Skipping {run_dir} because intrinsics could not be loaded: {exc}")
                continue

            self.K[str(run_dir)] = calib

            with open(split_path, 'r') as f:
                stem_tokens = [_normalize_stem_token(line) for line in f.readlines() if line.strip()]

            for stem in stem_tokens:
                img_path = run_dir / 'images' / f'{stem}{self.suf}'
                lidar_path = run_dir / 'lidar' / f'{stem}.pcd'
                if not img_path.exists():
                    alt_img = run_dir / 'images' / f'{stem}.jpg'
                    if alt_img.exists():
                        img_path = alt_img
                    else:
                        continue
                if not lidar_path.exists():
                    continue

                self.all_files.append({
                    'run_dir': str(run_dir),
                    'run_name': run_dir.name,
                    'stem': stem,
                    'img_path': str(img_path),
                    'lidar_path': str(lidar_path),
                    'intrinsics_path': str(intrinsics_path),
                    'calib': calib.copy(),
                })

        if self.max_samples is not None:
            self.all_files = self.all_files[:int(self.max_samples)]

        self.val_RT = []
        if self.split in ('val', 'test'):
            if self.val_perturb_file is None:
                self.val_perturb_file = os.path.join(
                    self.root_dir,
                    f'val_RT_thesis_{self.split}_r{self.max_r:.2f}_t{self.max_t:.2f}.csv'
                )

            if os.path.exists(self.val_perturb_file):
                print(f'VAL SET: Using this file: {self.val_perturb_file}')
                df_test_RT = pd.read_csv(self.val_perturb_file, sep=',')
                for _, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {self.val_perturb_file}')
                print("Generating a new one")
                with open(self.val_perturb_file, 'w', newline='') as csv_f:
                    writer = csv.writer(csv_f, delimiter=',')
                    writer.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                    for i in range(len(self.all_files)):
                        rng = np.random.RandomState(i + 1337)
                        rotz = rng.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                        roty = rng.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                        rotx = rng.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                        transl_x = rng.uniform(-self.max_t, self.max_t)
                        transl_y = rng.uniform(-self.max_t, self.max_t)
                        transl_z = rng.uniform(-self.max_t, self.max_t)
                        writer.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                        self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                            float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with validation perturbations"

    def _select_reference_rt(self, run_dir):
        """
        Select the correct static-rig reference extrinsic for a sample.

        Important:
          - For merged h1+h2 datasets we create symlinked run directories.
          - Therefore we resolve the real run path before checking rig identity.
          - If neither h1 nor h2 can be identified, we fall back to the
            user-provided reference_path loaded at init.
        """
        resolved = str(Path(run_dir).resolve())

        if 'static_mount_h1' in resolved and self.reference_RT_h1 is not None:
            return self.reference_RT_h1.astype(np.float32)

        if 'static_mount_h2' in resolved and self.reference_RT_h2 is not None:
            return self.reference_RT_h2.astype(np.float32)

        return self.reference_RT_default.astype(np.float32)

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        img_path = item['img_path']
        lidar_path = item['lidar_path']
        calib = item['calib'].copy()

        pc = _read_ascii_pcd_xyz1(lidar_path)
        valid_indices = (pc[:, 0] < -3.) | (pc[:, 0] > 3.) | (pc[:, 1] < -3.) | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()

        pc_org = torch.from_numpy(pc.astype(np.float32))
        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        E_RT = self._select_reference_rt(item['run_dir'])
        pc_rot = np.matmul(E_RT, pc_org.numpy()).astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        img = Image.open(img_path).convert('RGB')
        img_rotation = 0.

        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        if self.split == 'train':
            R_aug = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T_aug = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R_aug, T_aug)

        if self.split == 'train':
            rotz = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            roty = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            rotx = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {
            'rgb': img,
            'point_cloud': pc_in,
            'calib': torch.from_numpy(calib.astype(np.float32)),
            'pc_org': pc_org,
            'img_path': img_path,
            'tr_error': T,
            'rot_error': R,
            'rgb_name': os.path.basename(img_path),
            'item': f"{item['run_name']}/{item['stem']}",
            'extrin': E_RT,
            'initial_RT': initial_RT,
            'pc_lidar': pc.copy(),
        }
        return sample



class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf

        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        # for seq in ['00', '03', '05', '06', '07', '08', '09']:
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2 # 3x3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]
            # self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            # val_RT_file = os.path.join(dataset_dir, 'sequences',
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name+self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        pc_rot = np.matmul(RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        # pc_rot = np.matmul(RT, pc.T)
        # pc_rot = pc_rot.astype(np.float).T.copy()
        # pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()

        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #      if not torch.all(pc_in[3,:] == 1.):
        #         pc_in[3,:] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[1, :] *= -1

        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT}

        return sample


class DatasetLidarCameraKittiRaw(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_26_drive_0117_sync'):
        super(DatasetLidarCameraKittiRaw, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}

        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}

        for i in range(len(date_list)):
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive)
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib

        # date = val_sequence[:10]
        # seq = val_sequence
        # image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
        # image_list.sort()
        #
        # for image_name in image_list:
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
        #                                        str(image_name.split('.')[0]) + '.bin')):
        #         continue
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
        #                                        str(image_name.split('.')[0]) + '.jpg')):  # png
        #         continue
        #     self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        date = val_sequence[:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))

        for seq in seq_list:
            if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
                continue
            image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                                                   str(image_name.split('.')[0])+'.jpg')): # png
                    continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq not in test_list:
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    # self.all_files.append(os.path.join(date, seq, 'image_2/data', image_name.split('.')[0]))
    def __getitem__(self, idx):
        item = self.all_files[idx]
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.jpg') # png
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_lidar = pc.copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        # camera intrinsic parameter
        calib_cam02 = calib['K2']  # 3x3

        E_RT = RT_cam02

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img = Image.open(img_path)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = calib_cam02
        # calib = get_calib_kitti_odom(int(seq))
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT}
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name + '.png', 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}

        return sample

