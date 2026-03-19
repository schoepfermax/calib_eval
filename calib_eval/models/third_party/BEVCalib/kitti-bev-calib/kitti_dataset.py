from pykitti import odometry
import numpy as np
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset
import os

class KittiDataset(Dataset):
    def __init__(self, data_folder='./data/kitti-odemetry', suf='.png'):
        self.all_files = []
        self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                          '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.dataset_root = data_folder
        self.K = {}
        self.T = {}

        for seq in self.sequences:
            odom = odometry(data_folder, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo
            T_velo2_cam0_np = np.linalg.inv(T_cam02_velo_np)
            self.K[seq] = calib.K_cam2
            self.T[seq] = T_velo2_cam0_np
            image_list = os.listdir(os.path.join(self.dataset_root, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(self.dataset_root, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(self.dataset_root, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue

                self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        seq = self.all_files[idx].split('/')[0]
        id = self.all_files[idx].split('/')[1]
        img_path = os.path.join(self.dataset_root, 'sequences', seq, 'image_2', id+'.png')
        pcd_path = os.path.join(self.dataset_root, 'sequences', seq, 'velodyne', id+'.bin')
        if not os.path.exists(img_path) or not os.path.exists(pcd_path):
            print('File not exist')
            assert False
        img = Image.open(img_path)
        img.resize((1242, 375))
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        valid_ind = pcd[:, 0] < -3.
        valid_ind = valid_ind | (pcd[:, 0] > 3.)
        valid_ind = valid_ind | (pcd[:, 1] < -3.)
        valid_ind = valid_ind | (pcd[:, 1] > 3.)
        pcd = pcd[valid_ind, :]
        # print(f"x_min: {pcd[:, 0].min()}, x_max: {pcd[:, 0].max()}, y_min: {pcd[:, 1].min()}, y_max: {pcd[:, 1].max()}")
        gt_transform = self.T[seq]
        intrinsic = self.K[seq]
        return img, pcd, gt_transform, intrinsic
        
if __name__ == "__main__":
    dataset_root = './data/kitti-odemetry'
    train_dataset = KittiDataset(data_folder=dataset_root, split='train')
    print(len(train_dataset))
    val_dataset = KittiDataset(data_folder=dataset_root, split='val')
    print(len(val_dataset))
    print(train_dataset.all_files[-2])
    all_size = []
    for i in range(len(val_dataset)):
        img = val_dataset[i][0]
        # print(f"img size: {img.size}")
        if img.size not in all_size:
            all_size.append(img.size)
    print(all_size)
    # for i in range(len(train_dataset)):
    #     train_dataset[i]

    