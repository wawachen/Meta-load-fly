import torch
import numpy as np
from torch.utils.data import Dataset
import os, os.path
import scipy
import scipy.io

class DataModel_2d(Dataset):
    def __init__(self, flag, path):
        super().__init__()

        self.flag = flag
        self.pointcloud_folder = path
        self.all_filelist = self.get_all_filelist(self.pointcloud_folder)

    def get_all_filelist(self, file_path):
        if self.flag == "train":
            filelist = []
            for i in range(len(file_path)):
                DIR = file_path[i]
                for name in os.listdir(DIR):
                    filelist.append(os.path.join(DIR, name))

        if self.flag == "validation":
            filelist = []
            for i in range(len(file_path)):
                DIR = file_path[i]
                for name in os.listdir(DIR):
                    filelist.append(os.path.join(DIR, name))

        if self.flag == "test":
            filelist = []
            for i in range(len(file_path)):
                DIR = file_path[i]
                for name in os.listdir(DIR):
                    filelist.append(os.path.join(DIR, name))

        return filelist

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):

        # =====> sdf
        # print(self.all_filelist[idx])
        data = scipy.io.loadmat(self.all_filelist[idx])
        robot_state = data['configuration'] 
        coords = data['top_all']
        sdf = data['sdf']
        on_surface_index = data['on_surface_index'][0,:]
        off_surface_index = data['off_surface_index'][0,:]

        on_surface_num = on_surface_index.shape[0]
        off_surface_num = off_surface_index.shape[0]

        assert(off_surface_num+on_surface_num==sdf.shape[0])
    
        max_x = 4
        max_y = 2
        final_coords = coords
        final_coords[:,0] = final_coords[:,0]/max_x
        final_coords[:,1] = final_coords[:,1]/max_y 
        final_sdf = sdf.reshape(-1,1)

        if self.flag == "validation":
            final_coords1 = final_coords
            final_sdf1 = final_sdf

            assert final_coords.shape[0]==final_sdf.shape[0]
            assert final_coords1.shape[0]==final_sdf1.shape[0]
            total_samples = final_coords1.shape[0]

            # =====> robot state
            sel_robot_state = robot_state
            sel_robot_state = sel_robot_state.reshape(1, -1)
            sel_robot_state_1 = sel_robot_state[:,(0,1,3,4,6,7)]
            final_robot_states = np.tile(sel_robot_state_1, (total_samples, 1))
            # print(final_robot_states.shape)
        else:
            samples_num = 1500
            # point_cloud_size = final_coords.shape[0]
            rand_idcs_on = np.random.choice(on_surface_index, size=samples_num, replace=False)
            rand_idcs_off = np.random.choice(off_surface_index, size=samples_num, replace=False)

            final_coords1_on = final_coords[rand_idcs_on,:]
            final_coords1_off = final_coords[rand_idcs_off,:]
            final_sdf1_on = final_sdf[rand_idcs_on,:]
            final_sdf1_off = final_sdf[rand_idcs_off,:]

            final_coords1 = np.concatenate((final_coords1_on, final_coords1_off), axis=0)
            final_sdf1 = np.concatenate((final_sdf1_on, final_sdf1_off), axis=0)

            assert final_coords.shape[0]==final_sdf.shape[0]
            assert final_coords1.shape[0]==final_sdf1.shape[0]
            total_samples = final_coords1.shape[0]

            # =====> robot state
            sel_robot_state = robot_state
            sel_robot_state = sel_robot_state.reshape(1, -1)
            sel_robot_state_1 = sel_robot_state[:,(0,1,3,4,6,7)]
            final_robot_states = np.tile(sel_robot_state_1, (total_samples, 1))
            # print(final_robot_states.shape)

        return {'coords': torch.from_numpy(final_coords1).float(), 'states': torch.from_numpy(final_robot_states).float()},{'sdf': torch.from_numpy(final_sdf1).float()}