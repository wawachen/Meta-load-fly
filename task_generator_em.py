import numpy as np
import scipy.io as sio
import torch

class generate_task_data:
    def __init__(self):
        mat_contents_xyl1 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15.mat")
        data_obs_xyl1 = mat_contents_xyl1['obs']  # [5001,4]
        data_acs_xyl1 = mat_contents_xyl1['acs']  # [5000,2]
        new_train_in1 = np.concatenate([self.obs_preproc_3d(data_obs_xyl1[:-1]), data_acs_xyl1], axis=-1)
        new_train_targs1 = self.targ_proc(data_obs_xyl1[:-1], data_obs_xyl1[1:])

        mat_contents_xyl2 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.3_2agents_L1.0_dt_0.15.mat")
        data_obs_xyl2 = mat_contents_xyl2['obs']  # [5001,4]
        data_acs_xyl2 = mat_contents_xyl2['acs']  # [5000,2]
        new_train_in2 = np.concatenate([self.obs_preproc_3d(data_obs_xyl2[:-1]), data_acs_xyl2], axis=-1)
        new_train_targs2 = self.targ_proc(data_obs_xyl2[:-1], data_obs_xyl2[1:])

        mat_contents_xyl3 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.5_2agents_L0.8_dt_0.15.mat")
        data_obs_xyl3 = mat_contents_xyl3['obs']  # [5001,4]
        data_acs_xyl3 = mat_contents_xyl3['acs']  # [5000,2]
        new_train_in3 = np.concatenate([self.obs_preproc_3d(data_obs_xyl3[:-1]), data_acs_xyl3], axis=-1)
        new_train_targs3 = self.targ_proc(data_obs_xyl3[:-1], data_obs_xyl3[1:])

        mat_contents_xyl4 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.8_2agents_L1.2_dt_0.15.mat")
        data_obs_xyl4 = mat_contents_xyl4['obs']  # [5001,4]
        data_acs_xyl4 = mat_contents_xyl4['acs']  # [5000,2]
        new_train_in4 = np.concatenate([self.obs_preproc_3d(data_obs_xyl4[:-1]), data_acs_xyl4], axis=-1)
        new_train_targs4 = self.targ_proc(data_obs_xyl4[:-1], data_obs_xyl4[1:])

        self.new_train_in = [new_train_in1,new_train_in2,new_train_in3,new_train_in4]
        self.new_train_targs = [new_train_targs1,new_train_targs2,new_train_targs3,new_train_targs4]

        self.num_tasks = 4

    def generate_data(self):
        d_in = []
        d_out = []
        for i in range(self.num_tasks):
            x = self.new_train_in[i]
            y = self.new_train_targs[i]
            d_in.append(x)
            d_out.append(y)
        return d_in, d_out

    @staticmethod
    def obs_preproc_3d(obs):
        if isinstance(obs, np.ndarray): 
           return np.concatenate([obs[:, :9],np.sin(obs[:, 9]).reshape(-1,1), np.cos(obs[:, 9]).reshape(-1,1),np.sin(obs[:, 10]).reshape(-1,1), np.cos(obs[:, 10]).reshape(-1,1),np.sin(obs[:, 11]).reshape(-1,1), np.cos(obs[:, 11]).reshape(-1,1)], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, :9],
                obs[:, 9].sin().reshape(-1,1),
                obs[:, 9].cos().reshape(-1,1),
                obs[:, 10].sin().reshape(-1,1),
                obs[:, 10].cos().reshape(-1,1),
                obs[:, 11].sin().reshape(-1,1),
                obs[:, 11].cos().reshape(-1,1),
            ], dim=1)


    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
        