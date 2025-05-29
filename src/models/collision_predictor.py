import glob
import os
from pytorch_lightning import seed_everything
import torch
import numpy as np
from occupancy_predictor_2d import Predictor_Model_2d
import matplotlib.pyplot as plt

TORCH_DEVICE = torch.device('cuda')

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

class morphology_predictor:
    def __init__(self, cfg):
        #load morphorlogy predictor
        checkpoint_filepath = cfg.load_model_path
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    
        seed(cfg)
        seed_everything(cfg.seed)

        self.occupancy_model = Predictor_Model_2d(lr=cfg.lr,
                                dof=cfg.dof,
                                if_cuda=cfg.if_cuda,
                                if_test=True,
                                gamma=cfg.gamma,
                                num_workers=cfg.num_workers,
                                coord_system=cfg.coord_system,
                                lr_schedule=cfg.lr_schedule)

        ckpt = torch.load(checkpoint_filepath)
        self.occupancy_model.load_state_dict(ckpt['state_dict'])
        self.occupancy_model = self.occupancy_model.to('cuda')
        self.occupancy_model.eval()
        self.occupancy_model.freeze()
    
    def add_obstacles(self, obs_l, obs_points_l):
        self.obs_pos_l = []
        self.obs_points_l = []

        for obs in obs_l:
           self.obs_pos_l.append(np.array(obs))

        for obs_points in obs_points_l:
           self.obs_points_l.append(obs_points) 

    def occupancy_predictor(self, state,threshold_distance, threshold_global, pt=False):
        # ###############################
        if pt == True:
            N=401
            max_batch=64 ** 2
            
            # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
            voxel_origin = [0, 2]
            voxel_size = 4.0 / (N - 1)

            overall_index = torch.arange(0, N ** 2, 1, out=torch.LongTensor())
            samples = torch.zeros(N ** 2, 3)

            # transform first 2 columns to be the x, y index
            samples[:, 0] = overall_index % N
            samples[:, 1] = (overall_index.long() / N) % N

            # transform first 3 columns to be the x, y, z coordinate
            samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
            samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]
        
            num_samples = N ** 2
            samples.requires_grad = False

            max_x = 4
            max_y = 2
            sample_test = samples[:,:2].cuda()
            sample_test[:,0] = sample_test[:,0]/max_x
            sample_test[:,1] = sample_test[:,1]/max_y

            cur_obs = torch.from_numpy(state).float().to(TORCH_DEVICE)
            # state_e = cur_obs[None]

            # final_robot_states = np.tile(state_e, (sample_test.shape[0], 1))
            final_robot_states = cur_obs.expand(sample_test.shape[0],-1)
            # final_robot_states = torch.from_numpy(final_robot_states).float().cuda()
            sample_set = torch.cat((sample_test, final_robot_states), dim=1)
            samples[:, 2] = (self.occupancy_model.model(sample_set).squeeze().detach().cpu())

            sdf_np_index = np.where(samples[:, 2]<=threshold_distance)[0]
            
            # plt.scatter(samples[sdf_np_index, 0],samples[sdf_np_index, 1],c='g', alpha = 0.1) 

        # ###############################
        # print(state)
        # print(current_p,self.obs_pos)
        collsion_l = []

        # sample_obs_points = 800
        cur_obs = torch.from_numpy(state).float().to(TORCH_DEVICE)

        load_obs = cur_obs[0,4:6]
        load_obs_np = load_obs.detach().cpu().numpy()
        # state_e = cur_obs[None]
        # print(cur_obs.shape)
        # state_expand = cur_obs.expand(self.obs_points_l[i].shape[0],-1)

        for i in range(len(self.obs_pos_l)):
            # d = np.sqrt(((current_p-self.obs_pos_l[i])**2).sum())
            state_expand = cur_obs.expand(self.obs_points_l[i].shape[0],-1)
        
            # if d<0.8:
            #########################
            # obs_num = self.obs_points_l[i].shape[0]
            # obs_num_shuffle = np.random.permutation(obs_num)
            # obs_index = obs_num_shuffle[:sample_obs_points]
            obs_points_e = torch.from_numpy(self.obs_points_l[i][:,:2]).float().to(TORCH_DEVICE)
            #########################
        
            input_c = torch.cat((obs_points_e,state_expand),1)
            # print(input_c.shape)
            sdf_output = self.occupancy_model.model(input_c)
            sdf_output_np = sdf_output.detach().cpu().numpy()
            # sdf_output_np_repulsive = sdf_output_np.copy()

            sdf_np_index_on1 = np.where(sdf_output_np<=threshold_distance)[0]

            input_np = input_c.detach().cpu().numpy()
            obs_points_e_np = obs_points_e.detach().cpu().numpy()

            if pt == True:
                plt.scatter(obs_points_e_np[:,0]*4,obs_points_e_np[:,1]*2,c='r')  
                if i==0:
                    plt.scatter(input_np[sdf_np_index_on1, 0]*4,input_np[sdf_np_index_on1, 1]*2,c='b') 
                else:
                    plt.scatter(input_np[sdf_np_index_on1, 0]*4,input_np[sdf_np_index_on1, 1]*2,c='b')  
            
            # print(cost_sum)
            threshold = threshold_global
            if len(sdf_np_index_on1)>threshold:
                cost1 = 1 
            else:
                cost1 = 0
            # else:
            #     cost1 = 0

            collsion_l.append(cost1)
        
        if pt == True:
            plt.xlim((0,4.5))
            plt.ylim((-2,2))
            # plt.axis('equal')
        
            # plt.show()
            plt.pause(0.01)
            plt.cla()

        return np.any(collsion_l)