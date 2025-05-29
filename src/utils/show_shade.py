import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import deque

import glob
from dotmap import DotMap
from pytorch_lightning import seed_everything
from occupancy_predictor_2d import Predictor_Model_2d
import os
import torch
import scipy.io as sio
import scipy
from scipy.io import savemat

TORCH_DEVICE = torch.device('cuda')

task_num = 2
rn = "cross" #square_c,cross

if rn == "cross":
    threshold_distance = 0.05
else:
    if task_num == 0 or task_num == 2:
        threshold_distance = 0.06
    else:
        threshold_distance = 0.07

if rn == "cross":
    threshold_global = 0#
else:
    if task_num == 0:
        threshold_global = 50#
    if task_num == 1:
        threshold_global = 10#
    if task_num == 2:
        threshold_global = 10

if task_num == 0:
    dx1 = 0.3024
    dy1 = 0.0786

    dx2 = -0.294
    dy2 = 0.0662

if task_num == 1:
    dx1 = 0.2164
    dy1 = -0.0198

    dx2 = -0.4932
    dy2 = -0.018

if task_num == 2:
    dx1 = 0.4876
    dy1 = -0.01

    dx2 = -0.5756
    dy2 = -0.0092

# draw shaded area
shade_x = []
shade_y = []

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

class morphology_predictor:
    def __init__(self,obs_l,obs_points_l):
        #load morphorlogy predictor
        checkpoint_filepath = "/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_1/lightning_logs/version_0/checkpoints"
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
        cfg = DotMap()
        cfg.seed = 1
        cfg.lr = 0.00005 # more_layers: 0.00005, one layer: 0.0001
        cfg.if_cuda = True
        cfg.gamma = 0.5
        cfg.log_dir = 'logs'
        cfg.num_workers = 8
        cfg.model_name = 'Occupancy_predictor'
        cfg.lr_schedule = [10000000]
        cfg.num_gpus = 1
        cfg.epochs = 10
        cfg.dof = 6
        cfg.coord_system = 'cartesian'
        cfg.tag = '2d_movement'
        seed(cfg)
        seed_everything(cfg.seed)

        self.obs_pos_l = []
        self.obs_points_l = []

        for obs in obs_l:
           self.obs_pos_l.append(np.array(obs))

        for obs_points in obs_points_l:
           self.obs_points_l.append(obs_points) 

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

    def occupancy_predictor(self, state,pt=False):
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
            
            plt.scatter(samples[sdf_np_index, 0],samples[sdf_np_index, 1],c='g', alpha = 0.1) 

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
            plt.xlim((0,4))
            plt.ylim((-2,2))
            # plt.axis('equal')
        
            # plt.show()
            plt.pause(0.01)
            plt.cla()

        return np.any(collsion_l)

if __name__ == "__main__":
    if rn == "square":
        #add square obstacles here
        pos_obs = [3.0,0.5] #2d position
        resolution = 400
        obs_radius = 0.2 # 
        sample_num = round((obs_radius*2)/(4.0/resolution))
        # print(sample_num)

        pos_obs1 = [1,-1]
        obs_radius1 = 0.3
        sample_num1 = round((obs_radius1*2)/(4.0/resolution)) 
        # print(sample_num1)

        cloud_obs = np.zeros((sample_num**2,2))
        cloud_obs1 = np.zeros((sample_num1**2,2))

        for i in range(sample_num**2):
            r = obs_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            cloud_obs[i,0] = pos_obs[0] + r * np.cos(theta)
            cloud_obs[i,1] = pos_obs[1] + r * np.sin(theta)

        for i in range(sample_num1**2):
            r1 = obs_radius1 * np.sqrt(np.random.random())
            theta1 = np.random.random() * 2 * np.pi
            cloud_obs1[i,0] = pos_obs1[0] + r1* np.cos(theta1)
            cloud_obs1[i,1] = pos_obs1[1] + r1 * np.sin(theta1)

        N=401

        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [0, 2]
        voxel_size = 4.0 / (N - 1)

        overall_index = np.arange(0, N ** 2, 1)
        samples = np.zeros([N ** 2, 3])

        # transform first 2 columns to be the x, y index
        samples[:, 0] = overall_index % N
        samples[:, 1] = (overall_index / N) % N

        # transform first 3 columns to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
        samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]

        mytree = scipy.spatial.cKDTree(samples[:,:2])
        dist_ckd, indexes_ckd = mytree.query(cloud_obs)
        dist_ckd1, indexes_ckd1 = mytree.query(cloud_obs1)

        #make index uniques
        indexes_ckd_uni = list(np.unique(np.array(indexes_ckd)))
        indexes_ckd1_uni = list(np.unique(np.array(indexes_ckd1)))

        final_obs = samples[indexes_ckd_uni,:2]
        final_obs1 = samples[indexes_ckd1_uni,:2]

        # plt.scatter(final_obs[:, 0],final_obs[:, 1],c='b') 
        # plt.scatter(final_obs1[:,0],final_obs1[:,1],c='r')  
        # plt.show()
        # print(final_obs.shape)
        # print(final_obs1.shape)

        #normlization
        max_x = 4
        max_y = 2
        final_obs[:,0] = final_obs[:,0]/max_x
        final_obs[:,1] = final_obs[:,1]/max_y

        final_obs1[:,0] = final_obs1[:,0]/max_x
        final_obs1[:,1] = final_obs1[:,1]/max_y

        ##########
    
        model = morphology_predictor([pos_obs, pos_obs1], [final_obs, final_obs1])

        all_trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/2d_save_corrective_waypoints_collision_square_{0}.mat'.format(task_num))
        all_trajectory_np = all_trajectory['load']
    else:
        #add square obstacles here
        pos_obs = [2.0,0.0] #2d position
        resolution = 400
        obs_radius = 0.2 # 
        sample_num = round((obs_radius*2)/(4.0/resolution))

        cloud_obs = np.zeros((sample_num**2,2))

        for i in range(sample_num**2):
            r = obs_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            cloud_obs[i,0] = pos_obs[0] + r * np.cos(theta)
            cloud_obs[i,1] = pos_obs[1] + r * np.sin(theta)

        N=401

        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [0, 2]
        voxel_size = 4.0 / (N - 1)

        overall_index = np.arange(0, N ** 2, 1)
        samples = np.zeros([N ** 2, 3])

        # transform first 2 columns to be the x, y index
        samples[:, 0] = overall_index % N
        samples[:, 1] = (overall_index / N) % N

        # transform first 3 columns to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
        samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]

        mytree = scipy.spatial.cKDTree(samples[:,:2])
        dist_ckd, indexes_ckd = mytree.query(cloud_obs)

        #make index uniques
        indexes_ckd_uni = list(np.unique(np.array(indexes_ckd)))

        final_obs = samples[indexes_ckd_uni,:2]

        #normlization
        max_x = 4
        max_y = 2
        final_obs[:,0] = final_obs[:,0]/max_x
        final_obs[:,1] = final_obs[:,1]/max_y
        ##########

        model = morphology_predictor([pos_obs],[final_obs])
    
        all_trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/save_corrective_waypoints_collision_cross_{0}.mat'.format(task_num))
        all_trajectory_np = all_trajectory['load']

    for i in range(all_trajectory_np.shape[0]):
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
        sample_test[:, 1] = sample_test[:, 1] / max_y
        
        # creat state
        load_pos = np.array([[all_trajectory_np[i,0],all_trajectory_np[i,1]]])

        ##
        waypoints_uav1_p = np.zeros(2)
        waypoints_uav1_p[0] = load_pos[0,0]+dx1
        waypoints_uav1_p[1] = load_pos[0,1]+dy1
        
        waypoints_uav2_p = np.zeros(2) 
        waypoints_uav2_p[0] = load_pos[0,0]+dx2
        waypoints_uav2_p[1] = load_pos[0,1]+dy2
        
        ##
        uav1_pos = waypoints_uav1_p.reshape(1,-1)
        uav2_pos =  waypoints_uav2_p.reshape(1,-1)

        #normalisation
        uav1_pos_norm = np.zeros((1,2))
        uav1_pos_norm[0,0] = uav1_pos[0,0]/max_x
        uav1_pos_norm[0,1] = uav1_pos[0,1]/max_y

        uav2_pos_norm = np.zeros((1,2))
        uav2_pos_norm[0,0] = uav2_pos[0,0]/max_x
        uav2_pos_norm[0,1] = uav2_pos[0,1]/max_y

        load_pos_norm = np.zeros((1,2))
        load_pos_norm[0,0] = load_pos[0,0]/max_x
        load_pos_norm[0,1] = load_pos[0,1]/max_y
        
        state = np.concatenate([uav1_pos_norm,uav2_pos_norm,load_pos_norm],axis=1)

        cur_obs = torch.from_numpy(state).float().to(TORCH_DEVICE)
        # state_e = cur_obs[None]

        # final_robot_states = np.tile(state_e, (sample_test.shape[0], 1))
        final_robot_states = cur_obs.expand(sample_test.shape[0],-1)
        # final_robot_states = torch.from_numpy(final_robot_states).float().cuda()
        sample_set = torch.cat((sample_test, final_robot_states), dim=1)
        samples[:, 2] = (model.occupancy_model.model(sample_set).squeeze().detach().cpu())

        sdf_np_index = np.where(samples[:, 2] <= threshold_distance)[0]
        
        shade_x.append(samples[sdf_np_index, 0])
        shade_y.append(samples[sdf_np_index, 1])

    for i in range(len(model.obs_pos_l)):
        # d = np.sqrt(((current_p-self.obs_pos_l[i])**2).sum())
        state_expand = cur_obs.expand(model.obs_points_l[i].shape[0],-1)
    
        # if d<0.8:
        #########################
        # obs_num = self.obs_points_l[i].shape[0]
        # obs_num_shuffle = np.random.permutation(obs_num)
        # obs_index = obs_num_shuffle[:sample_obs_points]
        obs_points_e = torch.from_numpy(model.obs_points_l[i][:,:2]).float().to(TORCH_DEVICE)
        #########################
    
        input_c = torch.cat((obs_points_e,state_expand),1)
        # print(input_c.shape)
        sdf_output = model.occupancy_model.model(input_c)
        sdf_output_np = sdf_output.detach().cpu().numpy()
        # sdf_output_np_repulsive = sdf_output_np.copy()

        sdf_np_index_on1 = np.where(sdf_output_np<=threshold_distance)[0]

        input_np = input_c.detach().cpu().numpy()
        obs_points_e_np = obs_points_e.detach().cpu().numpy()

        
        plt.scatter(obs_points_e_np[:,0]*4,obs_points_e_np[:,1]*2,c='r')  

        
    plt.scatter(np.concatenate(shade_x), np.concatenate(shade_y), c='g', alpha=0.1)
    savemat('shade_x_{0}.mat'.format(task_num), mdict={'arr': np.concatenate(shade_x)})
    savemat('shade_y_{0}.mat'.format(task_num),mdict={'arr':np.concatenate(shade_y)})
    plt.xlim(0, 4)
    plt.ylim(-2, 2)
    plt.axis('equal')
    plt.show()