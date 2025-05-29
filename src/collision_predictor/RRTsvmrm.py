'''
Jingyu Chen modified from 

MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dotmap import DotMap
import torch
import scipy.io as sio
import scipy
from scipy.io import savemat
import time
import sys
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/models")
from collision_predictor import morphology_predictor
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/utils")
from utils import Graph, isInObstacle, nearest, newVertex, distance, Line, isThruObstacle

TORCH_DEVICE = torch.device('cuda')

THRESHOLD_DISTANCE = {
    "cross": 0.05,
    "square_c": {0: 0.06, 1: 0.07, 2: 0.06},
    "crowd1": {5: 0.05, "default": 0.03},
    "crowd2": 0.05
}
THRESHOLD_GLOBAL = {
    "cross": 0,
    "square_c": {0: 50, 1: 10, 2: 10},
    "crowd1": {5: 20, "default": 5},
    "crowd2": 10
}
# 0,1,2 4 is training env, 5,6 is testing env
UAV_OFFSETS = {
    0: (0.3024, 0.0786, -0.294, 0.0662),
    1: (0.2164, -0.0198, -0.4932, -0.018),
    2: (0.4876, -0.01, -0.5756, -0.0092),
    4: (0.2983, 0.0, -0.2983, 0.0),
    5: (0.1474, -0.0006, -0.4470, -0.0006),
    6: (0.2020, 0.0004, -0.3937, 0.0004)
}

def get_threshold_distance(rn, task_num):
    if rn == "square_c":
        return THRESHOLD_DISTANCE[rn].get(task_num, 0.07)
    if rn == "crowd1":
        return THRESHOLD_DISTANCE[rn].get(task_num, THRESHOLD_DISTANCE[rn]["default"])
    return THRESHOLD_DISTANCE.get(rn, 0.05)

def get_threshold_global(rn, task_num):
    if rn == "square_c":
        return THRESHOLD_GLOBAL[rn].get(task_num, 10)
    if rn == "crowd1":
        return THRESHOLD_GLOBAL[rn].get(task_num, THRESHOLD_GLOBAL[rn]["default"])
    return THRESHOLD_GLOBAL.get(rn, 0)

def get_uav_offsets(task_num):
    return UAV_OFFSETS.get(task_num, (0, 0, 0, 0))

def generate_obs_cloud(pos, radius, resolution=400, max_x=4.0, max_y=2.0):
    sample_num = round((radius * 2) / (max_x / resolution))
    cloud = np.zeros((sample_num ** 2, 2))
    for i in range(sample_num ** 2):
        r = radius * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        cloud[i, 0] = pos[0] + r * np.cos(theta)
        cloud[i, 1] = pos[1] + r * np.sin(theta)
    return cloud

def obs_cloud_to_grid(cloud, N=401, max_x=4.0, max_y=2.0):
    voxel_origin = [0, 2]
    voxel_size = max_x / (N - 1)
    overall_index = np.arange(0, N ** 2)
    samples = np.zeros([N ** 2, 3])
    samples[:, 0] = overall_index % N
    samples[:, 1] = (overall_index / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]
    mytree = scipy.spatial.cKDTree(samples[:, :2])
    _, indexes = mytree.query(cloud)
    unique_indexes = np.unique(indexes)
    final_obs = samples[unique_indexes, :2]
    final_obs[:, 0] /= max_x
    final_obs[:, 1] /= max_y
    return final_obs

def RRT(startpos, endpos, obstacles, n_iter, radius_l, radius_goal, stepSize):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius_l):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius_l)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        dist = distance(newvex, G.endpos)
        if dist < radius_goal:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            print('success')
            break
    return G

def RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius_l,radius_goal, stepSize):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()

        nearvex, nearidx = nearest(G, randvex, obstacles, radius_l)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)
        ######
        load_pos = np.array([[newvex[0],newvex[1]]])

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
        # print(state.shape)
        status = model.occupancy_predictor(state,threshold_distance,threshold_global)

        if status == 1:
            continue
       
        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        dist = distance(newvex, G.endpos)
        if dist < radius_goal:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            print('success')
            break
    return G


def dijkstra(G):
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)

        
    
if __name__ == '__main__':
    task_num = 0  #0,1,2,4,5,6 for cross and square, it generate routes based on the task 0,1,2; for crowd1 and crowd2, it generate routes based on the task 4,5,6
    rn = "cross"  #square_c,cross,crowd1,crowd2
    repeat = 0
    save_mat = True 
    max_x = 4.0
    max_y = 2.0
    max_z = 2.0

    threshold_distance = get_threshold_distance(rn, task_num)
    threshold_global = get_threshold_global(rn, task_num)
    dx1, dy1, dx2, dy2 = get_uav_offsets(task_num) 

    mat_save_path_square = './New_2d_save_corrective_waypoints_collision_square_{0}_{1}.mat'.format(task_num,repeat)
    mat_save_path_cross = './New_2d_save_corrective_waypoints_collision_cross_{0}_{1}.mat'.format(task_num,repeat)
    mat_save_path_crowd1 = './New_2d_save_corrective_waypoints_collision_crowd1_{0}_{1}.mat'.format(task_num,repeat)
    mat_save_path_crowd2 = './New_2d_save_corrective_waypoints_collision_crowd2_{0}_{1}.mat'.format(task_num,repeat)

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
    cfg.load_model_path = "/home/wawa/catkin_meta/src/MBRL_transport/checkpoints/logs_Occupancy_predictor_2d_movementall1_1/lightning_logs/version_0/checkpoints"

    model = morphology_predictor(cfg)
    route_name = rn

    #task 0,1,2
    if route_name == "square_c":
        obs_params = [([3.0, 0.5], 0.2), ([1, -1], 0.3)]
        obs_clouds = [generate_obs_cloud(pos, r) for pos, r in obs_params]
        final_obs_list = [obs_cloud_to_grid(cloud) for cloud in obs_clouds]
        model.add_obstacles([p for p, _ in obs_params], final_obs_list)
        trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/all_data/collision_predictor/original_path_obs/square/save_waypoints_collision_square_{0}.mat'.format(task_num)) 
        trajectory_load = trajectory['load']
        trajectory_uav1 = trajectory['uav1']
        trajectory_uav2 = trajectory['uav2']
        
        status_arr = np.zeros((trajectory_load.shape[0],1)) 

        plt.figure(figsize=(12, 12))

        start = time.time()
        
        for i in range(trajectory_load.shape[0]):
            load_pos = trajectory_load[i,:].reshape(1,-1)
            uav1_pos = trajectory_uav1[i,:].reshape(1,-1)
            uav2_pos =  trajectory_uav2[i,:].reshape(1,-1)

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
            
            # plt.scatter(load_pos[:,0],load_pos[:,1],c='k') 
            status = model.occupancy_predictor(state,threshold_distance,threshold_global,pt=False)
            
            if status == 1:
                status_s = 'True'
            else:
                status_s = 'False'
            
            status_arr[i,0] = status

        # print(sum(status_arr))
        index_fc = np.where(status_arr==0)[0]
        index_c = np.where(status_arr==1)[0]

        # #find gaps
        gaps = [0]
        for i in range(status_arr.shape[0]):
            if i == 0 or i==status_arr.shape[0]-1:
                continue
            if abs(status_arr[i+1]-status_arr[i])==1:
                gaps.append(i)

        gaps.append(status_arr.shape[0]-1)

        print(gaps)
        all_trajectory = []
        # #integration of final trajectory
        assert len(gaps)%2==0
        for j in range(len(gaps)):   
            if not j%2==0:    
                if gaps[j]==status_arr.shape[0]-1:
                    break

                startpos = (trajectory_load[gaps[j],0], trajectory_load[gaps[j],1])
                endpos = (trajectory_load[gaps[j+1]+1,0], trajectory_load[gaps[j+1]+1,1])
            
                obstacles = [(3.0, 0.5),(1,-1)]
                n_iter = 20000
                radius = [0.2,0.3]
                stepSize = 0.02

                G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)

                if G.success:
                    path = dijkstra(G)
                    path_np = np.array(path)
                    path_np_3d = np.ones((path_np.shape[0],3))*0.8
                    path_np_3d[:,:2] = path_np
                    all_trajectory.append(path_np_3d)
                else:
                    print("not found")
                    all_trajectory = []
                    break
            else:
                if j==0:   
                    all_trajectory.append(trajectory_load[gaps[j]:gaps[j+1]-1,:])
                else:
                    if gaps[j+1]==status_arr.shape[0]-1:
                        all_trajectory.append(trajectory_load[gaps[j]+2:gaps[j+1],:])
                    else:
                        all_trajectory.append(trajectory_load[gaps[j]+2:gaps[j+1]-1,:])
        
        print("Time costs: ",time.time()-start)

        all_trajectory_np = np.concatenate(all_trajectory,axis=0)
        plt.scatter(all_trajectory_np[:,0],all_trajectory_np[:,1],c='g')  
        plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
        plt.xlim(0, 4)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()

        if save_mat:
            savemat(mat_save_path_square, mdict={'load': all_trajectory_np})
    #task 0,1,2
    if route_name == "cross":
        obs_params = [([2.0, 0.0], 0.2)]
        obs_clouds = [generate_obs_cloud(pos, r) for pos, r in obs_params]
        final_obs_list = [obs_cloud_to_grid(cloud) for cloud in obs_clouds]
        model.add_obstacles([p for p, _ in obs_params], final_obs_list)
        trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/all_data/collision_predictor/original_path_obs/cross/save_waypoints_collision_cross_{0}.mat'.format(task_num)) 
        trajectory_load = trajectory['load']
        trajectory_uav1 = trajectory['uav1']
        trajectory_uav2 = trajectory['uav2']
        
        status_arr = np.zeros((trajectory_load.shape[0],1)) 

        start = time.time()

        plt.figure(figsize=(12, 12))

        for i in range(trajectory_load.shape[0]):
            load_pos = trajectory_load[i,:].reshape(1,-1)
            uav1_pos = trajectory_uav1[i,:].reshape(1,-1)
            uav2_pos =  trajectory_uav2[i,:].reshape(1,-1)

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
            status = model.occupancy_predictor(state,threshold_distance,threshold_global,pt=False)

            if status == 1:
                status_s = 'True'
            else:
                status_s = 'False'
           
            status_arr[i,0] = status

        index_fc = np.where(status_arr==0)[0]
        index_c = np.where(status_arr==1)[0]

        assert len(index_fc)+len(index_c) == status_arr.shape[0] 

        # #find gaps
        gaps = [0]
        for i in range(status_arr.shape[0]):
            if i == 0 or i==status_arr.shape[0]-1:
                continue
            if abs(status_arr[i+1]-status_arr[i])==1:
                gaps.append(i)

        gaps.append(status_arr.shape[0]-1)

        print(gaps)
        all_trajectory = []
        # #integration of final trajectory
        # print(index_c[0]-1)
        assert len(gaps)%2==0
        for j in range(len(gaps)):   
            if not j%2==0:    
                if gaps[j]==status_arr.shape[0]-1:
                    break

                startpos = (trajectory_load[gaps[j],0], trajectory_load[gaps[j],1])
                endpos = (trajectory_load[gaps[j+1]+1,0], trajectory_load[gaps[j+1]+1,1])
            
                obstacles = [(2.0, 0.0)]
                n_iter = 20000
                radius = [0.2]
                stepSize = 0.02

                G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)

                if G.success:
                    path = dijkstra(G)
                    # print(path)
                    # plot(G, obstacles, radius, path)
                    path_np = np.array(path)
                    path_np_3d = np.ones((path_np.shape[0],3))*0.8
                    path_np_3d[:,:2] = path_np
                    all_trajectory.append(path_np_3d)
                else:
                    print("not found")
                    all_trajectory = []
                    break
            else:
                if j==0:   
                    all_trajectory.append(trajectory_load[gaps[j]:gaps[j+1]-1,:])
                else:
                    if gaps[j+1]==status_arr.shape[0]-1:
                        all_trajectory.append(trajectory_load[gaps[j]+2:gaps[j+1],:])
                    else:
                        all_trajectory.append(trajectory_load[gaps[j]+2:gaps[j+1]-1,:])
        
        all_trajectory_np = np.concatenate(all_trajectory, axis=0)
        print("Time costs: ", time.time()-start)
        plt.scatter(all_trajectory_np[:,0],all_trajectory_np[:,1],c='g')  
        plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
        plt.xlim(0, 4)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()

        if save_mat:
            savemat(mat_save_path_cross, mdict={'load': all_trajectory_np})

    if route_name == "crowd1":
        obs_params = [([2.0, 1.2], 0.05), ([2.0, 0.0], 0.05), ([2.0, -1.2], 0.05), ([3.6, 0.0], 0.05), ([0.4, 0.0], 0.05)]
        obs_clouds = [generate_obs_cloud(pos, r) for pos, r in obs_params]
        final_obs_list = [obs_cloud_to_grid(cloud) for cloud in obs_clouds]
        model.add_obstacles([p for p, _ in obs_params], final_obs_list)
        startpos = (1.0,1.0)
        endpos = (3.0,-1.0)
        obstacles = [pos for pos, _ in obs_params]
        n_iter = 20000
        radius = [r for _, r in obs_params]
        stepSize = 0.02
        
        G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)

        if G.success:
            path = dijkstra(G)
            path_np = np.array(path)
            path_np_3d = np.ones((path_np.shape[0],3))*0.8
            path_np_3d[:,:2] = path_np
            all_trajectory = path_np_3d
        else:
            print("not found")
            all_trajectory = []
        
        if save_mat:
            savemat(mat_save_path_crowd1, mdict={'load': all_trajectory})

        plt.figure(figsize=(12, 12))
        plt.scatter(all_trajectory[:,0],all_trajectory[:,1],c='g')  
        plt.xlim(0, 4)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()
    
    if route_name == "crowd2":
        obs_params = [([4.2, 0.5], 0.05), ([1.5, -0.5], 0.05), ([2.5, -0.5], 0.05), ([4.2, 0.0], 0.05), ([1.5, -1.2], 0.05), ([2.5, -1.2], 0.05), ([4.2, -1.2], 0.05)]
        obs_clouds = [generate_obs_cloud(pos, r) for pos, r in obs_params]
        final_obs_list = [obs_cloud_to_grid(cloud) for cloud in obs_clouds]
        model.add_obstacles([p for p, _ in obs_params], final_obs_list)
        startpos = (1.0,1.0)
        endpos = (3.35,-2.0)
        obstacles = [pos for pos, _ in obs_params]
        n_iter = 20000
        radius = [r for _, r in obs_params]
        stepSize = 0.02
        
        G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)

        if G.success:
            path = dijkstra(G)
            path_np = np.array(path)
            path_np_3d = np.ones((path_np.shape[0],3))*0.8
            path_np_3d[:,:2] = path_np
            all_trajectory = path_np_3d
        else:
            print("not found")
            all_trajectory = []        
       
        if save_mat:
            savemat(mat_save_path_crowd2, mdict={'load': all_trajectory})
        
        plt.figure(figsize=(12, 12))
        plt.scatter(all_trajectory[:,0],all_trajectory[:,1],c='g')  
        plt.xlim(0, 4)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()
