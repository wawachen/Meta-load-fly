'''
Jingyu Chen modified from 

MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import scipy.io as sio
import time
import sys
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/utils")
from utils import Graph, isInObstacle, nearest, newVertex, distance

TORCH_DEVICE = torch.device('cuda')

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def detect_collision(load_pos, load_r, obs_l,r_l):
    sg = 0
    for obs, r in zip(obs_l, r_l):
        # print(np.linalg.norm(load_pos-obs),(r+load_r),r,load_r)
        if np.linalg.norm(load_pos-obs)<=(r+load_r):
            sg = 1
    
    return sg

def RRT(startpos, endpos, obstacles, n_iter, radius_l, radius_goal, stepSize,load_r):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius_l):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius_l)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        load_pos = np.array([[newvex[0],newvex[1]]])
        status = detect_collision(load_pos[0,:2],load_r,obstacles,radius_l)
    
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
    # Global parameters
    task_num = 0
    rn = "square"  #square_c,cross
    load_r = 0.2
    n_iter = 20000
    stepSize = 0.02
    radius_goal = 0.04
    max_x, max_y, max_z = 4.0, 2.0, 2.0

    # Obstacle parameters
    if rn == "square":
        obstacles = [(3.0, 0.5), (1, -1)]
        radius = [0.2, 0.3]
    else:  # cross
        obstacles = [(2.0, 0.0)]
        radius = [0.2]

    # Load trajectory
    base_path = '/home/wawa/catkin_meta/src/MBRL_transport/all_data/collision_predictor/original_path_obs'
    trajectory = sio.loadmat(f'{base_path}/{rn}/save_waypoints_collision_{rn}_{task_num}.mat')
    trajectory_load = trajectory['load']
    trajectory_uav1 = trajectory['uav1']
    trajectory_uav2 = trajectory['uav2']

    plt.figure(figsize=(12, 12))
    status_arr = np.zeros((trajectory_load.shape[0], 1))
    start = time.time()

    for i in range(trajectory_load.shape[0]):
        load_pos = trajectory_load[i,:].reshape(1,-1)
        status = detect_collision(load_pos[0,:2],load_r,obstacles,radius)

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
    assert len(gaps)%2==0
    for j in range(len(gaps)):   
        if not j%2==0:    
            if gaps[j]==status_arr.shape[0]-1:
                break

            startpos = (trajectory_load[gaps[j],0], trajectory_load[gaps[j],1])
            endpos = (trajectory_load[gaps[j+1]+1,0], trajectory_load[gaps[j+1]+1,1])
        
            G = RRT(startpos, endpos, obstacles, n_iter, radius, radius_goal, stepSize, load_r)

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
    
    all_trajectory_np = np.concatenate(all_trajectory, axis=0)
    print('The time cost is: ', time.time()-start)
    plt.scatter(all_trajectory_np[:,0],all_trajectory_np[:,1],c='g')  
    plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
    plt.xlim(0, 4)
    plt.ylim(-2, 2)
    plt.axis('equal')
    plt.show()

    # savemat('/home/wawa/catkin_meta/src/MBRL_transport/new_correction/New_2d_save_corrective_waypoints_collision_square.mat', mdict={'load': all_trajectory_np})