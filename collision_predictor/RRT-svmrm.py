'''
Jingyu Chen modified from 

MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

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
rn = "square_c" #square_c,cross

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

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

class Line():
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn

def Intersection(line, center, radius):
    a = np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a);
    t2 = (-b - np.sqrt(discriminant)) / (2 * a);

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True

def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def isInObstacle(vex, obstacles, radius_l):
    for (obs,radius) in zip(obstacles,radius_l):
        if distance(obs, vex) < radius:
            return True
    return False

def isThruObstacle(line, obstacles, radius_l):
    for (obs,radius) in zip(obstacles,radius_l):
        if Intersection(line, obs, radius):
            return True
    return False

def nearest(G, vex, obstacles, radius_l):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius_l):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx

def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min (stepSize, length)

    newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
    return newvex

def window(startpos, endpos):
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height


def isInWindow(pos, winx, winy, width, height):
    if winx < pos[0] < winx+width and \
        winy < pos[1] < winy+height:
        return True
    else:
        return False


class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

    def randomPosition(self):
        p = 0.5
        if random()>p:
            posx = self.endpos[0]+np.random.uniform(-1,1)*0.5
            posy = self.endpos[1]+np.random.uniform(-1,1)*0.5
        else:
            posx = np.random.uniform(0,4)
            posy = np.random.uniform(-2,2)

        return posx, posy


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
        status = model.occupancy_predictor(state)
        # if status == 1:
        #     # print(all_num)
        #     if all_num<filter_num:
        #         status = 0
        ######
        if status == 1:
            continue

        plt.scatter(newvex[0],newvex[1],c='r') 
        plt.pause(0.05)
       
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

def RRT_star(model,startpos, endpos, obstacles, n_iter, radius_l, radius_goal, stepSize):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius_l):
            continue

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
        status = model.occupancy_predictor(state)
        # if status == 1:
        #     # print(all_num)
        #     if all_num<filter_num:
        #         status = 0
        ######
        if status == 1:
            continue

        plt.scatter(newvex[0],newvex[1],c='r') 
        plt.pause(0.05)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        
        G.distances[newidx] = G.distances[nearidx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > 0.1:
                continue

            line = Line(vex, newvex)
            if isThruObstacle(line, obstacles, radius_l):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

        dist = distance(newvex, G.endpos)
        if dist < radius_goal:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

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


def plot(G, obstacles, radius_l, path=None):
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    fig, ax = plt.subplots()

    for obs,radius in zip(obstacles,radius_l):
        circle = plt.Circle(obs, radius, color='red')
        ax.add_artist(circle)

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    # plt.show()

def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path

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
        
    
if __name__ == '__main__':
    ##########
    route_name = rn
    if route_name == "square_c":
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

        model = morphology_predictor([pos_obs,pos_obs1],[final_obs,final_obs1])
        trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_square_{0}.mat'.format(task_num)) 
        # trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_cross.mat') 
        trajectory_load = trajectory['load']
        trajectory_uav1 = trajectory['uav1']
        trajectory_uav2 = trajectory['uav2']
        
        status_arr = np.zeros((trajectory_load.shape[0],1)) 
        max_x = 4.0
        max_y = 2.0
        max_z = 2.0

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
            # print(state.shape)
            plt.scatter(load_pos[:,0],load_pos[:,1],c='k') 
            status = model.occupancy_predictor(state,pt=True)
            # if i==0:
            #     plt.pause(10)
            if status == 1:
                status_s = 'True'
            else:
                status_s = 'False'
            plt.scatter(trajectory_load[:,0],trajectory_load[:,1],c='c')  
            plt.text(1.0, 2.0, 'Coordinate:({:.2f},{:.2f}), Collision: {}'.format(load_pos[:,0][0],load_pos[:,1][0],status_s),fontsize=14,color="white",bbox ={'facecolor':'grey', 'pad':10})
            plt.text(2.8, 0.5, 'Obstacle 1',fontsize=11,style ="oblique")
            plt.text(0.8, -1.0, 'Obstacle 2',fontsize=11,style ="oblique")
            #############
            # status = model.occupancy_predictor(state[0])
            # print(state[0])
            # model.generate_sdf_map(state,load_pos)
            # if status == 1:
            #     # print(all_num)
            #     if all_num<filter_num:
            #         status = 0
            status_arr[i,0] = status
        plt.show()

        # print(sum(status_arr))
        index_fc = np.where(status_arr==0)[0]
        index_c = np.where(status_arr==1)[0]

        plt.scatter(trajectory_load[index_fc,0],trajectory_load[index_fc,1],c='g')   
        plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
        # print(index_fc) 
        # print(index_c)

        assert len(index_fc)+len(index_c) == status_arr.shape[0] 
        # plt.scatter(trajectory_load[:,0],trajectory_load[:,1])
        # plt.plot(trajectory_uav1[:,0],trajectory_uav1[:,1])
        # plt.plot(trajectory_uav2[:,0],trajectory_uav2[:,1])
        plt.show()

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
            
                obstacles = [(3.0, 0.5),(1,-1)]
                n_iter = 20000
                radius = [0.2,0.3]
                stepSize = 0.02

                # G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
                # G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)
                G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)
                # G= RRT_star(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)
                plt.show()

                if G.success:
                    path = dijkstra(G)
                    # print(path)
                    # plot(G, obstacles, radius, path)
                    path_np = np.array(path)
                    path_np_3d = np.ones((path_np.shape[0],3))*0.8
                    path_np_3d[:,:2] = path_np
                    all_trajectory.append(path_np_3d)

                    # # path1 = dijkstra(G1)
                    # # print(path1)
                    # # plot(G1, obstacles, radius, path)
                    # plt.show()
                else:
                    # plot(G, obstacles, radius)
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
        
        all_trajectory_np = np.concatenate(all_trajectory,axis=0)
        plt.scatter(all_trajectory_np[:,0],all_trajectory_np[:,1],c='g')  
        plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
        plt.xlim(0, 4)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()
        # savemat('/home/wawa/catkin_meta/src/MBRL_transport/2d_save_corrective_waypoints_collision_square_{0}.mat'.format(task_num), mdict={'load': all_trajectory_np})
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
        # trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_square.mat') 
        trajectory = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_cross_{0}.mat'.format(task_num)) 
        trajectory_load = trajectory['load']
        trajectory_uav1 = trajectory['uav1']
        trajectory_uav2 = trajectory['uav2']
        
        status_arr = np.zeros((trajectory_load.shape[0],1)) 
        max_x = 4.0
        max_y = 2.0
        max_z = 2.0

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
            plt.scatter(load_pos[:,0],load_pos[:,1],c='k') 
            status = model.occupancy_predictor(state,pt=True)

            # if i==0:
            #     plt.pause(10)

            if status == 1:
                status_s = 'True'
            else:
                status_s = 'False'
            plt.scatter(trajectory_load[:,0],trajectory_load[:,1],c='c')  
            plt.text(1.0, 2.0, 'Coordinate:({:.2f},{:.2f}), Collision: {}'.format(load_pos[:,0][0],load_pos[:,1][0],status_s),fontsize=14,color="white",bbox ={'facecolor':'grey', 'pad':10})
            plt.text(2.0, 0.0, 'Obstacle 1',fontsize=11,style ="oblique")
            # print(state[0].shape)
            # if status == 1:
            #     # print(all_num)
            #     if all_num<35:
            #         status = 0
            status_arr[i,0] = status
        plt.show()

        # print(sum(status_arr))
        index_fc = np.where(status_arr==0)[0]
        index_c = np.where(status_arr==1)[0]

        plt.scatter(trajectory_load[index_fc,0],trajectory_load[index_fc,1],c='g')   
        plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
        # print(index_fc) 
        # print(index_c)

        assert len(index_fc)+len(index_c) == status_arr.shape[0] 
        # plt.scatter(trajectory_load[:,0],trajectory_load[:,1])
        # plt.plot(trajectory_uav1[:,0],trajectory_uav1[:,1])
        # plt.plot(trajectory_uav2[:,0],trajectory_uav2[:,1])
        plt.show()

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

                # G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
                G = RRT_morphology(model,startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)
                plt.show()
                # G= RRT(startpos, endpos, obstacles, n_iter, radius, 0.04, stepSize)

                if G.success:
                    path = dijkstra(G)
                    # print(path)
                    # plot(G, obstacles, radius, path)
                    path_np = np.array(path)
                    path_np_3d = np.ones((path_np.shape[0],3))*0.8
                    path_np_3d[:,:2] = path_np
                    all_trajectory.append(path_np_3d)

                    # # path1 = dijkstra(G1)
                    # # print(path1)
                    # # plot(G1, obstacles, radius, path)
                    # plt.show()
                else:
                    # plot(G, obstacles, radius)
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
        
        all_trajectory_np = np.concatenate(all_trajectory,axis=0)
        plt.scatter(all_trajectory_np[:,0],all_trajectory_np[:,1],c='g')  
        plt.scatter(trajectory_load[index_c,0],trajectory_load[index_c,1],c='r') 
        plt.xlim(0, 4)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.show()

        # savemat('/home/wawa/catkin_meta/src/MBRL_transport/save_corrective_waypoints_collision_cross_{0}.mat'.format(task_num), mdict={'load': all_trajectory_np})