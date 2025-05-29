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
import time

add_noise = 1
    
resolution = 400

pos_obs2 = [4.2, 0.5]
obs_radius2 = 0.05
sample_num2 = round((obs_radius2 * 2) / (4.0 / resolution))

pos_obs3 = [1.5, -0.5]
obs_radius3 = 0.05
sample_num3 = round((obs_radius3 * 2) / (4.0 / resolution))

pos_obs4 = [2.5, -0.5]
obs_radius4 = 0.05
sample_num4 = round((obs_radius4 * 2) / (4.0 / resolution))

pos_obs5 = [4.2, 0.0]
obs_radius5 = 0.05
sample_num5 = round((obs_radius5 * 2) / (4.0 / resolution))

pos_obs6 = [1.5, -1.2]
obs_radius6 = 0.05
sample_num6 = round((obs_radius6 * 2) / (4.0 / resolution))

pos_obs7 = [2.5, -1.2]
obs_radius7 = 0.05
sample_num7 = round((obs_radius7 * 2) / (4.0 / resolution))

pos_obs8 = [4.2, -1.2]
obs_radius8 = 0.05
sample_num8 = round((obs_radius8 * 2) / (4.0 / resolution))

cloud_obs2 = np.zeros((sample_num2 ** 2, 2))
cloud_obs3 = np.zeros((sample_num3 ** 2, 2))
cloud_obs4 = np.zeros((sample_num4 ** 2, 2))
cloud_obs5 = np.zeros((sample_num5 ** 2, 2))
cloud_obs6 = np.zeros((sample_num6 ** 2, 2))
cloud_obs7 = np.zeros((sample_num7 ** 2, 2))
cloud_obs8 = np.zeros((sample_num8 ** 2, 2))

for i in range(sample_num2**2):
    r2 = obs_radius2 * np.sqrt(np.random.random())
    theta2 = np.random.random() * 2 * np.pi
    cloud_obs2[i,0] = pos_obs2[0] + r2 * np.cos(theta2)
    cloud_obs2[i,1] = pos_obs2[1] + r2 * np.sin(theta2)

for i in range(sample_num3**2):
    r3 = obs_radius3 * np.sqrt(np.random.random())
    theta3 = np.random.random() * 2 * np.pi
    cloud_obs3[i,0] = pos_obs3[0] + r3 * np.cos(theta3)
    cloud_obs3[i, 1] = pos_obs3[1] + r3 * np.sin(theta3)

for i in range(sample_num4**2):
    r4 = obs_radius4 * np.sqrt(np.random.random())
    theta4 = np.random.random() * 2 * np.pi
    cloud_obs4[i,0] = pos_obs4[0] + r4 * np.cos(theta4)
    cloud_obs4[i, 1] = pos_obs4[1] + r4 * np.sin(theta4)

for i in range(sample_num5**2):
    r5 = obs_radius5 * np.sqrt(np.random.random())
    theta5 = np.random.random() * 2 * np.pi
    cloud_obs5[i,0] = pos_obs5[0] + r5 * np.cos(theta5)
    cloud_obs5[i, 1] = pos_obs5[1] + r5 * np.sin(theta5)

for i in range(sample_num6**2):
    r6 = obs_radius6 * np.sqrt(np.random.random())
    theta6 = np.random.random() * 2 * np.pi
    cloud_obs6[i,0] = pos_obs6[0] + r6 * np.cos(theta6)
    cloud_obs6[i, 1] = pos_obs6[1] + r6 * np.sin(theta6)

for i in range(sample_num7**2):
    r7 = obs_radius7 * np.sqrt(np.random.random())
    theta7 = np.random.random() * 2 * np.pi
    cloud_obs7[i,0] = pos_obs7[0] + r7 * np.cos(theta7)
    cloud_obs7[i, 1] = pos_obs7[1] + r7 * np.sin(theta7)

for i in range(sample_num8**2):
    r8 = obs_radius8 * np.sqrt(np.random.random())
    theta8 = np.random.random() * 2 * np.pi
    cloud_obs8[i,0] = pos_obs8[0] + r8 * np.cos(theta8)
    cloud_obs8[i, 1] = pos_obs8[1] + r8 * np.sin(theta8)

N=401

# NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
voxel_origin = [0, 2]
voxel_size = 4.5 / (N - 1)

overall_index = np.arange(0, N ** 2, 1)
samples = np.zeros([N ** 2, 3])

# transform first 2 columns to be the x, y index
samples[:, 0] = overall_index % N
samples[:, 1] = (overall_index / N) % N

# transform first 3 columns to be the x, y, z coordinate
samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]

mytree = scipy.spatial.cKDTree(samples[:,:2])
dist_ckd2, indexes_ckd2 = mytree.query(cloud_obs2)
dist_ckd3, indexes_ckd3 = mytree.query(cloud_obs3)
dist_ckd4, indexes_ckd4 = mytree.query(cloud_obs4)
dist_ckd5, indexes_ckd5 = mytree.query(cloud_obs5)
dist_ckd6, indexes_ckd6 = mytree.query(cloud_obs6)
dist_ckd7, indexes_ckd7 = mytree.query(cloud_obs7)
dist_ckd8, indexes_ckd8 = mytree.query(cloud_obs8)

#make index uniques
indexes_ckd2_uni = list(np.unique(np.array(indexes_ckd2)))
indexes_ckd3_uni = list(np.unique(np.array(indexes_ckd3)))
indexes_ckd4_uni = list(np.unique(np.array(indexes_ckd4)))
indexes_ckd5_uni = list(np.unique(np.array(indexes_ckd5)))
indexes_ckd6_uni = list(np.unique(np.array(indexes_ckd6)))
indexes_ckd7_uni = list(np.unique(np.array(indexes_ckd7)))
indexes_ckd8_uni = list(np.unique(np.array(indexes_ckd8)))

final_obs2 = samples[indexes_ckd2_uni,:2]
final_obs3 = samples[indexes_ckd3_uni,:2]
final_obs4 = samples[indexes_ckd4_uni,:2]
final_obs5 = samples[indexes_ckd5_uni,:2]
final_obs6 = samples[indexes_ckd6_uni,:2]
final_obs7 = samples[indexes_ckd7_uni,:2]
final_obs8 = samples[indexes_ckd8_uni,:2]

if add_noise:
    mean = 0
    std = 0.03
    noise2 = np.random.normal(mean, std, final_obs2.shape)
    noise3 = np.random.normal(mean, std, final_obs3.shape)
    noise4 = np.random.normal(mean, std, final_obs4.shape)
    noise5 = np.random.normal(mean, std, final_obs5.shape)
    noise6 = np.random.normal(mean, std, final_obs6.shape)
    noise7 = np.random.normal(mean, std, final_obs7.shape)
    noise8 = np.random.normal(mean, std, final_obs8.shape)

    final_obs2 += noise2
    final_obs3 += noise3
    final_obs4 += noise4
    final_obs5 += noise5
    final_obs6 += noise6
    final_obs7 += noise7
    final_obs8 += noise8

plt.scatter(final_obs2[:, 0], final_obs2[:, 1], c='r')
plt.scatter(final_obs3[:, 0], final_obs3[:, 1], c='r')
plt.scatter(final_obs4[:, 0], final_obs4[:, 1], c='r')
plt.scatter(final_obs5[:, 0], final_obs5[:, 1], c='r')
plt.scatter(final_obs6[:, 0], final_obs6[:, 1], c='r')
plt.scatter(final_obs7[:, 0], final_obs7[:, 1], c='r')
plt.scatter(final_obs8[:, 0], final_obs8[:, 1], c='r')
plt.xlabel('x [m]', fontsize=16)
plt.ylabel('y [m]', fontsize=16)
plt.title('Crowd2 scenario', fontsize=18)
plt.show()