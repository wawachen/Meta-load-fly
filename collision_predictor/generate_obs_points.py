import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

shape = "square"

if shape == "square":
    #add square obstacles here
    pos_obs = [3.0,0.5] #2d position
    resolution = 400
    obs_radius = 0.2 # 
    sample_num = round((obs_radius*2)/(4.0/resolution))

    pos_obs1 = [1,-1]
    obs_radius1 = 0.3
    sample_num1 = round((obs_radius1*2)/(4.0/resolution))

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

    # plt.figure(1)
    # plt.scatter(cloud_obs[:,0],cloud_obs[:,1])

    # plt.figure(2)
    # plt.scatter(cloud_obs1[:,0],cloud_obs1[:,1])
    # plt.show()

    # cloud_obs[:,0] = np.random.uniform(pos_obs[0]-obs_size/2.0,pos_obs[0]+obs_size/2.0,sample_num**2)
    # cloud_obs[:,1] = np.random.uniform(pos_obs[1]-obs_size/2.0,pos_obs[1]+obs_size/2.0,sample_num**2)

    # cloud_obs1[:,0] = np.random.uniform(pos_obs1[0]-obs_size1/2.0,pos_obs1[0]+obs_size1/2.0,sample_num1**2)
    # cloud_obs1[:,1] = np.random.uniform(pos_obs1[1]-obs_size1/2.0,pos_obs1[1]+obs_size1/2.0,sample_num1**2)

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

    #normlization
    max_x = 4
    max_y = 2
    final_obs[:,0] = final_obs[:,0]/max_x
    final_obs[:,1] = final_obs[:,1]/max_y

    final_obs1[:,0] = final_obs1[:,0]/max_x
    final_obs1[:,1] = final_obs1[:,1]/max_y

    # plt.figure(1)
    # plt.scatter(final_obs[:,0]*4,final_obs[:,1]*2)

    # plt.figure(2)
    # plt.scatter(final_obs1[:,0]*4,final_obs1[:,1]*2)
    # plt.show()

    mdic = {"obs":final_obs,"obs1":final_obs1,"obs_pos":pos_obs,"obs_pos1":pos_obs1,"obs_radius":obs_radius,"obs_radius1":obs_radius1}
    scipy.io.savemat("/home/wawa/catkin_meta/src/MBRL_transport/obs_points2.mat",mdic)
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

    # plt.figure(1)
    # plt.scatter(cloud_obs[:,0],cloud_obs[:,1])

    # plt.figure(2)
    # plt.scatter(cloud_obs1[:,0],cloud_obs1[:,1])
    # plt.show()

    # cloud_obs[:,0] = np.random.uniform(pos_obs[0]-obs_size/2.0,pos_obs[0]+obs_size/2.0,sample_num**2)
    # cloud_obs[:,1] = np.random.uniform(pos_obs[1]-obs_size/2.0,pos_obs[1]+obs_size/2.0,sample_num**2)

    # cloud_obs1[:,0] = np.random.uniform(pos_obs1[0]-obs_size1/2.0,pos_obs1[0]+obs_size1/2.0,sample_num1**2)
    # cloud_obs1[:,1] = np.random.uniform(pos_obs1[1]-obs_size1/2.0,pos_obs1[1]+obs_size1/2.0,sample_num1**2)

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

    # plt.figure(1)
    # plt.scatter(final_obs[:,0]*4,final_obs[:,1]*2)

    # plt.figure(2)
    # plt.scatter(final_obs1[:,0]*4,final_obs1[:,1]*2)
    # plt.show()

    mdic = {"obs":final_obs,"obs_pos":pos_obs,"obs_radius":obs_radius}
    scipy.io.savemat("/home/wawa/catkin_meta/src/MBRL_transport/obs_points1.mat",mdic)



