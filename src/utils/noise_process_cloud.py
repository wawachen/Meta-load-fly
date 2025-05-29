import scipy.spatial
import scipy.io
import numpy as np
import os
import scipy
import sys

task_num_str = 1
task_num = int(task_num_str)

#This code is going to transform the unordered 3d pointcloud into a 2d grid map
# task_num = 10

# this is for 2d movement 
if task_num == 1:
    load_path = "/home/wawa/catkin_meta/src/MBRL_transport/new_correction/noise_analysis"
    files = os.listdir(load_path)   
    file_num = len(files)

    save_path = "/home/wawa/catkin_meta/src/MBRL_transport/new_correction/noise_analysis/preprocess"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        assert(1==0)
 
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
    
train_num = int((file_num)*0.9)
file_list = list(np.random.permutation(list(range(0, file_num))))
    
train_file_list = file_list[:train_num]
test_file_list = file_list[train_num:]

file_count_train = 0
file_count_test = 0

for i in train_file_list:
    samples[:,2] = 0
    # 2d movement
    if task_num == 1:
        mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/new_correction/noise_analysis/{0}.mat".format(i))

    points = mat["top"]
    configs = mat["configuration"]

    # get testing robot states
    dist_ckd, indexes_ckd = mytree.query(points[:,:2])
    indexes_ckd_uni = list(np.unique(np.array(indexes_ckd)))

    if len(indexes_ckd_uni)<1500:
        continue

    samples[indexes_ckd_uni, 2] = 1
    # print(indexes_ckd)

    indexes_ckd_on = np.where(samples[:,2]==1)[0]
    indexes_ckd_off = np.where(samples[:,2]==0)[0]

    assert(indexes_ckd_on.shape[0]+indexes_ckd_off.shape[0])==samples.shape[0]

    #######transform a grid map into the signed distance function (interio is -, outer is +)
    img_tensor = samples[:,2].reshape(N,N)
    neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
    sd_img = img_tensor - 1.
    sd_img = sd_img.astype(np.uint8)
    signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
    signed_distances /= float(img_tensor.shape[1])
    signed_distances = signed_distances.reshape((-1,1))
    ##################################################################################

    mdic = {"configuration":configs,"top_all":samples[:,:2],"sdf":signed_distances,"on_surface_index":indexes_ckd_on,"off_surface_index":indexes_ckd_off}
    
    if not os.path.exists(save_path+"/train"):
        os.makedirs(save_path+"/train")

    scipy.io.savemat(save_path+"/train"+"/{0}.mat".format(file_count_train), mdic)
    file_count_train+=1

for i in test_file_list:
    # 2d movement
    samples[:,2] = 0
    if task_num == 1:
        mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/new_correction/noise_analysis/{0}.mat".format(i))
    
    points = mat["top"]
    configs = mat["configuration"]

    # get testing robot states
    dist_ckd, indexes_ckd = mytree.query(points[:,:2])
    indexes_ckd_uni = list(np.unique(np.array(indexes_ckd)))

    if len(indexes_ckd_uni)<1500:
        continue

    samples[indexes_ckd_uni, 2] = 1
    # print(indexes_ckd)

    indexes_ckd_on = np.where(samples[:,2]==1)[0]
    indexes_ckd_off = np.where(samples[:,2]==0)[0]

    assert(indexes_ckd_on.shape[0]+indexes_ckd_off.shape[0])==samples.shape[0]

    #######transform a grid map into the signed distance function (interio is -, outer is +)
    img_tensor = samples[:,2].reshape(N,N)
    neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
    sd_img = img_tensor - 1.
    sd_img = sd_img.astype(np.uint8)
    signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
    signed_distances /= float(img_tensor.shape[1])
    signed_distances = signed_distances.reshape((-1,1))
    ##################################################################################

    mdic = {"configuration":configs,"top_all":samples[:,:2],"sdf":signed_distances,"on_surface_index":indexes_ckd_on,"off_surface_index":indexes_ckd_off}
    
    if not os.path.exists(save_path+"/validation"):
        os.makedirs(save_path+"/validation")
    
    scipy.io.savemat(save_path+"/validation"+"/{0}.mat".format(file_count_test), mdic)
    file_count_test+=1




