import scipy.io as sio
import numpy as np
import pickle

mat_contents_x1 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15.mat")
data_obs_x1 = mat_contents_x1['obs'][:,(6,7,8)]  # [10001,4]
data_acs_x1 = mat_contents_x1['acs']  # [10000,2]

# for each dataset this property is the same
data_all_num = data_obs_x1.shape[0]
assert data_all_num == 2501

mat_contents_x2 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.3_2agents_L1.0_dt_0.15.mat")
data_obs_x2 = mat_contents_x2['obs'][:,(6,7,8)]  # [10001,4]
data_acs_x2 = mat_contents_x2['acs']  # [10000,2]

mat_contents_x3 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.5_2agents_L0.8_dt_0.15.mat")
data_obs_x3 = mat_contents_x3['obs'][:,(6,7,8)]  # [10001,4]
data_acs_x3 = mat_contents_x3['acs']  # [10000,2]

mat_contents_x4 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.8_2agents_L1.2_dt_0.15.mat")
data_obs_x4 = mat_contents_x4['obs'][:,(6,7,8)]  # [10001,4]
data_acs_x4 = mat_contents_x4['acs']  # [10000,2]

#we split 10001 sampling points into  (M-1)/(N-1) trajectories, M is total points, N is the path_length
l = 501 # k_shot+k_query should be equal or less than l
num_of_trajectories = int((data_all_num-1)/(l-1)) 

trajs1_obs = []
trajs1_acs = []  
trajs1_latent = []
hold_num_of_trajectories1 = int(0.9*num_of_trajectories)
for i in range(num_of_trajectories):
    trajs1_obs.append(data_obs_x1[(l-1)*i:(l-1)*(i+1)])
    trajs1_acs.append(data_acs_x1[(l-1)*i:(l-1)*(i+1)])
    trajs1_latent.append(np.zeros((trajs1_obs[0].shape[0],1)))
random_index1 = list(np.random.permutation(num_of_trajectories))
trajs1_obs_train = []
trajs1_acs_train = []
trajs1_latent_train = []
for index1 in random_index1[:hold_num_of_trajectories1]:
    trajs1_obs_train.append(trajs1_obs[index1])
    trajs1_acs_train.append(trajs1_acs[index1])
    trajs1_latent_train.append(trajs1_latent[index1])
trajs1_obs_hold = []
trajs1_acs_hold = [] 
trajs1_latent_hold = []
for indexh1 in random_index1[hold_num_of_trajectories1:]:
    trajs1_obs_hold.append(trajs1_obs[indexh1])
    trajs1_acs_hold.append(trajs1_acs[indexh1])
    trajs1_latent_hold.append(trajs1_latent[indexh1])

trajs2_obs = []
trajs2_acs = []  
trajs2_latent = []
hold_num_of_trajectories2 = int(0.9*num_of_trajectories)
for i in range(num_of_trajectories):
    trajs2_obs.append(data_obs_x2[(l-1)*i:(l-1)*(i+1)])
    trajs2_acs.append(data_acs_x2[(l-1)*i:(l-1)*(i+1)])
    trajs2_latent.append(np.ones((trajs2_obs[0].shape[0],1)))
random_index2 = list(np.random.permutation(num_of_trajectories))
trajs2_obs_train = []
trajs2_acs_train = []
trajs2_latent_train = []
for index2 in random_index2[:hold_num_of_trajectories2]:
    trajs2_obs_train.append(trajs2_obs[index2])
    trajs2_acs_train.append(trajs2_acs[index2])
    trajs2_latent_train.append(trajs2_latent[index2])
trajs2_obs_hold = []
trajs2_acs_hold = []
trajs2_latent_hold = []
for indexh2 in random_index2[hold_num_of_trajectories2:]:
    trajs2_obs_hold.append(trajs2_obs[indexh2])
    trajs2_acs_hold.append(trajs2_acs[indexh2])
    trajs2_latent_hold.append(trajs2_latent[indexh2])

trajs3_obs = []
trajs3_acs = []  
trajs3_latent = []
hold_num_of_trajectories3 = int(0.9*num_of_trajectories)
for i in range(num_of_trajectories):
    trajs3_obs.append(data_obs_x3[(l-1)*i:(l-1)*(i+1)])
    trajs3_acs.append(data_acs_x3[(l-1)*i:(l-1)*(i+1)])
    trajs3_latent.append(np.ones((trajs3_obs[0].shape[0],1))*2)
random_index3 = list(np.random.permutation(num_of_trajectories))
trajs3_obs_train = []
trajs3_acs_train = []
trajs3_latent_train = []
for index3 in random_index3[:hold_num_of_trajectories3]:
    trajs3_obs_train.append(trajs3_obs[index3])
    trajs3_acs_train.append(trajs3_acs[index3])
    trajs3_latent_train.append(trajs3_latent[index3])
trajs3_obs_hold = []
trajs3_acs_hold = []
trajs3_latent_hold = []
for indexh3 in random_index3[hold_num_of_trajectories3:]:
    trajs3_obs_hold.append(trajs3_obs[indexh3])
    trajs3_acs_hold.append(trajs3_acs[indexh3])
    trajs3_latent_hold.append(trajs3_latent[indexh3])

trajs4_obs = []
trajs4_acs = []
trajs4_latent = []  
hold_num_of_trajectories4 = int(0.9*num_of_trajectories)
for i in range(num_of_trajectories):
    trajs4_obs.append(data_obs_x4[(l-1)*i:(l-1)*(i+1)])
    trajs4_acs.append(data_acs_x4[(l-1)*i:(l-1)*(i+1)])
    trajs4_latent.append(np.ones((trajs4_obs[0].shape[0],1))*3)
random_index4 = list(np.random.permutation(num_of_trajectories))
trajs4_obs_train = []
trajs4_acs_train = []
trajs4_latent_train = []
for index4 in random_index4[:hold_num_of_trajectories4]:
    trajs4_obs_train.append(trajs4_obs[index4])
    trajs4_acs_train.append(trajs4_acs[index4])
    trajs4_latent_train.append(trajs4_latent[index4])
trajs4_obs_hold = []
trajs4_acs_hold = []
trajs4_latent_hold = []
for indexh4 in random_index4[hold_num_of_trajectories4:]:
    trajs4_obs_hold.append(trajs4_obs[indexh4])
    trajs4_acs_hold.append(trajs4_acs[indexh4])
    trajs4_latent_hold.append(trajs4_latent[indexh4])


obs_all_train = trajs1_obs_train+trajs2_obs_train+trajs3_obs_train+trajs4_obs_train #[trajectory_num, path length, dim]
acs_all_train = trajs1_acs_train+trajs2_acs_train+trajs3_acs_train+trajs4_acs_train
latent_all_train = trajs1_latent_train + trajs2_latent_train + trajs3_latent_train + trajs4_latent_train

obs_all_hold = trajs1_obs_hold+trajs2_obs_hold+trajs3_obs_hold+trajs4_obs_hold #[trajectory_num, path length, dim]
acs_all_hold = trajs1_acs_hold+trajs2_acs_hold+trajs3_acs_hold+trajs4_acs_hold
latent_all_hold = trajs1_latent_hold + trajs2_latent_hold + trajs3_latent_hold + trajs4_latent_hold

m_dict = {"train_obs":obs_all_train,"train_acs":acs_all_train,"train_latent":latent_all_train,"hold_obs":obs_all_hold,"hold_acs":acs_all_hold,"hold_latent":latent_all_hold}

with open('/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/VI/train_hold_data.pkl', 'wb') as f:
    pickle.dump(m_dict, f)
