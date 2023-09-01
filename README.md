# Meta-Load Fly: a Reinforcement Learning Framework for Adaptive Load Tracking with Collision Prediction
[Jingyu Chen](https://www.researchgate.net/profile/Jingyu-Chen-20) <br>   
The University of Sheffield

[Project website]() | [Paper]()

## Overview of the Meta-load-fly
![](https://github.com/wawachen/Meta-load-fly/blob/main/image/method_icra(1).png)
<p align="center">Overview of Meta-Load Fly with load trajectory tracking and path planning; A. Path planning B. Corrective policy</p>


## Introduction 
Our code mainly consists of `CrazyS`, `MBRL_transport` and `openai_ros` along with the `mav_comm` and `gemotry_tf2` for python3 compiling.
It has been tested in Ubuntu 18.04 Melodic ROS and Ubuntu 20.04 Neotic. The code structure is shown below.

<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/github_1.png" width="60%"></div>
<p align="center">The relationship between different packages</p>

**CrazyS**: Provide the model of the cable-suspended Firefly-load system and the low-based tracking controller. The joy plugin and wind plugin are modified.<br>
**MBRL_transport**：This is the main code where we contribute to our meta-load-fly framework consisting of the adaptive load trajectory tracking module and the collision predictor.<br>
**Openai_ros**: A bridge for connecting Gazebo with Pytorch. We also modify it here for building the task environment for transport. The basic movements of Firefly are defined here.<br>

## Install Dependencies
Firstly, create a ROS workspace (the tutorials can be found [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)). Create an empty package `MBRL_transport` and a src folder inside it. Copy the content of this repository into the src folder of  `MBRL_transport` package. Then, the config and launch folder in this repository should be moved outside of this src folder to form a complete ROS package. The `mav_comm` and `gemotry_tf2` ROS packages need to be installed.<br>

To install `CrazyS` and `Openai_ros`, please download our forked repositories

```
git clone git@github.com:wawachen/CrazyS.git<br>
git clone git@github.com:wawachen/openai_ros.git
```

If you are using Ubuntu 18.04 Melodic ROS, the tricky thing is that we will use Python3 in Melodic ROS whose default Python is 2.7.
Thus, when we import these packages into the Catkin workspace, we use the following command to indicate for ROS that we are using python3 not python2 for compiling files.

```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Then, for the `openai_ros` package, we will use the tf package which was originally designed for python2. Thus we have to download the source file from [here](https://github.com/ros/geometry2) and compile it in the Python3 environment using the following command,

```
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

Another bug will take place in the gazebo_plugin from `CrazyS` <br>
It shows no definition of *protobuf* as the mismatch between the library *protobuf* and the *protobuf* from our virtual environment.<br>
Therefore, the solution is that we can directly uninstall the *protobuf* of our virtual environment (we use conda here)

```
conda uninstall libprotobuf
```
If you are using Ubuntu 20.04 Neotic, as the default Python of Neotic is Python3, the Python 2 problem does not exist. Just install the following packages

```
pip install gym==0.15.4
pip install gitpython
pip install dotmap==1.2.20
pip install tqdm==4.19.4
pip install tensorflow
pip install tensorboardX
```
### Main code
In the start_training.py, it has different modes for different functions（the unmentioned modes are abandoned).  
```
if mode == "trajectory": collect the demo trajectory in one task
if mode == "trajectory_replay": collect the trajectories in other tasks by replaying the actions of the demo trajectory
if mode == "MBRL_learning": training for model-based RL and online running
if mode == "meta_learning": training for our proposed method
if mode == "train_offline_MBRL_model": train the model-based RL offline
if mode == "train_offline_meta_model": train the MAML offline
if mode == "embedding_NN": training for the fast adaptation through meta-learning embeddings (FAMLE)
if mode == "point_cloud_collection_additional": collect the point cloud data for collision predictor
```
To start the program, 
```
roslaunch MBRL_transport start_training.launch
```

### Data collection
We do not provide the data we collected before. To get the data, please run the following steps. The joy node can be found in `CrazyS/rotors_joy_interface/joy_firefly.cpp`. The operation rule of the wired Xbox 360 controller is shown below. The action for x,y, and z is the position deviation between [-0.03m, 0.03m] of the virtual leader. Press button B to close the ROS node to terminate the process.
<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/xbox360.png" width="60%"></div>
<p align="center">The rule of the wired Xbox 360 controller</p>

#### Dynamics model
To collect the data for the dynamics model, we first change the mode into `trajectory`. 
```
 wind_condition_x = 0.0
 L = 0.6
```
change the `wind_condition_x` and the neighbour distance `L` to the above configuration to get the demo trajectory.
Then, we change the mode to `trajectory_replay`. Change the `wind_condition_x` and the `neighbour distance L` to get different datasets (the configurations of the training and testing tasks are shown in the paper). The collection will automatically terminate when 2500 data points are collected. The saving files will be named `firefly_data_3d_wind_x{1}_2agents_L{2}_dt_0.15.mat` where the {1} and {2} are the corresponding conditions. For the tracking of the task with obstacles, the saving files are named `firefly_data_3d_wind_x{1}_2agents_L{2}_dt_0.15_obstacle.mat`. Different from the task without obstacles, we train the meta only in the testing tasks.

#### Collision predictor
To collect the data for the collision predictor, we change the mode to `point_cloud_collection_additional`.<be>
1. Change the world file into `3d_visual_env1` in `mav_with_waypoint_publisher.launch` <br>
2. Change `xy` into 0 in the `MBRL_transport_params.yaml`<br>
3. Uncomment subscribers about cameras in `firefly_env.py` <br>
4. Change the different conditions in `start_training.py`, and also change the conditions in the callback function of  `_point_callback_top` in `firefly_env.py` (for the naming of the point cloud data) <br>
5. We will get data in the `point_clouds_and_configurations_additional` folder, we delete the first 0-209 files for each task as they are bad data before stabilisation.<br>
6. Run `preprocess_command.sh` to get the pre-processed data inside each task of the `point_clouds_and_configurations_additional` folder <br>

### Meta training of dynamics model
In the `start_training.py`, we change the mode to `train_offline_meta_model`. This will give us the offline meta-model. The `load_model_m` in the `MBRL_transport_params.yaml` should be set to 0. The trained model is saved in the `model_3d` folder.<br>

### Training the corrective policy
Then, we change the mode to `meta_learning. Uncomment the commands in `start_training.py`, we will train the corrective policy <br> 
```
exp.run_experiment_meta_without_online(log_path)   # For meta-learning and adaptation without action correction
exp.run_experiment_meta_online1_1(log_path)        # For meta-learning and adaptation with action correction
exp.run_experiment_meta_online1_evaluation(log_path) # Evaluate the model of meta+PPO
```
`load_model_m` in the `MBRL_transport_params.yaml` should be set to 1. We must change `MPC(params,env,params_meta)` with no obs parameter. The trained model is saved in the `PPO_model` folder. <br>
For the scenarios with obstacles, in the `start_training.py`, we need to change `MPC(params,env,params_meta,obs=2)` where `obs` = 1, 2 and 3 correspond to the testing tasks 1,2 and 3. The `obs` will change the cost function of MPC as both the  payload and the UAVs are tracked.

### Baselines of the dynamics model
For **probabilistic ensembles with trajectory sampling (PETS)**, change the mode to `MBRL_learning`. In the following line of `start_training.py`
```
training = 0
exp.run_experiment(training,log_path)
```
For training, `training` is set to 1. `load_model_PETS` in the `MBRL_transport_params.yaml` is set to 0.<br>
For evaluation, `training` is set to 0. `load_model_PETS` in the `MBRL_transport_params.yaml` is set to 1.<be>
The model is saved in `PETS_3d_model` folder
***

For **fast adaptation through meta-learning embedding (FAMLE)**, change the mode to `embedding_NN`. In the following line of `start_training.py`
```
training = 0
exp.run_experiment_embedding_meta(training,logger,path=log_path)
```
For training, `training` is set to 1. For evaluation, `training` is set to 0. The saved is saved in `FAMLE_model`<be>
***

For proximal policy optimisation (PPO), <br>
To do

### Training of the collision predictor
Assume we have got the data in `point_clouds_and_configurations_additional` folder like this structure
```
-point_clouds_and_configurations_additional
            -firefly_points_3d_wind_x0.0_y0.0_2agents_L0.6
                        -preprocess
                        -210.mat
                        ..........
            -firefly_points_3d_wind_x0.3_y0.0_2agents_L1.0
            -firefly_points_3d_wind_x0.5_y0.0_2agents_L0.8
            -firefly_points_3d_wind_x0.6_y0.0_2agents_L1.4
            -firefly_points_3d_wind_x0.8_y0.0_2agents_L1.2
            -firefly_points_3d_wind_x1.0_y0.0_2agents_L0.8
```

1. Run `python3 generate_obs_points.py` to generate `obs.mat` containing normalised obs_points, pos, size. This mat will be used for collision detection.
2. Run `python3 start_siren_main.py`. Notice that we use Pytorch lightning to train the predictor (the Chinese tutorial for Pytorch lightning is [here](https://zhuanlan.zhihu.com/p/592784094)). We need to deal with `occupancy_predictor_2d.py` (defining the siren model and cost function) and `pointcloud_dataset.py` (defining how to get all data points from four tasks and feed them into the batch during the training)
3. In start_siren_main.py, we change `is_predict` to 1 for testing tasks and `is_predict` to 0 for training tasks. The model will be saved in `/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_{1}/lightning_logs/version_0/checkpoints` where {1} is the random seed you set. We also provide a code to visualise the results of the model we get by running `start_siren_visualisation.py`. The `matplotlib` will be used to plot the predicted sdf and the ground truth sdf.

### Running RRT with the collision predictor
This collision predictor is utilised to bias the tree-growth process of rapidly-exploring random tree (RRT) algorithm towards the goal points with a collision-free constraint.<br>

In this paper, we consider two scenarios, the **cross path** and the **square path**.<br>  

Firstly, we get the original full paths for different scenarios and tasks by changing `route_name` and `task_num` in `generate_route_points.py`. The original paths are named `save_waypoints_collision_cross_0.mat` or `save_waypoints_collision_square_0.mat`.<br>
Then, we change the configuration in `RRT-svmrm.py` to generate collision-free paths.
```
task_num = 2
rn = "square_c" #square_c,cross
```
The path will be saved in `save_corrective_waypoints_collision_cross_0.mat` after the visualisation process. To validate the collision-free paths, change the `route` in `MBRL_transport_params.yaml` for `square_xy` and `cross`. 

### License
This repository is released under the MIT license. See [LICENSE](https://github.com/wawachen/Meta-load-fly/blob/main/LICENSE) for additional details.
