# Meta-Load Fly: a Reinforcement Learning Framework for Adaptive Load Tracking with Collision Prediction
[Jingyu Chen](https://www.researchgate.net/profile/Jingyu-Chen-20) <br>   
The University of Sheffield

[Project website]() | [Paper]()

## Overview of the Meta-load-fly
![](https://github.com/wawachen/Meta-load-fly/blob/main/image/method_icra(1).png)
<p align="center">Overview of Meta-Load Fly with load trajectory tracking and path planning; A. Path planning B. Corrective policy</p>


## Introduction 
Our code mainly consists of *CrazyS*, *MBRL_transport* and *openai_ros* along with the *mav_comm* and *gemotry_tf2* for python3 compiling.
It has been tested in Ubuntu 18.04 Melodic ROS and Ubuntu 20.04 Neotic. The code structure is shown below.

<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/github_1.png" width="60%"></div>
<p align="center">The relationship between different packages</p>

**CrazyS**: Provide the model of the cable-suspended Firefly-load system and the low-based tracking controller. The joy plugin and wind plugin are modified.<br>
**MBRL_transport**：This is the main code where we contribute to our meta-load-fly framework consisting of the adaptive load trajectory tracking module and the collision predictor.<br>
**Openai_ros**: A bridge for connecting Gazebo with pytorch. We also modify it here for building the task environment for transport. The basic movements of Firefly are defined here.<br>

## Install Dependencies
Firstly, create a ROS workspace (the tutorials can be found [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)). Create an empty package **MBRL_transport** and a src folder inside it. Copy the content of this repository into the src folder of  **MBRL_transport** package. Then, the config and launch folder in this repository should be moved outside of this src folder to form a complete ROS package. The *mav_comm* and *gemotry_tf2* ROS packages need to be installed.<br>

To install **CrazyS** and **Openai_ros**, please download our forked repositories

```
git clone git@github.com:wawachen/CrazyS.git<br>
git clone git@github.com:wawachen/openai_ros.git
```

If you are using Ubuntu 18.04 Melodic ROS, the tricky thing is that we will use Python3 in Melodic ROS whose default Python is 2.7.
Thus, when we import these packages into the Catkin workspace, we use the following command to indicate for ROS that we are using python3 not python2 for compiling files.

```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Then, for the **openai_ros** package, we will use the tf package which was originally designed for python2. Thus we have to download the source file from [here](https://github.com/ros/geometry2) and compile it in the Python3 environment using the following command,

```
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

Another bug will take place in the gazebo_plugin from **CrazyS** <br>
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
if mode == "embedding_N: training for the fast adaptation through meta-learning embeddings (FAMLE)
if mode == "point_cloud_collection_additional": collect the point cloud data for collision predictor
```
To start the program, 
```
roslaunch MBRL_transport start_training.launch
```

### Data collection
We do not provide the data we collected before. To get the data, please run the following steps. The joy node can be found in CrazyS/rotors_joy_interface/joy_firefly.cpp. The operation rule of the wired Xbox 360 controller is shown below. The action for x,y, and z is the position deviation between [-0.03m, 0.03m] of the virtual leader. Press button B to close the ROS node to terminate the process.
<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/xbox360.png" width="60%"></div>
<p align="center">The rule of the wired Xbox 360 controller</p>

#### Dynamics model
To collect the data for the dynamics model, we first change the mode into *trajectory*. 
```
 wind_condition_x = 0.0
 L = 0.6
```
change the **wind_condition_x** and the **neighbour distance L** to the above configuration to get the demo trajectory.
Then, we change the mode to *trajectory_replay*. Change the **wind_condition_x** and the **neighbour distance L** to get different datasets (the configurations of the training and testing tasks are shown in the paper). The collection will automatically terminate when 2500 data points are collected. The saving files will be named *firefly_data_3d_wind_x{1}_2agents_L{2}_dt_0.15.mat* where the {1} and {2} are the corresponding conditions. For the tracking of the task with obstacles, the saving files are named *firefly_data_3d_wind_x{1}_2agents_L{2}_dt_0.15_obstacle.mat*. Different from the task without obstacles, we train the meta only in the testing tasks.

#### Collision predictor
To collect the data for the collision predictor, we change the mode to *point_cloud_collection_additional*.<be>
1. Change world file into *3d_visual_env1* in mav_with_waypoint_publisher.launch <br>
2. Change xy into 0 <br>
3. Uncomment subscribers about cameras in firefly_env.py <br>
4. Change the different conditions in start_training.py, and also change the conditions in the callback function of  _point_callback_top in firefly_env.py (for the naming of the point cloud data) <br>
5. We will get data in the *point_clouds_and_configurations_additional* folder, we delete the first 0-209 files for each task as they are bad data before stabilisation.<br>
6. Run preprocess_command.sh to get the pre-processed data inside each task of the *point_clouds_and_configurations_additional* folder <br>

### Meta training of dynamics model
Change the mode into *meta_learning*. In the start_training.py,uncomment the commands <br> 
```
exp.run_experiment_meta_without_online(log_path)   # For meta-learning and adaptation without action correction
exp.run_experiment_meta_online1_1(log_path)        # For meta-learning and adaptation with action correction
exp.run_experiment_meta_online1_evaluation(log_path) # Evaluate the model of meta+PPO
```
For the evaluation, *load_model_m* in the MBRL_transport_params.yaml should be set to 1 otherwise 0. The trained model is save in the model_3d

### Baselines of the dynamics model
For probabilistic ensembles with trajectory sampling (PETS), change the mode to MBRL_learning.
```
training = 0
exp.run_experiment(training,log_path)
```
For training, *training* is set to 1. *load_model_PETS* in the MBRL_transport_params.yaml is set to 0.
For evaluation, *training* is set to 0. *load_model_PETS* in the MBRL_transport_params.yaml is set to 1.


### Training of the collision predictor
python3 generate_obs_points.py to generate obs.mat containing normalised obs_points,pos,size
python3 start_siren.py notice that we use pytorch lightning to train the predictor. We need to deal with occupancy_predictor_2d.py(define the siren model and cost function) and pointcloud_dataset.py(define how to get all data points from four tasks and feed them into the batch during the training)
Change mode to train,run start_siren.py, we get model checkpoint
Change mode to evaluation, change type into val_data and test_data respectively to get the corresponding prediction and error for validation and testing


### License
This repository is released under the MIT license. See [LICENSE](https://github.com/wawachen/Meta-load-fly/blob/main/LICENSE) for additional details.
