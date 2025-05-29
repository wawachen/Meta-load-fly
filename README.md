# A meta-reinforcement learning method for adaptive payload transportation with variations
[Jingyu Chen](https://www.researchgate.net/profile/Jingyu-Chen-20) <br>
The University of Sheffield

[Project website](https://sites.google.com/view/meta-payload-fly/) | [Paper](https://www.sciencedirect.com/science/article/pii/S0925231225007040?dgcid=author)

## Overview of the Meta-load-fly
![](https://github.com/wawachen/Meta-load-fly/blob/main/image/method_icra(1).png)
<p align="center">Overview of Meta-Load Fly with load trajectory tracking and path planning; A. Path planning B. Corrective policy</p>
<table>
  <tr>
    <td><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/fly.gif" style="width: 100%;"></td>
    <td><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/adaptive.gif" style="width: 100%;"></td>
  </tr>
</table>

## 🚀 Introduction 
### Simulation environments
In this work, we consider the obstacle-free scenarios for payload tracking and obstacle scenarios for full uav-payload system tracking.
<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/All_envs.png" width="60%"></div>
<p align="center">The considered obstacle and obstacle-free environments</p>

### Code structure
Our code mainly consists of `CrazyS`, `MBRL_transport` and `openai_ros` along with the `mav_comm` and `gemotry_tf2` for python3 compiling.
It has been tested in Ubuntu 20.04 Neotic. The code structure is shown below.

<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/github_1.png" width="60%"></div>
<p align="center">The relationship between different packages</p>

**CrazyS**: Provide the model of the cable-suspended Firefly-load system and the low-based tracking controller. The joy plugin and wind plugin are modified.<br>
**MBRL_transport**：This is the main code where we contribute to our meta-load-fly framework consisting of the adaptive load trajectory tracking module and the collision predictor.<br>
**Openai_ros**: A bridge for connecting Gazebo with Pytorch. We also modify it here for building the task environment for transport. The basic movements of Firefly are defined here.<br>

## 🚀 Installation
Firstly, create a ROS workspace (the tutorials can be found [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)). Create an empty package `MBRL_transport` and `cd MBRL_transport`. Then clone the repository `git clone xx`. For `all_data`, `CrazyS`, `mav_comm` , `gemotry_tf2-noeric-devel` and `openai_ros` ROS packages, download them from [here](https://www.dropbox.com/scl/fo/dgz6au6wzdcy8ic11hj3n/AGTT0IKwSIKb_8HdGD10MpY?rlkey=kpl32nvalvu2ej39o5nyp198t&st=5mo2ugl8&dl=0). For `checkpoints`, download them from [here](https://www.jianguoyun.com/p/DcgO-eQQ0Ou7DRi5w_sFIAA). Put all of them in the root folder of `MBRL_transport`.<br>

If you are using Ubuntu 18.04 Melodic ROS, the tricky thing is that we will use Python3 in Melodic ROS whose default Python is 2.7.
Thus, when we import these packages into the Catkin workspace, we use the following command to indicate for ROS that we are using python3 not python2 for compiling files.

```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
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
## 🧠 Usage
To start the program, 
```
roslaunch MBRL_transport start_training.launch config_file:=xx.yaml
```
In the start_training.py, it defines different modes. The configuration files are defined in the folder config/policy and they control the experiment parameters and the task conditions.  

```
collect_dynamics_params.yaml: collect the demo trajectory in one task
replay_dynamics_params.yaml: collect the trajectories in other tasks by following the actions of the demo
MBRL_params.yaml: train the model-based RL and online running
Meta_params.yaml: train our proposed method in one condition
Meta_params1.yaml: train our proposed method in all conditions
Meta_params2.yaml: train only MAML in one condition
Meta_params3.yaml: train our proposed method for full UAV-load system tracking in all conditions
offline_Meta_params.yaml: train the MAML offline
ppo_params.yaml: train the model-free RL algorithm PPO
FAMLE_params.yaml: train FAMLE method
collect_pointcloud_params.yaml: collect the point cloud data for collision predictor
```
The other configuration file is defined in `MBRL_transport_params.yaml` and it controls the tracking routes, the environment as well as the interface for pointcloud collection.

### Data collection
To get the data, please run the following steps. The joy node can be found in `CrazyS/rotors_joy_interface/joy_firefly.cpp`. The operation rule of the wired Xbox 360 controller is shown below. The action for x,y, and z is the position deviation between [-0.03m, 0.03m] of the virtual leader. Press button B to close the ROS node to terminate the process.
<div align=center><img src="https://github.com/wawachen/Meta-load-fly/blob/main/image/xbox360.png" width="60%"></div>
<p align="center">The rule of the wired Xbox 360 controller</p>
We provide the collected data in `all_data/dynamics`.

### Dynamics model
To collect the data for the dynamics model, run `roslaunch MBRL_transport start_training.launch config_file:=collect_dynamics_params.yaml`. 
```
 wind_condition_x = 0.0
 L = 0.6
```
change the `wind_condition_x` and the neighbour distance `L` to the above configuration to get the demo trajectory.
Then, we run `roslaunch MBRL_transport start_training.launch config_file:=collect_dynamics_params.yaml`. Change the `wind_condition_x` and the `neighbour distance L` to get different datasets (the configurations of the training and testing tasks are shown in the paper). The collection will automatically terminate when 2500 data points are collected. The saving files will be named `firefly_data_3d_wind_x{1}_2agents_L{2}_dt_0.15.mat` where the {1} and {2} are the corresponding conditions. 

### Collision predictor
To collect the data for the collision predictor, we run `roslaunch MBRL_transport start_training.launch config_file:=collect_pointcloud_params.yaml`.<be>
In `MBRL_transport_params.yaml`, change the content.<br>
```
route: pointcloud
save_pointcloud_path: xxx
save_pointcloud: True
```
Run `preprocess_command.sh` to get the pre-processed data inside each task of the `train_point_clouds` folder <br>

### 🏋️ Meta training of dynamics model
Run `roslaunch MBRL_transport start_training.launch config_file:=offline_Meta_params.yaml`<br>

### 🏋️ Training the corrective policy
Run `roslaunch MBRL_transport start_training.launch config_file:=Meta_params.yaml` <br> 

### 🏋️ Training the Baselines of the dynamics model
For **probabilistic ensembles with trajectory sampling (PETS)**, run `roslaunch MBRL_transport start_training.launch config_file:=MBRL_params.yaml`<br> 
For **fast adaptation through meta-learning embedding (FAMLE)**, run `roslaunch MBRL_transport start_training.launch config_file:=FAMLE_params.yaml`<br> 
For proximal policy optimisation (PPO),  run `roslaunch MBRL_transport start_training.launch config_file:=ppo_params.yaml` <br>

### 🏋️ Training of the collision predictor
Assume we have got the data in `train_point_clouds` folder like this structure
```
-train_point_clouds
            -wind_x0.0_y0.0_2agents_L0.6
                        -preprocess
                        -210.mat
                        ..........
            -wind_x0.3_y0.0_2agents_L1.0
            -wind_x0.5_y0.0_2agents_L0.8
            -wind_x0.6_y0.0_2agents_L1.4
            -wind_x0.8_y0.0_2agents_L1.2
            -wind_x1.0_y0.0_2agents_L0.8
```

1. Run `python3 generate_obs_points.py` to generate `obs.mat` containing normalised obs_points, pos, size. This mat will be used for collision detection.
2. Run `python3 train_siren_main.py`. Notice that we use Pytorch lightning to train the predictor (the Chinese tutorial for Pytorch lightning is [here](https://zhuanlan.zhihu.com/p/592784094)). We need to deal with `occupancy_predictor_2d.py` (defining the siren model and cost function) and `pointcloud_dataset.py` (defining how to get all data points from four tasks and feed them into the batch during the training)
3. In train_siren_main.py, we change `is_predict` to 1 for testing tasks and `is_predict` to 0 for training tasks. The model will be saved in `/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_{1}/lightning_logs/version_0/checkpoints` where {1} is the random seed you set. We also provide a code to visualise the results of the model we get by running `start_siren_visualisation.py`. The `matplotlib` will be used to plot the predicted sdf and the ground truth sdf.

### 📊 Running RRT with the collision predictor
This collision predictor is utilised to bias the tree-growth process of rapidly-exploring random tree (RRT) algorithm towards the goal points with a collision-free constraint.<br>

In this paper, we consider four scenarios, the **cross path**, **square path**, **crowd1** and **crowd2**.<br>

Firstly, we get the original full paths for different scenarios and tasks by changing `route_name` and `task_num` in `generate_route_points.py`. The original paths are named `save_waypoints_collision_cross_0.mat` or `save_waypoints_collision_square_0.mat`.<br>
Then, we change the configuration in `RRT-svmrm.py` to generate collision-free paths.
```
task_num = 2
rn = "square_c" #square_c,cross
```
The path will be saved in `save_corrective_waypoints_collision_cross_0.mat` after the visualisation process. To validate the collision-free paths, change the `route` and `load_traj_path` of `MBRL_transport_params.yaml`. Run `roslaunch MBRL_transport start_training.launch config_file:=Meta_params3.yaml`. `load_traj_path` is the location of the collision-free path generated by our proposed path planner.

## 📄 Citation
```
@article{chen2025meta,
  title={A meta-reinforcement learning method for adaptive payload transportation with variations},
  author={Chen, Jingyu and Ma, Ruidong and Xu, Meng and Candan, Fethi and Mihaylova, Lyudmila and Oyekan, John},
  journal={Neurocomputing},
  volume={638},
  pages={130032},
  year={2025},
  publisher={Elsevier}
}
```

## License
This repository is released under the MIT license. See [LICENSE](https://github.com/wawachen/Meta-load-fly/blob/main/LICENSE) for additional details.
