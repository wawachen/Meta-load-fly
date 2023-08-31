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

![here](https://github.com/wawachen/Meta-load-fly/blob/main/image/github_1.png)
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

### Data collection
To do

### Meta training of dynamics model
To do

### Training of the collision predictor
To do

### License
This repository is released under the MIT license. See [LICENSE](https://github.com/wawachen/Meta-load-fly/blob/main/LICENSE) for additional details.
