#!/usr/bin/env python3

import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import torch
import numpy as np
from common.MPC import MPC
import random
import os
from tensorboardX import SummaryWriter
import scipy.io as sio
from omegaconf import OmegaConf
from models.model import nn_constructor, meta_nn_constructor
import pathlib
import datetime
from dataset.windNShot import WindNShot
from common.train import offline_train_meta,offline_test_meta
from common.Agent import Agent
from models.model import embedding_nn_constructor
from common.train import embedding_meta_train
from baselines.FAMLE import famle
from scipy.io import savemat
import scipy.io as sio

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # tf.random.set_seed()

if __name__ == '__main__':
    rospy.init_node('MBRL_firefly_transport',
                    anonymous=True, log_level=rospy.WARN)

    config_file = rospy.get_param('~config_file')
    config_path = str(pathlib.Path(__file__).parent.parent.joinpath('config','policy'))+"/"+config_file
    cfg = OmegaConf.load(config_path)

    now = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/firefly/task_and_robot_environment_name')
    if cfg.mode == "offline_Meta" or (cfg.mode == "FAMLE" and cfg.is_eval==0):
        rospy.loginfo("The environment will not be initialized")
    else:
        env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
        # Create the Gym environment
        rospy.loginfo("Gym environment done")
        rospy.loginfo("Starting Learning")

    # Modes: collect_trajectory, follow_open_trajectory, MBRL_learning, evaluation_dynamics,meta_learning

    try:
        if cfg.mode == "dynamics":
            #task 1: wind 0.0 L 0.6
            wind_condition_x = cfg.wind_condition_x
            wind_condition_y = cfg.wind_condition_y #y is always zero, we only consider wind in x axis
            L = cfg.L

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)
            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))

                # Initialize the environment and get first state of the robot
                observation, _ ,_= env.reset()

                A1 = []
                O, A = [observation], []
                O1 = []
                O2 = []

                #######################################
                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    rospy.logwarn("############### data points available =>" + str(i))
                    # Pick an action based on the current state
                    
                    action, action1, obs, obs1, obs2 = env.step_pos(L)
                    A1.append(action1) #for replay goal
                    A.append(action) #real displacement
                    O.append(obs)
                    O1.append(obs1)
                    O2.append(obs2)

                    #raw_input("Next Step...PRESS KEY")
                    # rospy.sleep(2.0)
                    i += 1
                    if i == 2500:
                        break
            
            if not os.path.exists(cfg.save_path):
                os.makedirs(cfg.save_path)  

            fileName = cfg.save_path+"/firefly_data_3d"

            fileName += "_wind" + "_x"+str(wind_condition_x)

            fileName += "_" + str(2) + "agents"+"_"+"L"+str(L)+"_"+"dt_"+str(0.15)

            fileName += ".mat"

            mdic = {"acs": A, "acs1": A1, "obs": O, "obs1":O1, "obs2":O2}

            if cfg.is_save:
                savemat(fileName, mdic)
            print("finish saving file")
        
            env.close()

        if cfg.mode == "dynamics_replay":
            #Total variations
            #task 1: wind 0.0 L 0.6  
            # task 2: wind 0.3 L 1.0 
            # task 3: wind 0.5 L 0.8 
            # task 4: wind 0.8 L 1.2

            # test task 1: wind 1.0 L 0.8
            # test task 2: wind 0.6 L 1.4
            wind_condition_x = cfg.wind_condition_x
            wind_condition_y = cfg.wind_condition_y #no use
            L = cfg.L

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)

            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))

                #Reload the action goals given by the joystick
                mat_contents = sio.loadmat(cfg.load_path)

                joy_waypoints = mat_contents['acs1']
                joy_waypoints[:,0] = joy_waypoints[:,0]*env.max_x
                joy_waypoints[:,1] = joy_waypoints[:,1]*env.max_y
                joy_waypoints[:,2] = joy_waypoints[:,2]*env.max_z
                # Initialize the environment and get first state of the robot
                observation, _,_ = env.reset()

                O, A = [observation], []
                O1 = []
                O2 = []
                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    rospy.logwarn("############### data points available =>" + str(i))
                    # Pick an action based on the current state
                    action, obs, obs1, obs2 = env.step_pos_replay(joy_waypoints[i,:],L)
                    
                    A.append(action)
                    O.append(obs)
                    O1.append(obs1)
                    O2.append(obs2)

                    i += 1
                    if i == 2500:
                        break

            fileName = cfg.save_path

            #wind speed: 0.0, 0.3, 0.5, 0.8
            fileName += "_wind"+ "_x"+str(wind_condition_x) 

            fileName += "_" + str(2) + "agents"+"_"+"L"+str(L)+"_"+"dt_"+str(0.15)

            fileName += ".mat"

            mdic = {"acs": A, "obs": O, "obs1": O1, "obs2": O2}

            if cfg.is_save:
                savemat(fileName, mdic)
            print("finish saving file")
        
            env.close()

        if cfg.mode == "MBRL":
            main_path = str(pathlib.Path(__file__).parent.parent.joinpath('experiments','outputs'))
            cfg.log_path = f'{main_path}/{cfg.exp_name}/{cfg.exp_name}_{now}'
            os.makedirs(cfg.log_path, exist_ok=True)

            logger = SummaryWriter(logdir=cfg.log_path) # used for tensorboard

            #task 1: wind 0.0 L 0.6  
            # task 2: wind 0.3 L 1.0 
            # task 3: wind 0.5 L 0.8 
            # task 4: wind 0.8 L 1.2
            # test task 1: wind 1.0 L 0.8
            # test task 2: wind 0.6 L 1.4
            from MBExperiment_MBRL import MBExperiment
            
            model = nn_constructor(cfg.model_params,cfg.log_path)
            policy = MPC(cfg.mpc_params, cfg.mode, env, model)
            exp = MBExperiment(env,policy,logger)
            exp.run_experiment(cfg.exp_params)
            env.close()

        if cfg.mode == "Meta":
            main_path = str(pathlib.Path(__file__).parent.parent.joinpath('experiments','outputs'))
            cfg.log_path = f'{main_path}/{cfg.exp_name}/{cfg.exp_name}_{now}'
            os.makedirs(cfg.log_path, exist_ok=True)
            
            logger = SummaryWriter(logdir=cfg.log_path) # used for tensorboard

            model = meta_nn_constructor(cfg.model_params, cfg.log_path)

            from MBExperiment_meta import MBExperiment
            policy = MPC(cfg.mpc_params, cfg.mode, env, model)
            agent = Agent(env)
            exp = MBExperiment(env,policy,agent,logger,cfg.log_path)

            if cfg.is_eval:
                if cfg.exp_name == "train_Meta_single_condition" or cfg.exp_name == "train_Meta_all_conditions":
                    exp.run_experiment_meta_online1_evaluation(cfg.exp_params,cfg.is_eval)
                if cfg.exp_name == "train_Meta_full_system":
                    exp.run_experiment_meta_online1_evaluation_full_uav(cfg.exp_params,cfg.is_eval)
            else:
                if cfg.exp_name == "train_Meta_single_condition":
                    exp.run_experiment_meta_online1_1(cfg.exp_params, cfg.is_eval)
                if cfg.exp_name == "train_Meta_all_conditions":
                    exp.run_experiment_meta_online1_1all(cfg.exp_params, cfg.is_eval)
                if cfg.exp_name == "train_MAML":
                    exp.run_experiment_meta_without_online(cfg.exp_params)
                if cfg.exp_name == "train_Meta_full_system":
                    exp.run_experiment_meta_online1_1all_full_uav(cfg.exp_params,cfg.is_eval)
            
            env.close()

        if cfg.mode == "offline_Meta":
            main_path = str(pathlib.Path(__file__).parent.parent.joinpath('experiments','outputs'))
            cfg.log_path = f'{main_path}/{cfg.exp_name}/{cfg.exp_name}_{now}'
            os.makedirs(cfg.log_path, exist_ok=True)
            logger = SummaryWriter(logdir=cfg.log_path) # used for tensorboard

            # db_train = WindNShot(50, cfg.meta_task_num, 501, cfg.n_way, cfg.k_spt, cfg.k_qry, integrated=cfg.inter, sequential=True)
            db_train = WindNShot(cfg.batch_num, cfg.meta_task_num, cfg.path_length, cfg.n_way, cfg.k_spt, cfg.k_qry, integrated=False, sequential=True)
            model = meta_nn_constructor(cfg.model_params, cfg.log_path)

            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
            np.random.seed(cfg.seed)

            if not cfg.is_eval:
                offline_train_meta(model, db_train, cfg, logger) 
            else:
                offline_test_meta(model, db_train)

        if cfg.mode == "FAMLE":
            main_path = str(pathlib.Path(__file__).parent.parent.joinpath('experiments','outputs'))
            cfg.log_path = f'{main_path}/{cfg.exp_name}/{cfg.exp_name}_{now}'
            os.makedirs(cfg.log_path, exist_ok=True)
            logger = SummaryWriter(logdir=cfg.log_path) # used for tensorboard

            if not cfg.is_eval:
                model = embedding_nn_constructor(cfg.model_params, cfg.exp_params.seed)
                embedding_meta_train(model, cfg.exp_params, logger, cfg.log_path)
            else:
                # seed 222,50,8,1,20,45,60,104,165,200
                torch.manual_seed(cfg.exp_params.seed)
                torch.cuda.manual_seed_all(cfg.exp_params.seed)
                np.random.seed(cfg.exp_params.seed)

                agent = Agent(env,meta=True)
                model = famle.load_model(cfg.model_params.load_model_path+"/model.pt",device=torch.device('cuda'))
                policy = MPC(cfg.mpc_params, cfg.mode, env, model)
            
                sample = agent.embedding_sample(cfg.exp_params.task_hor, policy, cfg.exp_params.wind_condition_x, cfg.exp_params.wind_condition_y, cfg.exp_params.L, log_data=cfg.exp_params.log_sample_data, data_path=cfg.log_path)
        
            if cfg.exp_params.log_sample_data:
                print("start logging")
                savemat(cfg.path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
                savemat(cfg.path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
                savemat(cfg.path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
                savemat(cfg.path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
                savemat(cfg.path+'/storeObs.mat', mdict={'arr': sample["obs"]})
                data = sio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
                savemat(cfg.path+'/store_destraj.mat', mdict={'arr': data['arr']})

            env.close()

        if cfg.mode == "ppo":
            main_path = str(pathlib.Path(__file__).parent.parent.joinpath('experiments','outputs'))
            cfg.log_path = f'{main_path}/{cfg.exp_name}/{cfg.exp_name}_{now}'
            os.makedirs(cfg.log_path, exist_ok=True)
            logger = SummaryWriter(logdir=cfg.log_path) # used for tensorboard

            from MBExperiment_ppo import MBExperiment
            exp = MBExperiment(env,logger,cfg.log_path)

            # os.makedirs(exp.logdir)
            if cfg.is_eval:
                exp.run_ppo_evaluation(cfg,cfg.is_eval)
            else:
                exp.run_experiment_ppo(cfg,cfg.is_eval)
                # exp.run_ppo_evaluation(log_path)
            
            env.close()


        if cfg.mode == "pointcloud":
            #used to collect 3d point clouds to fit a 2d signed distance function (sdf) by a Kinect camera
            #Total variations
            #task 1: wind 0.0 L 0.6  
            # task 2: wind 0.3 L 1.0 
            # task 3: wind 0.5 L 0.8 
            # task 4: wind 0.8 L 1.2
            # test task 1: wind 1.0 L 0.8
            # test task 2: wind 0.6 L 1.4

            wind_condition_x = cfg.wind_condition_x
            wind_condition_y = cfg.wind_condition_y 
            L = cfg.L

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)

            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))

                #Reload the action goals given by the joystick
                mat_contents = sio.loadmat(cfg.load_path)
                joy_waypoints = mat_contents['acs1']
                
                joy_waypoints[:,0] = joy_waypoints[:,0]*env.max_x
                joy_waypoints[:,1] = joy_waypoints[:,1]*env.max_y
                joy_waypoints[:,2] = joy_waypoints[:,2]*env.max_z
                # Initialize the environment and get first state of the robot
                observation, _,_ = env.reset()
                if rospy.get_param("/firefly/save_pointcloud"):
                    env.set_pos_callback_cloud_loop()
                #######################################
                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    rospy.logwarn("############### data points available =>" + str(i))
                    # Pick an action based on the current state
                    action, obs, obs1, obs2 = env.step_pos_replay(joy_waypoints[i,:],L)
                    if rospy.get_param("/firefly/save_pointcloud"):
                        env.set_pos_callback_cloud_loop()
                    
                    i += 1
                    if i == 100:
                        break

            print("finish saving file")
        
            env.close()
        
    except KeyboardInterrupt:
        env.close()

