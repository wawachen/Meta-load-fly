#!/usr/bin/env python3

# from gym import wrappers
# ROS packages required
# import scipy as sp
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import torch
import numpy as np
from scipy.io import savemat
from MBExperiment import MBExperiment
from MPC import MPC
import random
import tf
import os
from tensorboardX import SummaryWriter
from time import localtime, strftime
import scipy.io as sio
import glob
from dotmap import DotMap
from pytorch_lightning import seed_everything
from occupancy_predictor_2d import Predictor_Model_2d
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid

TORCH_DEVICE = torch.device('cuda')

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
    mode = "trajectory_replay"
    rospy.init_node('MBRL_firefly_transport',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/firefly/task_and_robot_environment_name')
    if mode == "evaluation_model":
        rospy.loginfo("The environment will not be initialized")
    else:
        env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
        # Create the Gym environment
        rospy.loginfo("Gym environment done")
        rospy.loginfo("Starting Learning")
    # Modes: collect_trajectory, follow_open_trajectory, MBRL_learning, evaluation_dynamics,meta_learning

    try:
        if mode == "trajectory":
            #task 1: wind 0.0 L 0.6
            wind_condition_x = 0.8
            wind_condition_y = 0.0 #y is always zero, we only consider wind in x axis
            L = 1.2

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)
            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))
                # space_3d = rospy.get_param("/firefly/3d_space")

                # Initialize the environment and get first state of the robot
                observation, _ ,_= env.reset()

                # Show on screen the actual situation of the robot
                # env.render()
                # for each episode, we test the robot for nsteps
                # pos_list = [[3.0,0.0,1.0,0.0,5.0],[3.0,0.0,0.5,0.0,2.0],[0.0,0.0,0.5,0.0,5.0],[0.0,0.0,1.0,0.0,2.0]]
                # with open("/home/wawa/catkin_RL_ws/src/MBRL_transport/config/square1.txt","r") as data:
                #     traj_string = [line.split() for line in data]

                # pos_list = [list(map(float, traj_p)) for traj_p in traj_string]
                # print(pos_list)
                ########################################
                A1 = []
                O, A = [observation], []
                O1 = []
                O2 = []
                #     A.append(policy.act(O[t], t))
                #     O.append(obs)

                # print("Rollout length: ", len(A))
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

            fileName = "/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d"

            fileName += "_wind" + "_x"+str(wind_condition_x)

            fileName += "_" + str(2) + "agents"+"_"+"L"+str(L)+"_"+"dt_"+str(0.15)

            fileName += "_obstacle.mat"

            mdic = {"acs": A, "acs1": A1, "obs": O, "obs1":O1, "obs2":O2}

            # savemat(fileName, mdic)
            print("finish saving file")
        
            env.close()

        if mode == "trajectory_replay":
            #Total variations
           #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4

            wind_condition_x = 0.6
            wind_condition_y = 0.0 #no use
            L = 1.4

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)

            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))

                #Reload the action goals given by the joystick
                # mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15.mat")
                mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15_obstacle.mat")

                joy_waypoints = mat_contents['acs1']
                joy_waypoints[:,0] = joy_waypoints[:,0]*env.max_x
                joy_waypoints[:,1] = joy_waypoints[:,1]*env.max_y
                joy_waypoints[:,2] = joy_waypoints[:,2]*env.max_z
                # Initialize the environment and get first state of the robot
                observation, _,_ = env.reset()

                # Show on screen the actual situation of the robot
                # env.render()
                # for each episode, we test the robot for nsteps
                # pos_list = [[3.0,0.0,1.0,0.0,5.0],[3.0,0.0,0.5,0.0,2.0],[0.0,0.0,0.5,0.0,5.0],[0.0,0.0,1.0,0.0,2.0]]
                # with open("/home/wawa/catkin_RL_ws/src/MBRL_transport/config/square1.txt","r") as data:
                #     traj_string = [line.split() for line in data]

                # pos_list = [list(map(float, traj_p)) for traj_p in traj_string]
                # print(pos_list)
                ########################################
                O, A = [observation], []
                O1 = []
                O2 = []
                #     A.append(policy.act(O[t], t))
                #     O.append(obs)

                # print("Rollout length: ", len(A))
                #######################################
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

                    #raw_input("Next Step...PRESS KEY")
                    # rospy.sleep(2.0)
                    i += 1
                    if i == 2500:
                        break

            fileName = "/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d"

            #wind speed: 0.0, 0.3, 0.5, 0.8
            fileName += "_wind"+ "_x"+str(wind_condition_x) 

            fileName += "_" + str(2) + "agents"+"_"+"L"+str(L)+"_"+"dt_"+str(0.15)

            fileName += "_obstacle.mat"

            mdic = {"acs": A, "obs": O, "obs1":O1, "obs2":O2}
    
            # savemat(fileName, mdic)
            print("finish saving file")
        
            env.close()

        if mode == "MBRL_learning":

            log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_3d_MBRL_evaluation_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))

            os.makedirs(log_path, exist_ok=True)
            logger = SummaryWriter(logdir=log_path) # used for tensorboard
            # #MBRL TRAINING 
            # set_global_seeds(0)
            # assert ctrl_type == 'MPC'
            params = DotMap()
            params.per = rospy.get_param("/firefly/per")
            params.prop_mode = rospy.get_param("/firefly/prop_mode")
            params.opt_mode = rospy.get_param("/firefly/opt_mode")
            params.npart = rospy.get_param("/firefly/npart")
            params.ign_var = rospy.get_param("/firefly/ign_var") 
            params.mode = rospy.get_param("/firefly/opt_mode")
            params.plan_hor = rospy.get_param("/firefly/plan_hor")
            params.num_nets = rospy.get_param("/firefly/num_nets")
            params.epsilon = rospy.get_param("/firefly/epsilon")
            params.alpha = rospy.get_param("/firefly/alpha")
            params.epochs = rospy.get_param("/firefly/epochs")
            params.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params.route = rospy.get_param("/firefly/route")
            params.popsize = rospy.get_param("/firefly/popsize")
            params.max_iters = rospy.get_param("/firefly/max_iters")
            params.num_elites = rospy.get_param("/firefly/num_elites")
            params.visualization = rospy.get_param("/firefly/visualization")
            params.load_model = rospy.get_param("/firefly/load_model_PETS")

            params1 = DotMap()
            params1.ntrain_iters = rospy.get_param("/firefly/ntrain_iters")
            params1.nrollouts_per_iter = rospy.get_param("/firefly/nrollouts_per_iter")
            params1.ninit_rollouts = rospy.get_param("/firefly/ninit_rollouts")
            params1.nrecord = rospy.get_param("/firefly/nrecord")
            params1.neval = rospy.get_param("/firefly/neval")
            params1.task_hor = rospy.get_param("/firefly/task_hor")
            params1.load_model = rospy.get_param("/firefly/load_model_PETS")
            params1.log_sample_data = rospy.get_param("/firefly/log_sample_data")

            policy = MPC(params,env)
            exp = MBExperiment(params1,env,policy,logger)

            # os.makedirs(exp.logdir)
            training = 0
            exp.run_experiment(training,log_path)
            env.close()

        if mode == "meta_learning":
            log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_3d_meta_evaluation_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
            os.makedirs(log_path, exist_ok=True)
            logger = SummaryWriter(logdir=log_path) # used for tensorboard

            # assert ctrl_type == 'MPC'
            params = DotMap()
            params.per = rospy.get_param("/firefly/per")
            params.prop_mode = rospy.get_param("/firefly/prop_mode")
            params.opt_mode = rospy.get_param("/firefly/opt_mode")
            params.npart = rospy.get_param("/firefly/npart")
            params.ign_var = rospy.get_param("/firefly/ign_var") 
            params.mode = rospy.get_param("/firefly/opt_mode")
            params.plan_hor = rospy.get_param("/firefly/plan_hor")
            params.num_nets = rospy.get_param("/firefly/num_nets")
            params.epsilon = rospy.get_param("/firefly/epsilon")
            params.alpha = rospy.get_param("/firefly/alpha")
            params.epochs = rospy.get_param("/firefly/epochs")
            # params.model_in = rospy.get_param("/firefly/model_in")
            # params.model_out = rospy.get_param("/firefly/model_out")
            params.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params.route = rospy.get_param("/firefly/route")
            params.popsize = rospy.get_param("/firefly/popsize")
            params.max_iters = rospy.get_param("/firefly/max_iters")
            params.num_elites = rospy.get_param("/firefly/num_elites")
            params.visualization = rospy.get_param("/firefly/visualization")
            test_model = rospy.get_param("/firefly/test_model")

            params1 = DotMap()
            params1.task_hor = rospy.get_param("/firefly/task_hor")
            params1.meta_train_iters = rospy.get_param("/firefly/meta_train_iters")
            params1.meta_nrollouts_per_iter = rospy.get_param("/firefly/meta_nrollouts_per_iter")
            params1.k_spt = rospy.get_param("/firefly/k_spt")
            params1.k_qry = rospy.get_param("/firefly/k_qry")
            params1.load_model = rospy.get_param("/firefly/load_model_m")
            params1.running_total_points = rospy.get_param("/firefly/running_total_points")
            params1.abandon_samples = rospy.get_param("/firefly/abandon_samples")
            params1.log_sample_data = rospy.get_param("/firefly/log_sample_data")
            params1.plan_hor = rospy.get_param("/firefly/plan_hor")

            ###################################################
            params_meta = DotMap()
            params_meta.meta_lr = rospy.get_param("/firefly/meta_lr")
            params_meta.update_lr = rospy.get_param("/firefly/update_lr")
            params_meta.update_step = rospy.get_param("/firefly/update_step")
            params_meta.update_step_test = rospy.get_param("/firefly/update_step_test")
            # params_meta.model_in = rospy.get_param("/firefly/model_in")
            # params_meta.model_out = rospy.get_param("/firefly/model_out")
            params_meta.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params_meta.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params_meta.load_model = rospy.get_param("/firefly/load_model_m")
            params_meta.m_task_num = rospy.get_param("/firefly/task_num")
            params_meta.n_way = rospy.get_param("/firefly/n_way")
            params_meta.k_spt = rospy.get_param("/firefly/k_spt")
            params_meta.k_qry = rospy.get_param("/firefly/k_qry")
            params_meta.m_epoch = rospy.get_param("/firefly/epoch_m")
            params_meta.m_epoch_running = rospy.get_param("/firefly/epoch_m_running")

            ###################################################
            policy = MPC(params,env,params_meta,obs=2)
            exp = MBExperiment(params1,env,policy,logger,meta=True)

            # os.makedirs(exp.logdir)
            if test_model:
                exp.test_experiment_meta()
            else:
                # exp.run_experiment_meta_without_online(log_path)
                # exp.run_experiment_meta_online1_1(log_path)
                exp.run_experiment_meta_online1_evaluation(log_path)
                # exp.run_experiment_meta_online1_1all(log_path)
            
            env.close()

        if mode == "train_offline_MBRL_model":
            log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_3d_MBRL_only_offline_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))

            os.makedirs(log_path, exist_ok=True)
            logger = SummaryWriter(logdir=log_path) # used for tensorboard
            #MBRL TRAINING 
            set_global_seeds(0)

            # assert ctrl_type == 'MPC'
            params = DotMap()
            params.per = rospy.get_param("/firefly/per")
            params.prop_mode = rospy.get_param("/firefly/prop_mode")
            params.npart = rospy.get_param("/firefly/npart")
            params.ign_var = rospy.get_param("/firefly/ign_var") 
            params.opt_mode = rospy.get_param("/firefly/opt_mode")
            params.plan_hor = rospy.get_param("/firefly/plan_hor")
            params.num_nets = rospy.get_param("/firefly/num_nets")
            params.epsilon = rospy.get_param("/firefly/epsilon")
            params.alpha = rospy.get_param("/firefly/alpha")
            params.epochs = rospy.get_param("/firefly/epochs")
            params.model_in = rospy.get_param("/firefly/model_in")
            params.model_out = rospy.get_param("/firefly/model_out")
            params.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params.route = rospy.get_param("/firefly/route")
            params.popsize = rospy.get_param("/firefly/popsize")
            params.max_iters = rospy.get_param("/firefly/max_iters")
            params.num_elites = rospy.get_param("/firefly/num_elites")

            params1 = DotMap()
            params1.ntrain_iters = rospy.get_param("/firefly/ntrain_iters")
            params1.nrollouts_per_iter = rospy.get_param("/firefly/nrollouts_per_iter")
            params1.ninit_rollouts = rospy.get_param("/firefly/ninit_rollouts")
            params1.nrecord = rospy.get_param("/firefly/nrecord")
            params1.neval = rospy.get_param("/firefly/neval")

            policy = MPC(params,env)
            exp = MBExperiment(params1,env,policy,logger)

            # os.makedirs(exp.logdir)
            exp.run_experiment_only_offline()
            env.close()

        if mode == "train_offline_meta_model":
            # space_3d = rospy.get_param("/firefly/3d_space")
            log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_3d_meta_only_offline_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))

            os.makedirs(log_path, exist_ok=True)
            logger = SummaryWriter(logdir=log_path) # used for tensorboard

            # assert ctrl_type == 'MPC'
            params = DotMap()
            params.per = rospy.get_param("/firefly/per")
            params.prop_mode = rospy.get_param("/firefly/prop_mode")
            params.opt_mode = rospy.get_param("/firefly/opt_mode")
            params.npart = rospy.get_param("/firefly/npart")
            params.ign_var = rospy.get_param("/firefly/ign_var") 
            params.mode = rospy.get_param("/firefly/opt_mode")
            params.plan_hor = rospy.get_param("/firefly/plan_hor")
            params.num_nets = rospy.get_param("/firefly/num_nets")
            params.epsilon = rospy.get_param("/firefly/epsilon")
            params.alpha = rospy.get_param("/firefly/alpha")
            params.epochs = rospy.get_param("/firefly/epochs")
            params.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params.route = rospy.get_param("/firefly/route")
            params.popsize = rospy.get_param("/firefly/popsize")
            params.max_iters = rospy.get_param("/firefly/max_iters")
            params.num_elites = rospy.get_param("/firefly/num_elites")
            params.visualization = rospy.get_param("/firefly/visualization")
            test_model = rospy.get_param("/firefly/test_model")

            params1 = DotMap()
            params1.task_hor = rospy.get_param("/firefly/task_hor")
            params1.meta_train_iters = rospy.get_param("/firefly/meta_train_iters")
            params1.meta_nrollouts_per_iter = rospy.get_param("/firefly/meta_nrollouts_per_iter")
            params1.k_spt = rospy.get_param("/firefly/k_spt")
            params1.k_qry = rospy.get_param("/firefly/k_qry")
            params1.load_model = rospy.get_param("/firefly/load_model_m")
            params1.running_total_points = rospy.get_param("/firefly/running_total_points")
            params1.abandon_samples = rospy.get_param("/firefly/abandon_samples")
            params1.log_sample_data = rospy.get_param("/firefly/log_sample_data")

            ###################################################
            params_meta = DotMap()
            params_meta.meta_lr = rospy.get_param("/firefly/meta_lr")
            params_meta.update_lr = rospy.get_param("/firefly/update_lr")
            params_meta.update_step = rospy.get_param("/firefly/update_step")
            params_meta.update_step_test = rospy.get_param("/firefly/update_step_test")
            params_meta.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params_meta.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params_meta.load_model = rospy.get_param("/firefly/load_model_m")
            params_meta.m_task_num = rospy.get_param("/firefly/task_num")
            params_meta.n_way = rospy.get_param("/firefly/n_way")
            params_meta.k_spt = rospy.get_param("/firefly/k_spt")
            params_meta.k_qry = rospy.get_param("/firefly/k_qry")
            params_meta.m_epoch = rospy.get_param("/firefly/epoch_m")
            params_meta.m_epoch_running = rospy.get_param("/firefly/epoch_m_running")

            ###################################################
            policy = MPC(params,env,params_meta)
            exp = MBExperiment(params1,env,policy,logger,meta=True)

            # os.makedirs(exp.logdir)
            exp.run_experiment_only_offline()
            # exp.run_experiment_only_offline_test()
            
            env.close()

        if mode == "embedding_NN":
            log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_3d_embedding_only_offline_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))

            os.makedirs(log_path, exist_ok=True)
            logger = SummaryWriter(logdir=log_path) # used for tensorboard

            # assert ctrl_type == 'MPC'
            params = DotMap()
            params.per = rospy.get_param("/firefly/per")
            params.prop_mode = rospy.get_param("/firefly/prop_mode")
            params.opt_mode = rospy.get_param("/firefly/opt_mode")
            params.npart = rospy.get_param("/firefly/npart")
            params.ign_var = rospy.get_param("/firefly/ign_var") 
            params.mode = rospy.get_param("/firefly/opt_mode")
            params.plan_hor = rospy.get_param("/firefly/plan_hor")
            params.num_nets = rospy.get_param("/firefly/num_nets")
            params.epsilon = rospy.get_param("/firefly/epsilon")
            params.alpha = rospy.get_param("/firefly/alpha")
            params.epochs = rospy.get_param("/firefly/epochs")
            params.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params.route = rospy.get_param("/firefly/route")
            params.popsize = rospy.get_param("/firefly/popsize")
            params.max_iters = rospy.get_param("/firefly/max_iters")
            params.num_elites = rospy.get_param("/firefly/num_elites")
            params.visualization = rospy.get_param("/firefly/visualization")
            test_model = rospy.get_param("/firefly/test_model")

            params1 = DotMap()
            params1.task_hor = rospy.get_param("/firefly/task_hor")
            params1.log_sample_data = rospy.get_param("/firefly/log_sample_data")
            params1.embedding = rospy.get_param("/firefly/em_embedding")

            ###################################################
            params_meta = DotMap()
            params_meta.embedding = rospy.get_param("/firefly/em_embedding")
            ###################################################
            policy = MPC(params,env,params_meta)
            exp = MBExperiment(params1,env,policy,logger,meta=True)

            # os.makedirs(exp.logdir)
            training = 0
            exp.run_experiment_embedding_meta(training,logger,path=log_path)
            
            env.close()

        if mode == "variation_inference":
            log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_3d_variation_inference_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))

            os.makedirs(log_path, exist_ok=True)
            logger = SummaryWriter(logdir=log_path) # used for tensorboard

            # assert ctrl_type == 'MPC'
            params = DotMap()
            params.per = rospy.get_param("/firefly/per")
            params.prop_mode = rospy.get_param("/firefly/prop_mode")
            params.opt_mode = rospy.get_param("/firefly/opt_mode")
            params.npart = rospy.get_param("/firefly/npart")
            params.ign_var = rospy.get_param("/firefly/ign_var") 
            params.mode = rospy.get_param("/firefly/opt_mode")
            params.plan_hor = rospy.get_param("/firefly/plan_hor")
            params.num_nets = rospy.get_param("/firefly/num_nets")
            params.epsilon = rospy.get_param("/firefly/epsilon")
            params.alpha = rospy.get_param("/firefly/alpha")
            params.epochs = rospy.get_param("/firefly/epochs")
            params.model_3d_in = rospy.get_param("/firefly/model_3d_in")
            params.model_3d_out = rospy.get_param("/firefly/model_3d_out")
            params.route = rospy.get_param("/firefly/route")
            params.popsize = rospy.get_param("/firefly/popsize")
            params.max_iters = rospy.get_param("/firefly/max_iters")
            params.num_elites = rospy.get_param("/firefly/num_elites")
            params.visualization = rospy.get_param("/firefly/visualization")
            test_model = rospy.get_param("/firefly/test_model")

            params1 = DotMap()
            params1.task_hor = rospy.get_param("/firefly/task_hor")
            params1.log_sample_data = rospy.get_param("/firefly/log_sample_data")
            params1.embedding = rospy.get_param("/firefly/em_embedding")
            params1.VI = rospy.get_param("/firefly/VI")

            ###################################################
            params_meta = DotMap()
            params_meta.VI = rospy.get_param("/firefly/VI")
            params_meta.embedding = rospy.get_param("/firefly/em_embedding")
            ###################################################
            policy = MPC(params,env,params_meta)
            exp = MBExperiment(params1,env,policy,logger,meta=True)

            trainning = 1
            exp.run_experiment_variation_inference(trainning)

            env.close()

        if mode == "point_cloud_collection":
            #used to collect 3d point clouds to fit a 2d signed distance function (sdf) by a Kinect camera
            #Total variations
            #task 1: wind 0.0 L 0.6  
            # task 2: wind 0.3 L 1.0 
            # task 3: wind 0.5 L 0.8 
            # task 4: wind 0.8 L 1.2
            wind_condition_x = 0.4
            wind_condition_y = 0.0
            L = 1.4

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)

            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))

                #Reload the action goals given by the joystick
                mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15.mat")
                joy_waypoints = mat_contents['acs1']
                
                joy_waypoints[:,0] = joy_waypoints[:,0]*env.max_x
                joy_waypoints[:,1] = joy_waypoints[:,1]*env.max_y
                joy_waypoints[:,2] = joy_waypoints[:,2]*env.max_z
                # Initialize the environment and get first state of the robot
                observation, _,_ = env.reset()
                env.set_pos_callback_cloud_loop()
                #######################################
                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    rospy.logwarn("############### data points available =>" + str(i))
                    # Pick an action based on the current state
                    action, obs, obs1, obs2 = env.step_pos_replay(joy_waypoints[i,:],L)
                    env.set_pos_callback_cloud_loop()
                    
                    i += 1
                    if i == 2500:
                        break

            print("finish saving file")
        
            env.close()

        if mode == "point_cloud_collection_additional":
            #used to collect 3d point clouds to fit a 2d signed distance function (sdf) by a Kinect camera
            #Total variations
            #task 1: wind 0.0 L 0.6  
            # task 2: wind 0.3 L 1.0 
            # task 3: wind 0.5 L 0.8 
            # task 4: wind 0.8 L 1.2
            # test task 1: wind 1.0 L 0.8
            # test task 2: wind 0.6 L 1.4

            wind_condition_x = 0.6
            wind_condition_y = 0.0
            L = 1.4

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)

            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))

                #Reload the action goals given by the joystick
                mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15_cloud_final.mat")
                joy_waypoints = mat_contents['acs1']
                
                joy_waypoints[:,0] = joy_waypoints[:,0]*env.max_x
                joy_waypoints[:,1] = joy_waypoints[:,1]*env.max_y
                joy_waypoints[:,2] = joy_waypoints[:,2]*env.max_z
                # Initialize the environment and get first state of the robot
                observation, _,_ = env.reset()
                env.set_pos_callback_cloud_loop()
                #######################################
                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    rospy.logwarn("############### data points available =>" + str(i))
                    # Pick an action based on the current state
                    action, obs, obs1, obs2 = env.step_pos_replay(joy_waypoints[i,:],L)
                    env.set_pos_callback_cloud_loop()
                    
                    i += 1
                    if i == 5000:
                        break

            print("finish saving file")
        
            env.close()
            
        if mode == "grid_map_collection":
            #visual encoder-decoder by a Kinect camera
            #Total variations
            #task 1: wind 0.0 L 0.6  
            # task 2: wind 0.3 L 1.0 
            # task 3: wind 0.5 L 0.8 
            # task 4: wind 0.8 L 1.2
            wind_condition_x = 0.8
            wind_condition_y = 0.0 #y is always zero, we only consider wind in x axis
            L = 1.2

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)
            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))
                # space_3d = rospy.get_param("/firefly/3d_space")

                # Initialize the environment and get first state of the robot
                observation, _ = env.reset()

                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    rospy.logwarn("############### data points available =>" + str(i))
                    # Pick an action based on the current state
                    
                    action, action1, obs, obs1, obs2 = env.step_pos(L)

                    i += 1
                    if i == 1200:
                        break

            print("finish saving file")
        
            env.close()


        if mode == "sdf_grid_map_visualization":
            checkpoint_filepath = "/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2agents_all_1/lightning_logs/version_0/checkpoints"
            checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
            cfg = DotMap()
            cfg.seed = 1
            cfg.lr = 0.00005 # more_layers: 0.00005, one layer: 0.0001
            cfg.if_cuda = True
            cfg.gamma = 0.5
            cfg.log_dir = 'logs'
            cfg.num_workers = 8
            cfg.model_name = 'Occupancy_predictor'
            cfg.data_filepath1 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/firefly_points_3d_wind_x0.0_y0.0_2agents_L0.6/preprocess"
            cfg.data_filepath2 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/firefly_points_3d_wind_x0.3_y0.0_2agents_L1.0/preprocess"
            cfg.data_filepath3 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/firefly_points_3d_wind_x0.5_y0.0_2agents_L0.8/preprocess"
            cfg.data_filepath4 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/firefly_points_3d_wind_x0.8_y0.0_2agents_L1.2/preprocess"
            cfg.data_filepath5 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/firefly_points_3d_wind_x1.0_y0.0_2agents_L0.9/preprocess"
            cfg.data_filepath6 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/firefly_points_3d_wind_x0.4_y0.0_2agents_L1.4/preprocess"
            cfg.lr_schedule = [10000000]
            cfg.num_gpus = 1
            cfg.epochs = 10
            cfg.dof = 12
            cfg.coord_system = 'cartesian'
            cfg.tag = '2agents_all'
            seed(cfg)
            seed_everything(cfg.seed)

            log_dir = '_'.join([cfg.log_dir,
                                cfg.model_name,
                                cfg.tag,
                                str(cfg.seed)])

            occupancy_model = Predictor_Model(lr=cfg.lr,
                                    dof=cfg.dof,
                                    if_cuda=cfg.if_cuda,
                                    if_test=True,
                                    gamma=cfg.gamma,
                                    log_dir=log_dir,
                                    num_workers=cfg.num_workers,
                                    data_filepath=[cfg.data_filepath1,cfg.data_filepath2,cfg.data_filepath3,cfg.data_filepath4,cfg.data_filepath5,cfg.data_filepath6],
                                    coord_system=cfg.coord_system,
                                    lr_schedule=cfg.lr_schedule)

            ckpt = torch.load(checkpoint_filepath)
            occupancy_model.load_state_dict(ckpt['state_dict'])
            occupancy_model = occupancy_model.to('cuda')
            occupancy_model.eval()
            occupancy_model.freeze()

            obs_points = env.obs_p
            obs_points1 = env.obs_p1
            obs_pos = env.obs_p_pos
            obs_pos1 = env.obs_p1_pos

            sdf_pub_makers = rospy.Publisher('/sdf_makers', Marker, queue_size=10)
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.action = marker.ADD
            marker.pose.position.x = obs_pos[0,0]
            marker.pose.position.y = obs_pos[0,1]
            marker.pose.position.z = 1.9
            marker.pose.orientation.w=1.0
            marker.id = 0
            marker.type = marker.TEXT_VIEW_FACING
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            sdf_pub_makers1 = rospy.Publisher('/sdf_makers1', Marker, queue_size=10)
            obs_pub_makers = rospy.Publisher('/obs_makers', Marker, queue_size=10)
            marker1 = Marker()
            marker1.header.frame_id = "world"
            marker1.header.stamp = rospy.Time.now()
            marker1.action = marker1.ADD
            marker1.pose.position.x = obs_pos1[0,0]
            marker1.pose.position.y = obs_pos1[0,1]
            marker1.pose.position.z = 1.9
            marker1.pose.orientation.w=1.0
            marker1.id = 1
            marker1.type = marker1.TEXT_VIEW_FACING
            marker1.color.r = 1.0
            marker1.color.g = 0.0
            marker1.color.b = 0.0
            marker1.color.a = 1.0
            marker1.scale.x = 0.2
            marker1.scale.y = 0.2
            marker1.scale.z = 0.2

            ###################################################
            map_pub_makers = rospy.Publisher('/sdf_map_makers', OccupancyGrid, queue_size=100)
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = rospy.Time.now()
            grid_msg.header.frame_id = "world"
            grid_msg.info.origin.position.x = 0.0;
            grid_msg.info.origin.position.y = 2.0;
            grid_msg.info.origin.position.z = 0.0;
            grid_msg.info.origin.orientation.x = 1.0;
            grid_msg.info.origin.orientation.y = 0.0;
            grid_msg.info.origin.orientation.z = 0.0;
            grid_msg.info.origin.orientation.w = 0.0;
            grid_msg.info.resolution = 0.0157
            grid_msg.info.width = 256
            grid_msg.info.height = 256

            N=256
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

            ###################################################
            
            wind_condition_x = 0.0
            wind_condition_y = 0.0 #y is always zero, we only consider wind in x axis
            L = 0.6

            env.wind_controller_x.publish(wind_condition_x)
            env.wind_controller_y.publish(wind_condition_y)
            env.set_L(L)
            sample_obs_num = 300

            for x in range(1):
                rospy.logdebug("############### WALL START EPISODE=>" + str(x))
                
                observation, _ ,_= env.reset()

                # A1 = []
                # O, A = [observation], []
                # O1 = []
                # O2 = []
               
                i = 0

                while not rospy.is_shutdown():
                    if env.shutdown_joy:
                        break

                    # rospy.logwarn("############### data points available =>" + str(i))
                    # # Pick an action based on the current state
                    
                    action, action1, obs, obs1, obs2 = env.step_pos(L)
                    ###############################################################
                    # print(self.sy_cur_obs[6:8])
                    # current_p = obs[6:8].copy()
                    # current_p[0] = current_p[0]*4.0
                    # current_p[1] = current_p[1]*2.0
                    # d = np.sqrt(((current_p-obs_pos)**2).sum(axis=1))[0]
                    # d1 = np.sqrt(((self.sy_cur_obs[6:8]-obs_pos1)**2).sum(axis=1))[0]
                    # print("dis: ",d)

                    #obs
                    state_e = torch.from_numpy(obs[None,:]).float().to(TORCH_DEVICE)
                    state_e1 = state_e.expand(sample_obs_num,-1)
                    state_expand = state_e1.view(-1,obs.shape[0])

                    obs_num = obs_points.shape[0]
                    obs_num_shuffle = np.random.permutation(obs_num)
                    obs_index = obs_num_shuffle[:sample_obs_num]
                    obs_points_e = torch.from_numpy(obs_points[obs_index,:]).float().to(TORCH_DEVICE)
                    obs_p_final = obs_points_e.view(-1,2)
                    assert(state_expand.shape[0]==obs_p_final.shape[0])

                    input_c = torch.cat((obs_p_final,state_expand),1)
                    sdf_output = occupancy_model.model(input_c)
                    sdf_output_np = sdf_output.detach().cpu().numpy()
                    sdf_output_a = sdf_output_np.reshape(-1,1)
                    sdf_output_index = np.where(sdf_output_a>0.050)[0]
                    sdf_output_a[sdf_output_index] = 0
                    sdf_np_p_index = np.where((sdf_output_a>0)&(sdf_output_a<=0.050))[0]
                    sdf_output_a[sdf_np_p_index] = -0.1 
                    sdf_output_cost = np.sum(sdf_output_a)

                    marker.text = str(sdf_output_cost)
                    sdf_pub_makers.publish(marker)

                    #obs1
                    state1_e = torch.from_numpy(obs[None,:]).float().to(TORCH_DEVICE)
                    state1_e1 = state1_e.expand(sample_obs_num,-1)
                    state_expand1 = state1_e1.view(-1,obs.shape[0])

                    obs_num1 = obs_points1.shape[0]
                    obs_num_shuffle1 = np.random.permutation(obs_num1)
                    obs_index1 = obs_num_shuffle1[:sample_obs_num]
                    obs_points_e1 = torch.from_numpy(obs_points1[obs_index1,:]).float().to(TORCH_DEVICE)
                    obs_p_final1 = obs_points_e1.view(-1,2)
                    assert(state_expand1.shape[0]==obs_p_final1.shape[0])

                    input_c1 = torch.cat((obs_p_final1,state_expand1),1)
                    sdf_output1 = occupancy_model.model(input_c1)
                    sdf_output_np1 = sdf_output1.detach().cpu().numpy()
                    sdf_output_a1 = sdf_output_np1.reshape(-1,1)
                    sdf_ouput_index1 = np.where(sdf_output_a1>0.05)[0]
                    sdf_output_a1[sdf_ouput_index1] = 0
                    sdf_np_p_index1 = np.where((sdf_output_a1>0)&(sdf_output_a1<=0.050))[0]
                    sdf_output_a1[sdf_np_p_index1] = -0.1 
                    sdf_output_cost1 = np.sum(sdf_output_a1)

                    marker1.text = str(sdf_output_cost1)
                    sdf_pub_makers1.publish(marker1)
                     
                    ###############################################################
                    # A1.append(action1) #for replay goal
                    # A.append(action) #real displacement
                    # O.append(obs)
                    # O1.append(obs1)
                    # O2.append(obs2)
                    obsm = Marker()
                    obsm.header.stamp = rospy.Time.now()
                    obsm.header.frame_id = "world"
                    obsm.type = obsm.CUBE;  
                    obsm.action = obsm.ADD
                    obsm.id = 200
                    obsm.pose.position.x = 3
                    obsm.pose.position.y = 0.5
                    obsm.pose.position.z = 0.75
                    obsm.pose.orientation.w = 1.0
                    obsm.scale.x = .4
                    obsm.scale.y = .4
                    obsm.scale.z = 1.5
                    obsm.color.a = 1
                    obsm.color.r = 0.4
                    obsm.color.g = 0.4
                    obsm.color.b = 0.2
                    obsm.lifetime = rospy.Duration()

                    obsm1 = Marker()
                    obsm1.header.stamp = rospy.Time.now()
                    obsm1.header.frame_id = "world"
                    obsm1.type = obsm1.CUBE;  
                    obsm1.action = obsm1.ADD
                    obsm1.id = 201
                    obsm1.pose.position.x = 1
                    obsm1.pose.position.y = -1
                    obsm1.pose.position.z = 0.75
                    obsm1.pose.orientation.w = 1.0
                    obsm1.scale.x = .6   
                    obsm1.scale.y = .6
                    obsm1.scale.z = 1.5
                    obsm1.color.a = 1
                    obsm1.color.r = 0.4
                    obsm1.color.g = 0.4
                    obsm1.color.b = 0.2
                    obsm1.lifetime = rospy.Duration()

                    obs_pub_makers.publish(obsm)
                    obs_pub_makers.publish(obsm1)

                    final_robot_states = np.tile(obs, (sample_test.shape[0], 1))
                    final_robot_states = torch.from_numpy(final_robot_states).float().cuda()
                    sample_set = torch.cat((sample_test, final_robot_states), dim=1)
                    grid_data = occupancy_model.model(sample_set).squeeze().detach().cpu().numpy()*100.0
                    on_index = np.where(grid_data<=0.05)[0]
                    # off_index = np.where(grid_data>0)[0]
                    grid_data[on_index] = 0
                    # grid_data[off_index] = 100
                    grid_msg.data = list(grid_data.astype(int))
                    map_pub_makers.publish(grid_msg)

                    i += 1
        
            env.close()
        
        
    except KeyboardInterrupt:
        env.close()

