from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dotmap import DotMap
from scipy.io import savemat
from tqdm import trange
from Agent import Agent
import scipy.io as sio
import torch 
import numpy as np
import rospy
import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from replay_buffer import ReplayBuffer
from neural_nets import PPO_model,Conv_autoencoder
import time
from time import localtime, strftime
from tensorboardX import SummaryWriter
from normalization import Normalization, RewardScaling

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')

class MBExperiment:
    def __init__(self, params, env, policy, logger, meta = False):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """

        # Assert True arguments that we currently do not support
        # assert params.sim_cfg.get("stochastic", False) == False

        self.env = env
        self.task_hor = params.task_hor
        self.log_sample_data = params.log_sample_data
        self.horizon = params.plan_hor
        # self.space_3d = rospy.get_param("/firefly/3d_space")
        # print(self.wind_test_types)
        self.meta = meta
        #params of MBRL
        if not meta:
            self.agent = Agent(env)
            self.ntrain_iters = params.ntrain_iters
            self.nrollouts_per_iter = params.nrollouts_per_iter
            self.ninit_rollouts = params.ninit_rollouts
            self.neval = params.neval
        else:
            if not params.embedding:
                #params of Meta learning
                self.agent = Agent(env,meta=True)
                self.meta_train_iters = params.meta_train_iters
                self.meta_nrollouts_per_iter = params.meta_nrollouts_per_iter
                self.k_spt = params.k_spt
                self.k_qry = params.k_qry
                self.load_model = params.load_model
                self.running_total_points = params.running_total_points
                self.abandon_samples = params.abandon_samples
            else:
                if params.VI:
                    pass
                else:
                    self.agent = Agent(env,meta=True)

        self.policy = policy
        self.logger = logger


    def run_experiment(self,training,path):
        """Perform model-based experiment.
        """
        # seed 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(200)
        torch.cuda.manual_seed_all(200)
        np.random.seed(200)

        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4

        wind_condition_x = 0.6
        wind_condition_y = 0.0
        L = 1.4

        if training:
            # Perform initial rollouts
            samples = []

            mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x{0}_2agents_L{1}_dt_0.15.mat".format(wind_condition_x,L))
            train_obs = mat_contents['obs'] 
            train_acs = mat_contents['acs']

            #samples [episode, steps,n]
            self.policy.train(train_obs, train_acs,self.logger)

            # Training loop
            for i in trange(self.ntrain_iters):
                print("####################################################################")
                print("Starting training iteration %d." % (i + 1))

                samples = []

                #horizon, policy, wind_test_type, adapt_size=None, log_data=None, data_path=None
                #MBRL is baseline no need to log data in agent sampling
                for j in range(max(self.neval, self.nrollouts_per_iter)):
                    samples.append(
                            self.agent.sample(
                                self.task_hor, self.policy, wind_condition_x,wind_condition_y,L
                            )
                        )
                # print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:self.neval]])
                self.logger.add_scalar('Reward', np.mean([sample["reward_average"] for sample in samples[:]]), i)
                samples = samples[:self.nrollouts_per_iter]

                if i < self.ntrain_iters - 1:
                    #add new samples into the whole dataset and train the whole dataset
                    self.policy.train(
                        [sample["obs"] for sample in samples],
                        [sample["ac"] for sample in samples],
                        self.logger
                    )

            self.logger.close()
        else:
            sample = self.agent.sample(self.task_hor, self.policy, wind_condition_x,wind_condition_y, L, log_data=self.log_sample_data, data_path=path)
            #if path_length of path i is less than k_spt+k_qry, we have to resample it
            
            if self.log_sample_data:
                print("start logging")
                savemat(path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
                savemat(path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
                savemat(path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
                savemat(path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
                savemat(path+'/storeObs.mat', mdict={'arr': sample["obs"]})
                data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
                savemat(path+'/store_destraj.mat', mdict={'arr': data['arr']})
            
            self.logger.close()

    # def run_experiment_meta(self):
    #     """Perform meta experiment.
    #        we load the offline meta model and do the online training
    #     """
    #     torch.manual_seed(222)
    #     torch.cuda.manual_seed_all(222)
    #     np.random.seed(222)

    #     if not self.load_model:
    #         self.policy.offline_train(self.logger)

    #     for i in range(self.meta_train_iters):
    #         samples = []
    #         total_num = 0

    #         while total_num < self.running_total_points:
    #             #runing rollouts for collection samples for meta training
    #             chosen_winds = np.random.choice(self.wind_test_types,self.meta_nrollouts_per_iter,replace=False)
    #             for j in range(self.meta_nrollouts_per_iter):
    #                 sample = self.agent.sample(self.task_hor, self.policy, chosen_winds[j], self.k_spt)
    #                 #if path_length of path i is less than k_spt+k_qry, we have to resample it
    #                 if sample['ac'].shape[0] > self.k_spt+self.k_qry:
    #                     samples.append(sample)
    #                 total_num += sample['ac'].shape[0]
            
    #         self.logger.add_scalar('Reward', np.mean([sample["reward_sum"] for sample in samples[:]]), i)

    #         self.policy.train_meta([sample["obs"] for sample in samples],
    #                 [sample["ac"] for sample in samples], self.logger,i)
        
    #     self.logger.close()

    # def test_experiment_meta(self):
    #     """Perform meta experiment.
    #        we load the offline meta model and do the online training
    #     """
    #     torch.manual_seed(222)
    #     torch.cuda.manual_seed_all(222)
    #     np.random.seed(222)

    #     samples = []
    #     #runing rollouts for collection samples for meta training
    #     chosen_winds = np.random.choice(self.wind_test_types,self.meta_nrollouts_per_iter,replace=False)
    #     for j in range(self.meta_nrollouts_per_iter):
    #         sample = self.agent.sample(self.task_hor, self.policy, chosen_winds[j], self.k_spt)
    #         samples.append(sample)
        
    #         self.logger.add_scalar('Reward', samples[j]["reward_sum"], j)
        
    #     self.logger.close()

    def run_experiment_meta_without_online(self, path):
        """Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # seed 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)

        # if not self.load_model:
        #     self.policy.offline_train(self.logger)

        #runing rollouts for collection samples for meta training
        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        wind_condition_x = 1.0
        wind_condition_y = 0.0
        L = 0.8

        sample = self.agent.sample(self.task_hor, self.policy, wind_condition_x,wind_condition_y, L, adapt_size = self.k_spt, log_data=self.log_sample_data, data_path=path)
        #if path_length of path i is less than k_spt+k_qry, we have to resample it
        
        if self.log_sample_data:
            savemat(path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            savemat(path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
            savemat(path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
            savemat(path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
            savemat(path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            savemat(path+'/store_destraj.mat', mdict={'arr': data['arr']})
            # savemat(path+'/storeAcs.mat', mdict={'arr': sample["ac"]})
        # self.logger.add_scalar('Reward', sample["reward_sum"], 0)
        ##########
        #plot results
        # data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
        # figure1 = plt.figure('figure1')
        # plt.plot(sample["rewards"])
        # plt.plot(sample["error_x"])
        # plt.plot(sample["error_y"])
        # plt.plot(sample["error_z"])
        # plt.gca().legend(('Euclidean', 'x', 'y', 'z'))
        # plt.title('position error')
        # plt.xlabel('t')
        # plt.ylabel('error(m)')

        # figure2 = plt.figure('figure2')
        # ax = figure2.gca(projection='3d')
        # ax.plot3D(sample["obs"][:,6]*4,sample["obs"][:,7]*2,sample["obs"][:,8]*2, 'gray')
        # ax.plot3D(data['arr'][:,0],data['arr'][:,1],data['arr'][:,2], 'r--')
        # plt.title('3d trajectory')
        # ax.set_xlabel('x(m)')
        # ax.set_ylabel('y(m)')
        # ax.set_zlabel('z(m)')

        # # figure3 = plt.figure('figure3')
        # # plt.plot(sample["prediction_error"])
        # # plt.title('Prediction error')
        # # plt.xlabel('t')
        # # plt.ylabel('MSE')

        # plt.show()

        ##########
        self.logger.close()

    
    # def run_experiment_meta_online1(self, path):
    #     """
    #        Correct actions
    #        Perform meta experiment.
    #        we load the offline meta model and without the online training, only one episode adaptation
    #     """

    #     torch.manual_seed(222)
    #     torch.cuda.manual_seed_all(222)
    #     np.random.seed(222)
    #     train_iters = 1600

    #     #ppo logger
    #     log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
    #     os.makedirs(log_path_ppo, exist_ok=True)
    #     logger_ppo = SummaryWriter(logdir=log_path_ppo) # used for tensorboard

    #     args = DotMap()
    #     # if not use image, change dimension
    #     args.with_image = False
    #     args.state_dim = 48
    #     args.action_dim = 3
    #     args.max_action = 0.1
    #     args.batch_size = 2048
    #     args.mini_batch_size = 64
    #     args.max_train_steps = train_iters * self.task_hor
    #     args.lr_a  = 3e-4
    #     args.lr_c = 3e-4
    #     args.gamma = 0.99
    #     args.lamda = 0.95
    #     args.epsilon = 0.2
    #     args.K_epochs = 10
    #     args.entropy_coef  = 0.005#0.01
    #     args.use_grad_clip = True
    #     args.use_lr_decay = True
    #     args.use_adv_norm = True
    #     # args.horizon = 1
    #     args.evaluate_s = 0

    #     #runing rollouts for collection samples for meta training
    #     wind_condition_x = 0.0
    #     wind_condition_y = 0.0
    #     L = 0.6

    #     replay_buffer = ReplayBuffer(args)
    #     ppo_agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)
        
    #     if args.with_image:
    #         act_encoder = Conv_autoencoder().to(TORCH_DEVICE)
    #         act_encoder.load_encoder()
    #         act_encoder.eval()

    #     adapt_size = self.k_spt
    #     log_data=self.log_sample_data
    #     data_path=path
    #     total_steps = 0
    #     evaluate_frequency = 20
    #     reward_index = 0
    #     reward_index_log = 0 
    #     reward_repeat = []
    #     episode_n = 0
    #     repeat_eval = False
    #     # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    #     if self.meta:
    #         self.adapt_buffer = dict(obs=[],act=[])
    #         self.env.wind_controller_x.publish(wind_condition_x)
    #         self.env.wind_controller_y.publish(wind_condition_y)
    #     else:
    #         self.env.wind_controller_x.publish(wind_condition_x)
    #         self.env.wind_controller_y.publish(wind_condition_y)

    #     for i in range(train_iters):

    #         if i == 1:
    #             if self.meta:
    #                 self.policy.model.save_model(0,model_path=data_path)
        
    #         times, rewards = [], []
    #         errorx = []
    #         errory = []
    #         errorz = []
    #         self.env.set_L(L)
    #         o1, goal, g_s = self.env.reset()
    #         O, A, reward_sum, done = [o1], [], 0, False
    #         top_act_seq = []
    #         prediction_error = []
    #         # reward_scaling.reset()

    #         past_corrected_goals = [np.zeros((1,3))]
    #         past_traj = [np.zeros((1,15))]
    #         past_traj_error = [np.zeros((1,3))]

    #         if i>0:
    #             if args.with_image == True:
    #                 g_map = torch.from_numpy(self.env.get_depth_map()).cuda().float()
    #                 g_map_conv = g_map[None][None]
    #                 d = act_encoder.encoder_forward(g_map_conv)
    #                 O_t = torch.from_numpy(O[t]).cuda().float()
    #                 s_all = torch.cat((d,O_t[None]),dim=1)
    #                 assert s_all.shape[1]==72
    #             else:
    #                 #normalize goal input in observation
    #                 O_t = torch.from_numpy(o1).cuda().float()
    #                 g_normalize_s = g_s.copy()
    #                 g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
    #                 g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
    #                 g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
    #                 g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

    #                 # print("old:",past_traj[-1])
    #                 g_normalize_past = past_traj[-1]
    #                 g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
    #                 g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
    #                 g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
    #                 g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

    #                 g_normalize_past_c = past_corrected_goals[-1]
    #                 g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
    #                 g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
    #                 g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
    #                 g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

    #                 e_normalize_past = past_traj_error[-1]
    #                 e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

    #                 s_all = torch.cat((O_t[None],g_current,g_old,g_old_correct,e_old),dim=1)
    #                 assert s_all.shape[1]==12+15+15+3+3

    #         # if log_data:
    #         #     obs1_l = []
    #         #     obs2_l = []
    #         #     obs1 = self.env.get_uav_obs()[0]
    #         #     obs2 = self.env.get_uav_obs()[1]
    #         #     obs1_l.append(obs1)
    #         #     obs2_l.append(obs2)
    #         if i == 0:
    #             if self.meta:
    #                 self.adapt_buffer['obs'].append(o1)
    #                 self.policy.model.fast_adapted_params = None

    #         self.policy.reset()
        
    #         if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
    #             for t in range(self.task_hor):
    #                 # if t>100:
    #                 #     self.env.wind_controller_x.publish(0.2)
    #                 #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
    #                 #     self.env.set_L(0.8)
    #                 # break
    #                 start = time.time()

    #                 action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
                    
    #                 a = ppo_agent.evaluate(s_all) 
    #                 a_correct = a.reshape(1,3).copy()

    #                 past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
    #                 # print(g_s)
    #                 past_traj.append(g_s.copy())
    #                 action+=a
    #                 # print(store_top_s)
    #                 self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
    #                 self.env.pub_action_sequence1(store_bad_s)
    #                 A.append(action)
    #                 top_act_seq.append(act_l)
    #                 times.append(time.time() - start)

    #                 obs, reward, done, (goal, g_s) = self.env.step(A[t])

    #                 new_error_traj = np.zeros((1,3))
    #                 new_error_traj[:,0] = reward['error_x']
    #                 new_error_traj[:,1] = reward['error_y']
    #                 new_error_traj[:,2] = reward['error_z']
    #                 past_traj_error.append(new_error_traj)
                    
    #                 #reward process
    #                 reward['reward'] = -reward['reward']**2
    #                 # reward['reward'] = reward_scaling(reward['reward'])

    #                 if args.with_image == True:
    #                     if_obs = self.policy.occupancy_predictor(obs)
    #                     # print(if_obs)
    #                     #post process reward and done
    #                     if if_obs:
    #                         done = 1
    #                         reward['reward']-=50

    #                 if args.with_image == True:
    #                     g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
    #                     g_map_conv1 = g_map1[None][None]
    #                     d1 = act_encoder.encoder_forward(g_map_conv1)
    #                     obs_t = torch.from_numpy(obs).cuda().float()
    #                     s_all1 = torch.cat((d1,obs_t[None]),dim=1)
    #                     assert(s_all1.shape[1]==72)
    #                 else:
    #                     O_t1 = torch.from_numpy(obs).cuda().float()
    #                     g_normalize_s1 = g_s.copy()
    #                     g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
    #                     g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
    #                     g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
    #                     g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

    #                     # print("old1:",past_traj[-1])
    #                     g_normalize_past1 = past_traj[-1]
    #                     g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
    #                     g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
    #                     g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
    #                     g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

    #                     g_normalize_past_c1 = past_corrected_goals[-1]
    #                     g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
    #                     g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
    #                     g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
    #                     g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

    #                     e_normalize_past1 = past_traj_error[-1]
    #                     e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

    #                     s_all1 = torch.cat((O_t1[None],g_current1,g_old1,g_old_correct1,e_old1),dim=1)
    #                     assert s_all1.shape[1]==12+15+15+3+3

    #                 s_all = s_all1

    #                 # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
    #                 O.append(obs)
    #                 reward_sum += reward['reward']
    #                 rewards.append(reward['reward'])
    #                 errorx.append(reward['abs_error_x'])
    #                 errory.append(reward['abs_error_y'])
    #                 errorz.append(reward['abs_error_z'])
    #                 if done:
    #                     break
                
    #             if reward_index%3==0 and reward_index!=0:
    #                 repeat_eval = True

    #             reward_repeat.append(reward_sum)
                
    #             if repeat_eval:
    #                 reward_sum_av = np.mean(np.array(reward_repeat))
    #                 logger_ppo.add_scalar('Episode reward', reward_sum_av, reward_index_log)
    #                 ppo_agent.save_network(reward_index_log)
    #                 reward_index_log+=1
    #                 reward_repeat = []

    #             reward_index+=1
    #         else:
    #             for t in range(self.task_hor):
    #                 # if t>100:
    #                 #     self.env.wind_controller_x.publish(0.2)
    #                 #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
    #                 #     self.env.set_L(0.8)
    #                 # break
    #                 start = time.time()
    #                 if self.meta:
    #                     if i == 0:
    #                         if len(self.adapt_buffer['act'])>adapt_size:
    #                             #transform trajectories into adapt dataset
    #                             new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

    #                             new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

    #                             new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
    #                             new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
                
    #                             self.policy.model.adapt(new_train_in, new_train_targs)

    #                 #add ppo here    
    #                 # print("i:",self.env.trajectory.get_i())
    #                 # print("s_all:", s_all)

    #                 action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
    #                 if i>0:
    #                     a, a_logprob = ppo_agent.choose_action(s_all) 
    #                     # print("a:",a)
                    
    #                     #before added to goal, we need transform it to dimension [horizon,1,dim=3]
    #                     a_correct = a.reshape(1,3).copy()

    #                     past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
    #                     # print(g_s)
    #                     past_traj.append(g_s.copy())
    #                     # print(past_traj)

    #                     action+=a
    #                 # print(store_top_s)
    #                 self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
    #                 self.env.pub_action_sequence1(store_bad_s)
    #                 A.append(action)
    #                 top_act_seq.append(act_l)
    #                 times.append(time.time() - start)

    #                 obs, reward, done, (goal, g_s) = self.env.step(A[t])

    #                 new_error_traj = np.zeros((1,3))
    #                 new_error_traj[:,0] = reward['error_x']
    #                 new_error_traj[:,1] = reward['error_y']
    #                 new_error_traj[:,2] = reward['error_z']
    #                 past_traj_error.append(new_error_traj)
                    
    #                 #reward process
    #                 reward['reward'] = -reward['reward']**2
    #                 # reward['reward'] = reward_scaling(reward['reward'])

    #                 if i>0:
    #                     if args.with_image == True:
    #                         if_obs = self.policy.occupancy_predictor(obs)
    #                         # print(if_obs)
    #                         #post process reward and done
    #                         if if_obs:
    #                             done = 1
    #                             reward['reward']-=50

    #                     if args.with_image == True:
    #                         g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
    #                         g_map_conv1 = g_map1[None][None]
    #                         d1 = act_encoder.encoder_forward(g_map_conv1)
    #                         obs_t = torch.from_numpy(obs).cuda().float()
    #                         s_all1 = torch.cat((d1,obs_t[None]),dim=1)
    #                         assert(s_all1.shape[1]==72)
    #                     else:
    #                         O_t1 = torch.from_numpy(obs).cuda().float()
    #                         g_normalize_s1 = g_s.copy()
    #                         g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
    #                         g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
    #                         g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
    #                         g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

    #                         # print("old1:",past_traj[-1])
    #                         g_normalize_past1 = past_traj[-1]
    #                         g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
    #                         g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
    #                         g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
    #                         g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

    #                         g_normalize_past_c1 = past_corrected_goals[-1]
    #                         g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
    #                         g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
    #                         g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
    #                         g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

    #                         e_normalize_past1 = past_traj_error[-1]
    #                         e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

    #                         s_all1 = torch.cat((O_t1[None],g_current1,g_old1,g_old_correct1,e_old1),dim=1)
    #                         assert s_all1.shape[1]==12+15+15+3+3

    #                     if done or t == self.task_hor-1:
    #                         dw = True
    #                     else:
    #                         dw = False

    #                 # print(goal)
    #                 # print("reward:",reward['reward'])
    #                 if i>0:
    #                     replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)
    #                     # print("obs:", s_all)
    #                     # print("next_obs:",s_all1)
    #                     # print("action:",a)
    #                     # print("reward:",reward['reward'])

    #                     s_all = s_all1
    #                     total_steps+=1

    #                     if replay_buffer.count == args.batch_size:
    #                         ppo_agent.update(replay_buffer, total_steps)
    #                         replay_buffer.count = 0

    #                 prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

    #                 if i==0:
    #                     if self.meta:
    #                         self.adapt_buffer['obs'].append(obs)
    #                         self.adapt_buffer['act'].append(A[t])

    #                 # if log_data:
    #                 #     obs1 = self.env.get_uav_obs()[0]
    #                 #     obs2 = self.env.get_uav_obs()[1]
    #                 #     obs1_l.append(obs1)
    #                 #     obs2_l.append(obs2)
    #                 O.append(obs)
    #                 reward_sum += reward['reward']
    #                 rewards.append(reward['reward'])
    #                 errorx.append(reward['abs_error_x'])
    #                 errory.append(reward['abs_error_y'])
    #                 errorz.append(reward['abs_error_z'])
    #                 if done:
    #                     break
                
    #             episode_n+=1
    #             repeat_eval = False

    #         print("Average action selection time: ", np.mean(times))
    #         print("Rollout length: ", len(A))
    #         print("Rollout reward: ", reward_sum)

    #     self.logger.close()

    def run_experiment_ppo(self,path):
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        train_iters = 1600

        #ppo logger
        log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_pure_PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(log_path_ppo, exist_ok=True)
        logger_ppo = SummaryWriter(logdir=log_path_ppo) # used for tensorboard

        args = DotMap()
        # if not use image, change dimension
        args.state_dim = 24
        args.action_dim = 3
        args.max_action = 1.0
        args.batch_size = 2048
        args.mini_batch_size = 64
        args.max_train_steps = train_iters * self.task_hor
        args.lr_a  = 3e-4
        args.lr_c = 3e-4
        args.gamma = 0.99
        args.lamda = 0.95
        args.epsilon = 0.2
        args.K_epochs = 10
        args.entropy_coef  = 0.005#0.01
        args.use_grad_clip = True
        args.use_lr_decay = True
        args.use_adv_norm = True
        # args.horizon = 1
        args.evaluate_s = 0

        #runing rollouts for collection samples for meta training
        wind_condition_x = 0.0
        wind_condition_y = 0.0
        L = 0.6

        replay_buffer = ReplayBuffer(args)
        ppo_agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)

        total_steps = 0
        evaluate_frequency = 20
        reward_index = 0
        reward_index_log = 0 
        reward_repeat = []
        episode_n = 0
        repeat_eval = False
        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        for i in range(train_iters):
            self.env.wind_controller_x.publish(0.0)
            self.env.wind_controller_y.publish(0.0)
            self.env.set_L(L)
        
            times, rewards = [], []
            
            o1, goal, g_s = self.env.reset()
            O, A, reward_sum, done = [o1], [], 0, False
            # reward_scaling.reset()

            past_corrected_goals = [np.zeros((1,3))]
            past_traj = [np.zeros((1,3))]
            past_traj_error = [np.zeros((1,3))]

            #normalize goal input in observation
            O_t = torch.from_numpy(o1).cuda().float()
            g_normalize_s = g_s.copy()
            g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
            g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
            g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
            g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

            # print("old:",past_traj[-1])
            g_normalize_past = past_traj[-1]
            g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
            g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
            g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
            g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

            g_normalize_past_c = past_corrected_goals[-1]
            g_normalize_past_c[:,0] = g_normalize_past_c[:,0]
            g_normalize_past_c[:,1] = g_normalize_past_c[:,1]
            g_normalize_past_c[:,2] = g_normalize_past_c[:,2]
            g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

            e_normalize_past = past_traj_error[-1]
            e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

            s_all = torch.cat((O_t[None],g_current[:,:3],g_old[:,:3],g_old_correct,e_old),dim=1)
            assert s_all.shape[1]==12+3+3+3+3

            self.policy.reset()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                self.env.wind_controller_x.publish(wind_condition_x)
                self.env.wind_controller_y.publish(wind_condition_y)
                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    
                    a = ppo_agent.evaluate(s_all) 

                    past_corrected_goals.append(a.reshape(1,3).copy())
                    # print(g_s)
                    past_traj.append(g_s.copy())
                    action=a
                    # print(store_top_s)
                    A.append(action)
                    times.append(time.time() - start)

                    obs, reward, done, (goal, g_s) = self.env.step(A[t])

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    if done==1:
                        reward['reward']-=20
                    # reward['reward'] = reward_scaling(reward['reward'])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                    g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                    # print("old1:",past_traj[-1])
                    g_normalize_past1 = past_traj[-1]
                    g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                    g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                    g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                    g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                    g_normalize_past_c1 = past_corrected_goals[-1]
                    g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]
                    g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]
                    g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]
                    g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                    e_normalize_past1 = past_traj_error[-1]
                    e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                    s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                    assert s_all1.shape[1]==12+3+3+3+3

                    s_all = s_all1

                    # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
        
                    if done:
                        break
                
                if reward_index%3==0 and reward_index!=0:
                    repeat_eval = True

                reward_repeat.append(reward_sum/len(A))
                
                if repeat_eval:
                    reward_sum_av = np.mean(np.array(reward_repeat))
                    reward_sum_std = np.std(np.array(reward_repeat))
                    logger_ppo.add_scalar('Episode reward', reward_sum_av, reward_index_log)
                    logger_ppo.add_scalar('Episode reward std', reward_sum_std, reward_index_log)
                    ppo_agent.save_network(reward_index_log)
                    reward_index_log+=1
                    reward_repeat = []

                reward_index+=1
            else:
                self.env.wind_controller_x.publish(wind_condition_x)
                self.env.wind_controller_y.publish(wind_condition_y)
                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    #add ppo here    
                    # print("i:",self.env.trajectory.get_i())
                    # print("s_all:", s_all)

                    a, a_logprob = ppo_agent.choose_action(s_all) 
                    # print("a:",a)
                
                    #before added to goal, we need transform it to dimension [horizon,1,dim=3]
                    past_corrected_goals.append(a.reshape(1,3).copy())
                    # print(g_s)
                    past_traj.append(g_s.copy())
                    # print(past_traj)

                    action=a
                    # print(store_top_s)
                    A.append(action)
                    times.append(time.time() - start)

                    obs, reward, done, (goal, g_s) = self.env.step(A[t])

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    if done==1:
                        reward['reward']-=20
                    # reward['reward'] = reward_scaling(reward['reward'])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                    g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                    # print("old1:",past_traj[-1])
                    g_normalize_past1 = past_traj[-1]
                    g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                    g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                    g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                    g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                    g_normalize_past_c1 = past_corrected_goals[-1]
                    g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]
                    g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]
                    g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]
                    g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                    e_normalize_past1 = past_traj_error[-1]
                    e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                    s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                    assert s_all1.shape[1]==12+3+3+3+3

                    if done or t == self.task_hor-1:
                        dw = True
                    else:
                        dw = False

                    # print(goal)
                    # print("reward:",reward['reward'])
                   
                    replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)
                    # print("obs:", s_all)
                    # print("next_obs:",s_all1)
                    # print("action:",a)
                    # print("reward:",reward['reward'])

                    s_all = s_all1
                    total_steps+=1

                    if replay_buffer.count == args.batch_size:
                        ppo_agent.update(replay_buffer, total_steps)
                        replay_buffer.count = 0

                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                    
                    if done:
                        break
                
                episode_n+=1
                repeat_eval = False

            print("Average action selection time: ", np.mean(times))
            print("Rollout length: ", len(A))
            print("Rollout reward: ", reward_sum)

        self.logger.close()


    def run_experiment_meta_online1_1(self, path):
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        #222,50,8
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        train_iters = 1600

        #ppo logger
        log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(log_path_ppo, exist_ok=True)
        logger_ppo = SummaryWriter(logdir=log_path_ppo) # used for tensorboard

        args = DotMap()
        # if not use image, change dimension
        args.with_image = False
        args.state_dim = 24
        args.action_dim = 3
        args.max_action = 0.1
        args.batch_size = 2048
        args.mini_batch_size = 64
        args.max_train_steps = train_iters * self.task_hor
        args.lr_a  = 3e-4
        args.lr_c = 3e-4
        args.gamma = 0.99
        args.lamda = 0.95
        args.epsilon = 0.2
        args.K_epochs = 10
        args.entropy_coef  = 0.005#0.01
        args.use_grad_clip = True
        args.use_lr_decay = True
        args.use_adv_norm = True
        # args.horizon = 1
        args.evaluate_s = 0

        #runing rollouts for collection samples for meta training
        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        wind_condition_x = 1.0
        wind_condition_y = 0.0
        L = 0.8

        replay_buffer = ReplayBuffer(args)
        ppo_agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)
        
        if args.with_image:
            act_encoder = Conv_autoencoder().to(TORCH_DEVICE)
            act_encoder.load_encoder()
            act_encoder.eval()

        adapt_size = self.k_spt
        log_data=self.log_sample_data
        data_path=path
        total_steps = 0
        evaluate_frequency = 20
        reward_index = 0
        reward_index_log = 0 
        reward_repeat = []
        episode_n = 0
        repeat_eval = False
        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        for i in range(train_iters):
            if self.meta:
                self.adapt_buffer = dict(obs=[],act=[])
                self.env.wind_controller_x.publish(0.0)
                self.env.wind_controller_y.publish(0.0)
                self.env.set_L(L)
            else:
                self.env.wind_controller_x.publish(0.0)
                self.env.wind_controller_y.publish(0.0)
                self.env.set_L(L)

            if i == 1:
                if self.meta:
                    self.policy.model.save_model(0,model_path=data_path)
        
            times, rewards = [], []
            errorx = []
            errory = []
            errorz = []
            o1, goal, g_s = self.env.reset()
            O, A, reward_sum, done = [o1], [], 0, False
            top_act_seq = []
            prediction_error = []
            # reward_scaling.reset()

            past_corrected_goals = [np.zeros((1,3))]
            past_traj = [np.zeros((1,3))]
            past_traj_error = [np.zeros((1,3))]

            if i>0:
                if args.with_image == True:
                    g_map = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                    g_map_conv = g_map[None][None]
                    d = act_encoder.encoder_forward(g_map_conv)
                    O_t = torch.from_numpy(O[t]).cuda().float()
                    s_all = torch.cat((d,O_t[None]),dim=1)
                    assert s_all.shape[1]==72
                else:
                    #normalize goal input in observation
                    O_t = torch.from_numpy(o1).cuda().float()
                    g_normalize_s = g_s.copy()
                    g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
                    g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
                    g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
                    g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                    # print("old:",past_traj[-1])
                    g_normalize_past = past_traj[-1]
                    g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
                    g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
                    g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
                    g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

                    g_normalize_past_c = past_corrected_goals[-1]
                    g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
                    g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
                    g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
                    g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

                    e_normalize_past = past_traj_error[-1]
                    e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

                    s_all = torch.cat((O_t[None],g_current[:,:3],g_old[:,:3],g_old_correct,e_old),dim=1)
                    assert s_all.shape[1]==12+3+3+3+3

            # if log_data:
            #     obs1_l = []
            #     obs2_l = []
            #     obs1 = self.env.get_uav_obs()[0]
            #     obs2 = self.env.get_uav_obs()[1]
            #     obs1_l.append(obs1)
            #     obs2_l.append(obs2)
            
            if self.meta:
                self.adapt_buffer['obs'].append(o1)
                self.policy.model.fast_adapted_params = None

            self.policy.reset()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                self.env.wind_controller_x.publish(wind_condition_x)
                self.env.wind_controller_y.publish(wind_condition_y)
                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    if self.meta:
                        if len(self.adapt_buffer['act'])>adapt_size:
                            #transform trajectories into adapt dataset
                            new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                            new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                            new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                            new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
            
                            self.policy.model.adapt(new_train_in, new_train_targs)

                    action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
                    
                    a = ppo_agent.evaluate(s_all) 
                    a_correct = a.reshape(1,3).copy()

                    past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
                    # print(g_s)
                    past_traj.append(g_s.copy())
                    action+=a
                    # print(store_top_s)
                    self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
                    self.env.pub_action_sequence1(store_bad_s)
                    A.append(action)
                    top_act_seq.append(act_l)
                    times.append(time.time() - start)

                    obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    if done_f:
                        reward['reward']-=200
                    # reward['reward'] = reward_scaling(reward['reward'])
                    if self.meta:
                        self.adapt_buffer['obs'].append(obs)
                        self.adapt_buffer['act'].append(A[t])

                    if args.with_image == True:
                        if_obs = self.policy.occupancy_predictor(obs)
                        # print(if_obs)
                        # post process reward and done
                        if if_obs:
                            done = 1
                            reward['reward']-=50

                    if args.with_image == True:
                        g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                        g_map_conv1 = g_map1[None][None]
                        d1 = act_encoder.encoder_forward(g_map_conv1)
                        obs_t = torch.from_numpy(obs).cuda().float()
                        s_all1 = torch.cat((d1,obs_t[None]),dim=1)
                        assert(s_all1.shape[1]==72)
                    else:
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                        g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                        g_normalize_past_c1 = past_corrected_goals[-1]
                        g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                        g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                        g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                        g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                        e_normalize_past1 = past_traj_error[-1]
                        e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                        s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                        assert s_all1.shape[1]==12+3+3+3+3

                    s_all = s_all1

                    # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                
                    if done:
                        break
                
                if reward_index%3==0 and reward_index!=0:
                    repeat_eval = True

                reward_repeat.append(reward_sum/len(A))
                
                if repeat_eval:
                    reward_sum_av = np.mean(np.array(reward_repeat))
                    reward_sum_std = np.std(np.array(reward_repeat))
                    logger_ppo.add_scalar('Episode reward', reward_sum_av, reward_index_log)
                    logger_ppo.add_scalar('Episode reward std', reward_sum_std, reward_index_log)
                    ppo_agent.save_network(reward_index_log)
                    reward_index_log+=1
                    reward_repeat = []

                reward_index+=1
            else:
                self.env.wind_controller_x.publish(wind_condition_x)
                self.env.wind_controller_y.publish(wind_condition_y)
                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    if self.meta:
                        if len(self.adapt_buffer['act'])>adapt_size:
                            #transform trajectories into adapt dataset
                            new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                            new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                            new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                            new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
            
                            self.policy.model.adapt(new_train_in, new_train_targs)

                    #add ppo here    
                    # print("i:",self.env.trajectory.get_i())
                    # print("s_all:", s_all)

                    action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
                    if i>0:
                        a, a_logprob = ppo_agent.choose_action(s_all) 
                        # print("a:",a)
                    
                        #before added to goal, we need transform it to dimension [horizon,1,dim=3]
                        a_correct = a.reshape(1,3).copy()

                        past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
                        # print(g_s)
                        past_traj.append(g_s.copy())
                        # print(past_traj)

                        action+=a
                    # print(store_top_s)
                    self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
                    self.env.pub_action_sequence1(store_bad_s)
                    A.append(action)
                    top_act_seq.append(act_l)
                    times.append(time.time() - start)

                    obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    if done_f:
                        reward['reward']-=200
                    # reward['reward'] = reward_scaling(reward['reward'])

                    if i>0:
                        if args.with_image == True:
                            if_obs = self.policy.occupancy_predictor(obs)
                            # print(if_obs)
                            #post process reward and done
                            if if_obs:
                                done = 1
                                reward['reward']-=50

                        if args.with_image == True:
                            g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                            g_map_conv1 = g_map1[None][None]
                            d1 = act_encoder.encoder_forward(g_map_conv1)
                            obs_t = torch.from_numpy(obs).cuda().float()
                            s_all1 = torch.cat((d1,obs_t[None]),dim=1)
                            assert(s_all1.shape[1]==72)
                        else:
                            O_t1 = torch.from_numpy(obs).cuda().float()
                            g_normalize_s1 = g_s.copy()
                            g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                            g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                            g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                            g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                            # print("old1:",past_traj[-1])
                            g_normalize_past1 = past_traj[-1]
                            g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                            g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                            g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                            g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                            g_normalize_past_c1 = past_corrected_goals[-1]
                            g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                            g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                            g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                            g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                            e_normalize_past1 = past_traj_error[-1]
                            e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()
                            # print(e_old1)

                            s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                            assert s_all1.shape[1]==12+3+3+3+3

                        if done or t == self.task_hor-1:
                            dw = True
                        else:
                            dw = False

                    # print(goal)
                    # print("reward:",reward['reward'])
                    if i>0:
                        replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)
                        # print("obs:", s_all)
                        # print("next_obs:",s_all1)
                        # print("action:",a)
                        # print("reward:",reward['reward'])

                        s_all = s_all1
                        total_steps+=1

                        if replay_buffer.count == args.batch_size:
                            ppo_agent.update(replay_buffer, total_steps)
                            replay_buffer.count = 0

                    prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

                    
                    if self.meta:
                        self.adapt_buffer['obs'].append(obs)
                        self.adapt_buffer['act'].append(A[t])

                    # if log_data:
                    #     obs1 = self.env.get_uav_obs()[0]
                    #     obs2 = self.env.get_uav_obs()[1]
                    #     obs1_l.append(obs1)
                    #     obs2_l.append(obs2)
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                   
                    if done:
                        break
                
                episode_n+=1
                repeat_eval = False

            print("Average action selection time: ", np.mean(times))
            print("Rollout length: ", len(A))
            print("Rollout reward: ", reward_sum)

        self.logger.close()


    def run_experiment_meta_online1_1all(self, path):
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4

        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        train_iters = 2500

        #ppo logger
        log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(log_path_ppo, exist_ok=True)
        logger_ppo = SummaryWriter(logdir=log_path_ppo) # used for tensorboard

        args = DotMap()
        # if not use image, change dimension
        args.with_image = False
        args.state_dim = 24
        args.action_dim = 3
        args.max_action = 0.1
        args.batch_size = 2048
        args.mini_batch_size = 64
        args.max_train_steps = train_iters * self.task_hor
        args.lr_a  = 3e-4
        args.lr_c = 3e-4
        args.gamma = 0.99
        args.lamda = 0.95
        args.epsilon = 0.2
        args.K_epochs = 10
        args.entropy_coef  = 0.005#0.01
        args.use_grad_clip = True
        args.use_lr_decay = True
        args.use_adv_norm = True
        # args.horizon = 1
        args.evaluate_s = 0

        #runing rollouts for collection samples for meta training
        wind_condition_x = 0.0
        wind_condition_y = 0.0
        L = 0.6

        wind_condition_x1 = 1.0
        wind_condition_y1 = 0.0
        L1 = 0.8

        wind_condition_x2 = 0.6
        wind_condition_y2 = 0.0
        L2 = 1.4

        replay_buffer = ReplayBuffer(args)
        ppo_agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)
        
        if args.with_image:
            act_encoder = Conv_autoencoder().to(TORCH_DEVICE)
            act_encoder.load_encoder()
            act_encoder.eval()

        adapt_size = self.k_spt
        log_data=self.log_sample_data
        data_path=path
        total_steps = 0
        evaluate_frequency = 20
        reward_index = 0
        reward_index_log = 0 
        reward_repeat = []
        episode_n = 0
        repeat_eval = False
        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        for i in range(train_iters):
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                if reward_index%3==0:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L)
                if reward_index%3==1:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L1)
                if reward_index%3==2:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L2)
            else:
                if episode_n%3==0:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L)
                if episode_n%3==1:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L1)
                if episode_n%3==2:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L2)

            if self.meta:
                self.adapt_buffer = dict(obs=[],act=[])

            if i == 1:
                if self.meta:
                    self.policy.model.save_model(0,model_path=data_path)
        
            times, rewards = [], []
            errorx = []
            errory = []
            errorz = []
            # self.env.set_L(L)
            o1, goal, g_s = self.env.reset()
            O, A, reward_sum, done = [o1], [], 0, False
            top_act_seq = []
            prediction_error = []
            # reward_scaling.reset()

            past_corrected_goals = [np.zeros((1,3))]
            past_traj = [np.zeros((1,3))]
            past_traj_error = [np.zeros((1,3))]

            if i>0:
                if args.with_image == True:
                    g_map = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                    g_map_conv = g_map[None][None]
                    d = act_encoder.encoder_forward(g_map_conv)
                    O_t = torch.from_numpy(O[t]).cuda().float()
                    s_all = torch.cat((d,O_t[None]),dim=1)
                    assert s_all.shape[1]==72
                else:
                    #normalize goal input in observation
                    O_t = torch.from_numpy(o1).cuda().float()
                    g_normalize_s = g_s.copy()
                    g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
                    g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
                    g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
                    g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                    # print("old:",past_traj[-1])
                    g_normalize_past = past_traj[-1]
                    g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
                    g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
                    g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
                    g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

                    g_normalize_past_c = past_corrected_goals[-1]
                    g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
                    g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
                    g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
                    g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

                    e_normalize_past = past_traj_error[-1]
                    e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

                    s_all = torch.cat((O_t[None],g_current[:,:3],g_old[:,:3],g_old_correct,e_old),dim=1)
                    assert s_all.shape[1]==12+3+3+3+3

            # if log_data:
            #     obs1_l = []
            #     obs2_l = []
            #     obs1 = self.env.get_uav_obs()[0]
            #     obs2 = self.env.get_uav_obs()[1]
            #     obs1_l.append(obs1)
            #     obs2_l.append(obs2)
            
            if self.meta:
                self.adapt_buffer['obs'].append(o1)
                self.policy.model.fast_adapted_params = None

            self.policy.reset()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                if reward_index%3==0:
                    self.env.wind_controller_x.publish(wind_condition_x)
                    self.env.wind_controller_y.publish(wind_condition_y)
                    self.policy.set_obs(1)
                if reward_index%3==1:
                    self.env.wind_controller_x.publish(wind_condition_x1)
                    self.env.wind_controller_y.publish(wind_condition_y1)
                    self.policy.set_obs(2)
                if reward_index%3==2:
                    self.env.wind_controller_x.publish(wind_condition_x2)
                    self.env.wind_controller_y.publish(wind_condition_y2)
                    self.policy.set_obs(3)

                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()

                    if self.meta:
                        if len(self.adapt_buffer['act'])>adapt_size:
                            #transform trajectories into adapt dataset
                            new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                            new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                            new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                            new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
            
                            self.policy.model.adapt(new_train_in, new_train_targs)

                    action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
                    
                    a = ppo_agent.evaluate(s_all) 
                    a_correct = a.reshape(1,3).copy()

                    past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
                    # print(g_s)
                    past_traj.append(g_s.copy())
                    action+=a
                    # print(store_top_s)
                    self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
                    self.env.pub_action_sequence1(store_bad_s)
                    A.append(action)
                    top_act_seq.append(act_l)
                    times.append(time.time() - start)

                    obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    # reward['reward'] = reward_scaling(reward['reward'])
                    if self.meta:
                        self.adapt_buffer['obs'].append(obs)
                        self.adapt_buffer['act'].append(A[t])

                    if args.with_image == True:
                        if_obs = self.policy.occupancy_predictor(obs)
                        # print(if_obs)
                        #post process reward and done
                        if if_obs:
                            done = 1
                            reward['reward']-=50

                    if args.with_image == True:
                        g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                        g_map_conv1 = g_map1[None][None]
                        d1 = act_encoder.encoder_forward(g_map_conv1)
                        obs_t = torch.from_numpy(obs).cuda().float()
                        s_all1 = torch.cat((d1,obs_t[None]),dim=1)
                        assert(s_all1.shape[1]==72)
                    else:
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                        g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                        g_normalize_past_c1 = past_corrected_goals[-1]
                        g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                        g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                        g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                        g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                        e_normalize_past1 = past_traj_error[-1]
                        e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                        s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                        assert s_all1.shape[1]==12+3+3+3+3

                    s_all = s_all1

                    # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                    errorx.append(reward['abs_error_x'])
                    errory.append(reward['abs_error_y'])
                    errorz.append(reward['abs_error_z'])
                    if done:
                        break
                
                if reward_index%3==0 and reward_index!=0:
                    repeat_eval = True

                reward_repeat.append(reward_sum/len(A))
                
                if repeat_eval:
                    reward_sum_av = np.mean(np.array(reward_repeat))
                    reward_sum_std = np.std(np.array(reward_repeat))
                    logger_ppo.add_scalar('Episode reward', reward_sum_av, reward_index_log)
                    logger_ppo.add_scalar('Episode reward std', reward_sum_std, reward_index_log)
                    ppo_agent.save_network(reward_index_log)
                    reward_index_log+=1
                    reward_repeat = []

                reward_index+=1
            else:
                if episode_n%3==0:
                    self.env.wind_controller_x.publish(wind_condition_x)
                    self.env.wind_controller_y.publish(wind_condition_y)
                    self.policy.set_obs(1)
        
                if episode_n%3==1:
                    self.env.wind_controller_x.publish(wind_condition_x1)
                    self.env.wind_controller_y.publish(wind_condition_y1)
                    self.policy.set_obs(2)

                if episode_n%3==2:
                    self.env.wind_controller_x.publish(wind_condition_x2)
                    self.env.wind_controller_y.publish(wind_condition_y2)
                    self.policy.set_obs(3)

                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    if self.meta:
                        if len(self.adapt_buffer['act'])>adapt_size:
                            #transform trajectories into adapt dataset
                            new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                            new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                            new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                            new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
            
                            self.policy.model.adapt(new_train_in, new_train_targs)

                    #add ppo here    
                    # print("i:",self.env.trajectory.get_i())
                    # print("s_all:", s_all)

                    action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
                    if i>0:
                        a, a_logprob = ppo_agent.choose_action(s_all) 
                        # print("a:",a)
                    
                        #before added to goal, we need transform it to dimension [horizon,1,dim=3]
                        a_correct = a.reshape(1,3).copy()

                        past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
                        # print(g_s)
                        past_traj.append(g_s.copy())
                        # print(past_traj)

                        action+=a
                    # print(store_top_s)
                    self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
                    self.env.pub_action_sequence1(store_bad_s)
                    A.append(action)
                    top_act_seq.append(act_l)
                    times.append(time.time() - start)

                    obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    # reward['reward'] = reward_scaling(reward['reward'])

                    if i>0:
                        if args.with_image == True:
                            if_obs = self.policy.occupancy_predictor(obs)
                            # print(if_obs)
                            #post process reward and done
                            if if_obs:
                                done = 1
                                reward['reward']-=50

                        if args.with_image == True:
                            g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                            g_map_conv1 = g_map1[None][None]
                            d1 = act_encoder.encoder_forward(g_map_conv1)
                            obs_t = torch.from_numpy(obs).cuda().float()
                            s_all1 = torch.cat((d1,obs_t[None]),dim=1)
                            assert(s_all1.shape[1]==72)
                        else:
                            O_t1 = torch.from_numpy(obs).cuda().float()
                            g_normalize_s1 = g_s.copy()
                            g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                            g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                            g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                            g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                            # print("old1:",past_traj[-1])
                            g_normalize_past1 = past_traj[-1]
                            g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                            g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                            g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                            g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                            g_normalize_past_c1 = past_corrected_goals[-1]
                            g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                            g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                            g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                            g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                            e_normalize_past1 = past_traj_error[-1]
                            e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                            s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                            assert s_all1.shape[1]==12+3+3+3+3

                        if done or t == self.task_hor-1:
                            dw = True
                        else:
                            dw = False

                    # print(goal)
                    # print("reward:",reward['reward'])
                    if i>0:
                        replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)
                        # print("obs:", s_all)
                        # print("next_obs:",s_all1)
                        # print("action:",a)
                        # print("reward:",reward['reward'])

                        s_all = s_all1
                        total_steps+=1

                        if replay_buffer.count == args.batch_size:
                            ppo_agent.update(replay_buffer, total_steps)
                            replay_buffer.count = 0

                    prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

                    if self.meta:
                        self.adapt_buffer['obs'].append(obs)
                        self.adapt_buffer['act'].append(A[t])

                    # if log_data:
                    #     obs1 = self.env.get_uav_obs()[0]
                    #     obs2 = self.env.get_uav_obs()[1]
                    #     obs1_l.append(obs1)
                    #     obs2_l.append(obs2)
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                    errorx.append(reward['abs_error_x'])
                    errory.append(reward['abs_error_y'])
                    errorz.append(reward['abs_error_z'])
                    if done:
                        break
                
                episode_n+=1
                repeat_eval = False

            print("Average action selection time: ", np.mean(times))
            print("Rollout length: ", len(A))
            print("Rollout reward: ", reward_sum)

        self.logger.close()

    def run_experiment_only_offline_test(self):
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)

        self.policy.offline_test(self.logger)


    def run_experiment_only_offline(self):
        if self.meta:
            torch.manual_seed(222)
            torch.cuda.manual_seed_all(222)
            np.random.seed(222)

            self.policy.offline_train(self.logger) 
        else:
            wind_condition_x = 0.0
            wind_condition_y = 0.0
            mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x{0}_y{1}_2agents_dt_0.3.mat".format(wind_condition_x,wind_condition_y))
            #this part has to be modified later
            train_obs = mat_contents['obs'] #[N+1,4] or [N+1,12]
            train_acs = mat_contents['acs'] #[N,2] or [N,3]

            #samples [episode, steps,n]
            self.policy.offline_train_MBRL(train_obs, train_acs,self.logger)
            self.logger.close()

    ## defined offline embedding training
    def run_experiment_embedding_meta(self,training,logger, path=None):

        if training:
            self.policy.embedding_meta_train(logger)
        else:
            # seed 222,50,8,1,20,45,60,104,165,200
            torch.manual_seed(200)
            torch.cuda.manual_seed_all(200)
            np.random.seed(200)
            # If already trained model available, just load it
            self.policy.load_embedding_model()

            #runing rollouts for collection samples for meta training
            #runing rollouts for collection samples for meta training
        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
            wind_condition_x = 0.6
            wind_condition_y = 0.0
            L = 1.4

            sample = self.agent.embedding_sample(self.task_hor, self.policy, wind_condition_x,wind_condition_y, L, log_data=self.log_sample_data, data_path=path)
            #if path_length of path i is less than k_spt+k_qry, we have to resample it
            
            if self.log_sample_data:
                print("start logging")
                savemat(path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
                savemat(path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
                savemat(path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
                savemat(path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
                savemat(path+'/storeObs.mat', mdict={'arr': sample["obs"]})
                data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
                savemat(path+'/store_destraj.mat', mdict={'arr': data['arr']})

            # if self.log_sample_data:
            #     savemat(path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            #     savemat(path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            #     savemat(path+'/storeAcs.mat', mdict={'arr': sample["ac"]})
            # # self.logger.add_scalar('Reward', sample["reward_sum"], 0)
            
            # self.logger.close()
              #plot results
            # data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            # figure1 = plt.figure('figure1')
            # plt.plot(sample["rewards"])
            # plt.plot(sample["error_x"])
            # plt.plot(sample["error_y"])
            # plt.plot(sample["error_z"])
            # plt.gca().legend(('Euclidean', 'x', 'y', 'z'))
            # plt.title('position error')
            # plt.xlabel('t')
            # plt.ylabel('error(m)')

            # figure2 = plt.figure('figure2')
            # ax = plt.axes(projection='3d')
            # ax.plot3D(sample["obs"][:,6]*4,sample["obs"][:,7]*4,sample["obs"][:,8]*2, 'gray')
            # ax.plot3D(data['arr'][:,0],data['arr'][:,1],data['arr'][:,2], 'r--')
            # plt.title('3d trajectory')
            # ax.set_xlabel('x(m)')
            # ax.set_ylabel('y(m)')
            # ax.set_zlabel('z(m)')

            # plt.show()

    def run_experiment_variation_inference(self,training):
        if training:
            self.policy.variation_inference_train(self.logger)
        else:
            # seed 222,50,8
            # If already trained model available, just load it
            self.policy.load_VI_model()

            #runing rollouts for collection samples for meta training
            wind_condition_x = 0.0
            wind_condition_y = 0.0
            L = 0.6

            sample = self.agent.VI_sample(self.task_hor, self.policy, wind_condition_x,wind_condition_y, L)
            #if path_length of path i is less than k_spt+k_qry, we have to resample it
            
            # if self.log_sample_data:
            #     savemat(path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            #     savemat(path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            #     savemat(path+'/storeAcs.mat', mdict={'arr': sample["ac"]})
            # # self.logger.add_scalar('Reward', sample["reward_sum"], 0)
            
            # self.logger.close()
              #plot results
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            figure1 = plt.figure('figure1')
            plt.plot(sample["rewards"])
            plt.plot(sample["error_x"])
            plt.plot(sample["error_y"])
            plt.plot(sample["error_z"])
            plt.gca().legend(('Euclidean', 'x', 'y', 'z'))
            plt.title('position error')
            plt.xlabel('t')
            plt.ylabel('error(m)')

            figure2 = plt.figure('figure2')
            ax = plt.axes(projection='3d')
            ax.plot3D(sample["obs"][:,6]*4,sample["obs"][:,7]*4,sample["obs"][:,8]*2, 'gray')
            ax.plot3D(data['arr'][:,0],data['arr'][:,1],data['arr'][:,2], 'r--')
            plt.title('3d trajectory')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')

            plt.show()


    def run_experiment_meta_online1_evaluation(self, path): 
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # seed 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        eval_iters = 1

        #ppo logger
        log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_PPO_model_eval",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(log_path_ppo, exist_ok=True)
        logger_ppo = SummaryWriter(logdir=log_path_ppo) # used for tensorboard

        args = DotMap()
        # if not use image, change dimension
        args.with_image = False
        args.state_dim = 24
        args.action_dim = 3
        args.max_action = 0.1
        args.batch_size = 2048
        args.mini_batch_size = 64
        args.max_train_steps = eval_iters * self.task_hor
        args.lr_a  = 3e-4
        args.lr_c = 3e-4
        args.gamma = 0.99
        args.lamda = 0.95
        args.epsilon = 0.2
        args.K_epochs = 10
        args.entropy_coef  = 0.005
        args.use_grad_clip = True
        args.use_lr_decay = True
        args.use_adv_norm = True
        # args.horizon = 1
        args.evaluate_s = 1
        adapt_size = self.k_spt

        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        #runing rollouts for collection samples for meta training
        wind_condition_x = 1.0
        wind_condition_y = 0.0
        L = 0.8

        ppo_agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)
        
        if args.with_image:
            act_encoder = Conv_autoencoder().to(TORCH_DEVICE)
            act_encoder.load_encoder()
            act_encoder.eval()

        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        if self.meta:
            self.adapt_buffer = dict(obs=[],act=[])
            self.env.wind_controller_x.publish(0.0)
            self.env.wind_controller_y.publish(0.0)
            self.env.set_L(L)
        else:
            self.env.wind_controller_x.publish(0.0)
            self.env.wind_controller_y.publish(0.0)
            self.env.set_L(L)
    
        times, rewards = [], []
        errorx = []
        errory = []
        errorz = []
        A_ori = []
        A_c = []
        
        o1, goal, g_s = self.env.reset()
        O, A, reward_sum, done = [o1], [], 0, False
        top_act_seq = []
        prediction_error = []
        # reward_scaling.reset()
        obs1_l = []
        obs2_l = []
        obs1 = self.env.get_uav_obs()[0]
        obs2 = self.env.get_uav_obs()[1]
        obs1_l.append(obs1)
        obs2_l.append(obs2)
        
        past_corrected_goals = [np.zeros((1,3))]
        past_traj = [np.zeros((1,15))]
        past_traj_error = [np.zeros((1,3))]

        if args.with_image == True:
            g_map = torch.from_numpy(self.env.get_depth_map()).cuda().float()
            g_map_conv = g_map[None][None]
            d = act_encoder.encoder_forward(g_map_conv)
            O_t = torch.from_numpy(O[t]).cuda().float()
            s_all = torch.cat((d,O_t[None]),dim=1)
            assert s_all.shape[1]==72
        else:
            #normalize goal input in observation
            O_t = torch.from_numpy(o1).cuda().float()
            g_normalize_s = g_s.copy()
            g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
            g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
            g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
            g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

            # print("old:",past_traj[-1])
            g_normalize_past = past_traj[-1]
            g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
            g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
            g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
            g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

            g_normalize_past_c = past_corrected_goals[-1]
            g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
            g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
            g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
            g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

            e_normalize_past = past_traj_error[-1]
            e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

            s_all = torch.cat((O_t[None],g_current[:,:3],g_old[:,:3],g_old_correct,e_old),dim=1)
            assert s_all.shape[1]==12+3+3+3+3

        if self.meta:
            self.adapt_buffer['obs'].append(o1)
            self.policy.model.fast_adapted_params = None

        self.policy.reset()

        self.env.wind_controller_x.publish(wind_condition_x)
        self.env.wind_controller_y.publish(wind_condition_y)
        self.policy.set_obs(2)
    
        for t in range(self.task_hor):
            # if t>100:
            #     self.env.wind_controller_x.publish(0.2)
            #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
            #     self.env.set_L(0.8)
            # break
            start = time.time()
            if self.meta:
                if len(self.adapt_buffer['act'])>adapt_size:
                    #transform trajectories into adapt dataset
                    new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                    new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                    new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                    new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
    
                    self.policy.model.adapt(new_train_in, new_train_targs)

            action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
            a = ppo_agent.evaluate(s_all) 
            a_correct = a.reshape(1,3).copy()
            A_c.append(a_correct)
            A_ori.append(action.reshape(1,3).copy())

            past_corrected_goals.append(a_correct+action.reshape(1,3).copy())
                # print(g_s)
            past_traj.append(g_s.copy())
            action+=a
            # print(store_top_s)

            self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
            self.env.pub_action_sequence1(store_bad_s)
            A.append(action)
            top_act_seq.append(act_l)
            times.append(time.time() - start)

            obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])

            new_error_traj = np.zeros((1,3))
            new_error_traj[:,0] = reward['error_x']
            new_error_traj[:,1] = reward['error_y']
            new_error_traj[:,2] = reward['error_z']
            past_traj_error.append(new_error_traj)

            if self.meta:
                self.adapt_buffer['obs'].append(obs)
                self.adapt_buffer['act'].append(A[t])
            
            #reward process
            # reward['reward'] = reward['reward']
            # reward['reward'] = reward_scaling(reward['reward'])

            obs1 = self.env.get_uav_obs()[0]
            obs2 = self.env.get_uav_obs()[1]
            obs1_l.append(obs1)
            obs2_l.append(obs2)

            if args.with_image == True:
                if_obs = self.policy.occupancy_predictor(obs)
                # print(if_obs)
                #post process reward and done
                if if_obs:
                    done = 1
                    reward['reward']-=50

            if args.with_image == True:
                g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                g_map_conv1 = g_map1[None][None]
                d1 = act_encoder.encoder_forward(g_map_conv1)
                obs_t = torch.from_numpy(obs).cuda().float()
                s_all1 = torch.cat((d1,obs_t[None]),dim=1)
                assert(s_all1.shape[1]==72)
            else:
                O_t1 = torch.from_numpy(obs).cuda().float()
                g_normalize_s1 = g_s.copy()
                g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                # print("old1:",past_traj[-1])
                g_normalize_past1 = past_traj[-1]
                g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                g_normalize_past_c1 = past_corrected_goals[-1]
                g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                e_normalize_past1 = past_traj_error[-1]
                e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                s_all1 = torch.cat((O_t1[None],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1),dim=1)
                assert s_all1.shape[1]==12+3+3+3+3

                s_all = s_all1

            # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
            O.append(obs)
            reward_sum += reward['reward']
            rewards.append(reward['reward'])
            errorx.append(reward['abs_error_x'])
            errory.append(reward['abs_error_y'])
            errorz.append(reward['abs_error_z'])
            if done:
                break
        
        
        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))
        print("Rollout reward: ", reward_sum)

        # savemat(data_path+'/storeElites.mat', mdict={'arr': top_act_seq})
        savemat(path+'/storeUAV1.mat', mdict={'arr': obs1_l})
        savemat(path+'/storeUAV2.mat', mdict={'arr': obs2_l})
        
        sample = {
            "obs": np.array(O),
            "ac": np.array(A),
            "ac_ori":np.array(A_ori),
            "ac_c":np.array(A_c),
            "reward_sum": reward_sum,
            "reward_average":-reward_sum/len(A),
            "rewards": np.array(rewards),
            "error_x": np.array(errorx),
            "error_y": np.array(errory),
            "error_z": np.array(errorz),
            "prediction_error": np.array(prediction_error)
        }

        if self.log_sample_data:
            savemat(path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            savemat(path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
            savemat(path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
            savemat(path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
            savemat(path+'/store_ac.mat', mdict={'arr': sample["ac"]})
            savemat(path+'/store_ac_ori.mat', mdict={'arr': sample["ac_ori"]})
            savemat(path+'/store_ac_c.mat', mdict={'arr': sample["ac_c"]})
            savemat(path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            savemat(path+'/store_destraj.mat', mdict={'arr': data['arr']})

        logger_ppo.close()


    def run_experiment_meta_online_im(self, path):
        """
           Correct actions with image
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """

        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        train_iters = 1600

        #ppo logger
        log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(log_path_ppo, exist_ok=True)
        logger_ppo = SummaryWriter(logdir=log_path_ppo) # used for tensorboard

        args = DotMap()
        # if not use image, change dimension
        args.with_image = True
        args.state_dim = 115
        args.action_dim = self.horizon*2
        args.max_action = 1.0
        args.batch_size = 2048
        args.mini_batch_size = 64
        args.max_train_steps = train_iters * self.task_hor
        args.lr_a  = 3e-4
        args.lr_c = 3e-4
        args.gamma = 0.99
        args.lamda = 0.95
        args.epsilon = 0.2
        args.K_epochs = 10
        args.entropy_coef  = 0.01#0.01
        args.use_grad_clip = True
        args.use_lr_decay = True
        args.use_adv_norm = True
        # args.horizon = 1
        args.evaluate_s = 0

        #runing rollouts for collection samples for meta training
        wind_condition_x = 0.0
        wind_condition_y = 0.0
        L = 0.6

        replay_buffer = ReplayBuffer(args)
        ppo_agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)
        
        if args.with_image:
            act_encoder = Conv_autoencoder().to(TORCH_DEVICE)
            act_encoder.load_encoder()
            act_encoder.eval()

        adapt_size = self.k_spt
        log_data=self.log_sample_data
        data_path=path
        total_steps = 0
        evaluate_frequency = 20
        reward_index = 0
        reward_index_log = 0 
        reward_repeat = []
        episode_n = 0
        repeat_eval = False
        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        if self.meta:
            self.adapt_buffer = dict(obs=[],act=[])
            self.env.wind_controller_x.publish(wind_condition_x)
            self.env.wind_controller_y.publish(wind_condition_y)
        else:
            self.env.wind_controller_x.publish(wind_condition_x)
            self.env.wind_controller_y.publish(wind_condition_y)

        for i in range(train_iters):

            if i == 1:
                if self.meta:
                    self.policy.model.save_model(0,model_path=data_path)
        
            times, rewards = [], []
            errorx = []
            errory = []
            errorz = []
            self.env.set_L(L)
            o1, goal, g_s = self.env.reset()
            O, A, reward_sum, done = [o1], [], 0, False
            top_act_seq = []
            prediction_error = []
            # reward_scaling.reset()

            past_corrected_goals = [np.zeros((5,2))]
            past_traj = [np.zeros((5,3))]
            past_traj_error = [np.zeros((1,3))]

            if i>0:
                if args.with_image == True:
                    O_t = torch.from_numpy(o1).cuda().float()
                    g_normalize_s = g_s.copy()
                    g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
                    g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
                    g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
                    g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                    # print("old:",past_traj[-1])
                    g_normalize_past = past_traj[-1]
                    g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
                    g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
                    g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
                    g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

                    g_normalize_past_c = past_corrected_goals[-1]
                    g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/4.0
                    g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
                    g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

                    e_normalize_past = past_traj_error[-1]
                    e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

                    g_map = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                    g_map_conv = g_map[None][None]
                    d = act_encoder.encoder_forward(g_map_conv)

                    s_all = torch.cat((O_t[None],g_current,g_old,g_old_correct,e_old,d),dim=1)

                    assert s_all.shape[1]==12+15+15+10+3+60
                else:
                    #normalize goal input in observation
                    O_t = torch.from_numpy(o1).cuda().float()
                    g_normalize_s = g_s.copy()
                    g_normalize_s[:,0] = g_normalize_s[:,0]/4.0
                    g_normalize_s[:,1] = g_normalize_s[:,1]/2.0
                    g_normalize_s[:,2] = g_normalize_s[:,2]/2.0
                    g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                    # print("old:",past_traj[-1])
                    g_normalize_past = past_traj[-1]
                    g_normalize_past[:,0] = g_normalize_past[:,0]/4.0
                    g_normalize_past[:,1] = g_normalize_past[:,1]/2.0
                    g_normalize_past[:,2] = g_normalize_past[:,2]/2.0
                    g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

                    g_normalize_past_c = past_corrected_goals[-1]
                    g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
                    g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
                    g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
                    g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

                    e_normalize_past = past_traj_error[-1]
                    e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

                    s_all = torch.cat((O_t[None],g_current,g_old,g_old_correct,e_old),dim=1)
                    assert s_all.shape[1]==12+15+15+3+3

            # if log_data:
            #     obs1_l = []
            #     obs2_l = []
            #     obs1 = self.env.get_uav_obs()[0]
            #     obs2 = self.env.get_uav_obs()[1]
            #     obs1_l.append(obs1)
            #     obs2_l.append(obs2)
            if i == 0:
                if self.meta:
                    self.adapt_buffer['obs'].append(o1)
                    self.policy.model.fast_adapted_params = None

            self.policy.reset()
            self.env.set_pos_callback_depth_loop()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()

                    a = ppo_agent.evaluate(s_all) 
                    a_expand = a.reshape(-1,2)[:,None]
                    a_correct = a.reshape(-1,2).copy()

                    past_corrected_goals.append(a_correct+g_s[:,:2].copy())
                    # print(g_s)
                    past_traj.append(g_s.copy())
                    goal_c = goal.clone().detach().cpu().numpy()
                    goal_c[:,:,:2] = goal_c[:,:,:2]+a_expand
                    self.env.pub_corrective_desire_goals(goal_c[:,0,:])
                    # goal_rc = goal_c[:,0,:].copy()
                    # self.env.set_target(goal_rc[0,:])
                    goal_c = torch.from_numpy(goal_c)

                    action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal_c) #[6,5,2] store top s

                    # print(store_top_s)
                    self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
                    self.env.pub_action_sequence1(store_bad_s)
                    A.append(action)
                    top_act_seq.append(act_l)
                    times.append(time.time() - start)

                    obs, reward, done, (goal, g_s) = self.env.step(A[t])
                    self.env.set_pos_callback_depth_loop()

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']**2
                    # reward['reward'] = reward_scaling(reward['reward'])

                    if args.with_image == True:
                        if_obs = self.policy.occupancy_predictor(obs)
                        # print(if_obs)
                        #post process reward and done
                        if if_obs:
                            done = 1
                            reward['reward']-=50

                    if args.with_image == True:
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                        g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                        g_normalize_past_c1 = past_corrected_goals[-1]
                        g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/4.0
                        g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                        g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                        e_normalize_past1 = past_traj_error[-1]
                        e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                        g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                        g_map_conv1 = g_map1[None][None]
                        d1 = act_encoder.encoder_forward(g_map_conv1)

                        s_all1 = torch.cat((O_t1[None],g_current1,g_old1,g_old_correct1,e_old1,d1),dim=1)

                        assert(s_all1.shape[1]==12+15+15+10+3+60)
                    else:
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                        g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                        g_normalize_past_c1 = past_corrected_goals[-1]
                        g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                        g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                        g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                        g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                        e_normalize_past1 = past_traj_error[-1]
                        e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                        s_all1 = torch.cat((O_t1[None],g_current1,g_old1,g_old_correct1,e_old1),dim=1)
                        assert s_all1.shape[1]==12+15+15+3+3

                    s_all = s_all1

                    # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                    errorx.append(reward['abs_error_x'])
                    errory.append(reward['abs_error_y'])
                    errorz.append(reward['abs_error_z'])
                    if done:
                        break
                
                if reward_index%3==0 and reward_index!=0:
                    repeat_eval = True

                reward_repeat.append(reward_sum)
                
                if repeat_eval:
                    reward_sum_av = np.mean(np.array(reward_repeat))
                    logger_ppo.add_scalar('Episode reward', reward_sum_av, reward_index_log)
                    ppo_agent.save_network(reward_index_log)
                    reward_index_log+=1
                    reward_repeat = []

                reward_index+=1
            else:
                for t in range(self.task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    if self.meta:
                        if i == 0:
                            if len(self.adapt_buffer['act'])>adapt_size:
                                #transform trajectories into adapt dataset
                                new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                                new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                                new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                                new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
                
                                self.policy.model.adapt(new_train_in, new_train_targs)

                    #add ppo here    
                    # print("i:",self.env.trajectory.get_i())
                    # print("s_all:", s_all)
                    if i>0:
                        a, a_logprob = ppo_agent.choose_action(s_all) 
                        # print("a:",a)
                        a_expand = a.reshape(-1,2)[:,None]
                        #before added to goal, we need transform it to dimension [horizon,1,dim=3]
                        a_correct = a.reshape(-1,2).copy()

                        past_corrected_goals.append(a_correct+g_s[:,:2].copy())
                        # print(g_s)
                        past_traj.append(g_s.copy())
                        # print(past_traj)
                        goal_c = goal.clone().detach().cpu().numpy()
                        goal_c[:,:,:2] = goal_c[:,:,:2]+a_expand
                        self.env.pub_corrective_desire_goals(goal_c[:,0,:])
                        # goal_rc = goal_c[:,0,:].copy()
                        # self.env.set_target(goal_rc[0,:])
                        goal_c = torch.from_numpy(goal_c)
                        # print("goal:",goal_c)

                    if i>0:
                        action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal_c) #[6,5,2] store top s
                    else:
                        action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal)

                    # print(store_top_s)
                    self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
                    self.env.pub_action_sequence1(store_bad_s)
                    A.append(action)
                    top_act_seq.append(act_l)
                    times.append(time.time() - start)

                    obs, reward, done, (goal, g_s) = self.env.step(A[t])
                    self.env.set_pos_callback_depth_loop()

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)
                    
                    #reward process
                    reward['reward'] = -reward['reward']**2
                    # reward['reward'] = reward_scaling(reward['reward'])

                    if i>0:
                        if args.with_image == True:
                            if_obs = self.policy.occupancy_predictor(obs)
                            # print(if_obs)
                            #post process reward and done
                            if if_obs:
                                done = 1
                                reward['reward']-=50

                        if args.with_image == True:
                            O_t1 = torch.from_numpy(obs).cuda().float()
                            g_normalize_s1 = g_s.copy()
                            g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                            g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                            g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                            g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                            # print("old1:",past_traj[-1])
                            g_normalize_past1 = past_traj[-1]
                            g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                            g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                            g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                            g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                            g_normalize_past_c1 = past_corrected_goals[-1]
                            g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/4.0
                            g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                            g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                            e_normalize_past1 = past_traj_error[-1]
                            e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                            g_map1 = torch.from_numpy(self.env.get_depth_map()).cuda().float()
                            g_map_conv1 = g_map1[None][None]
                            d1 = act_encoder.encoder_forward(g_map_conv1)

                            s_all1 = torch.cat((O_t1[None],g_current1,g_old1,g_old_correct1,e_old1,d1),dim=1)

                            s_all1.shape[1]==12+15+15+10+3+60
                        else:
                            O_t1 = torch.from_numpy(obs).cuda().float()
                            g_normalize_s1 = g_s.copy()
                            g_normalize_s1[:,0] = g_normalize_s1[:,0]/4.0
                            g_normalize_s1[:,1] = g_normalize_s1[:,1]/2.0
                            g_normalize_s1[:,2] = g_normalize_s1[:,2]/2.0
                            g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                            # print("old1:",past_traj[-1])
                            g_normalize_past1 = past_traj[-1]
                            g_normalize_past1[:,0] = g_normalize_past1[:,0]/4.0
                            g_normalize_past1[:,1] = g_normalize_past1[:,1]/2.0
                            g_normalize_past1[:,2] = g_normalize_past1[:,2]/2.0
                            g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                            g_normalize_past_c1 = past_corrected_goals[-1]
                            g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                            g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                            g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                            g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                            e_normalize_past1 = past_traj_error[-1]
                            e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

                            s_all1 = torch.cat((O_t1[None],g_current1,g_old1,g_old_correct1,e_old1),dim=1)
                            assert s_all1.shape[1]==12+15+15+3+3

                        if done or t == self.task_hor-1:
                            dw = True
                        else:
                            dw = False

                    # print(goal)
                    # print("reward:",reward['reward'])
                    if i>0:
                        replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)
                        # print("obs:", s_all)
                        # print("next_obs:",s_all1)
                        # print("action:",a)
                        # print("reward:",reward['reward'])

                        s_all = s_all1
                        total_steps+=1

                        if replay_buffer.count == args.batch_size:
                            ppo_agent.update(replay_buffer, total_steps)
                            replay_buffer.count = 0

                    prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

                    if i==0:
                        if self.meta:
                            self.adapt_buffer['obs'].append(obs)
                            self.adapt_buffer['act'].append(A[t])

                    # if log_data:
                    #     obs1 = self.env.get_uav_obs()[0]
                    #     obs2 = self.env.get_uav_obs()[1]
                    #     obs1_l.append(obs1)
                    #     obs2_l.append(obs2)
                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
                    errorx.append(reward['abs_error_x'])
                    errory.append(reward['abs_error_y'])
                    errorz.append(reward['abs_error_z'])

                    if i==0:
                        if len(A)==50:
                            break
                    if done:
                        break
                
                episode_n+=1
                repeat_eval = False

            print("Average action selection time: ", np.mean(times))
            print("Rollout length: ", len(A))
            print("Rollout reward: ", reward_sum)

        self.logger.close()

