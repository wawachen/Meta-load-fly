from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import savemat
import torch 
import numpy as np
import scipy.io as scio
from common.replay_buffer import ReplayBuffer
from models.ppo import PPO_model
import time

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')


class MBExperiment:
    def __init__(self, env, policy, agent, logger, path):
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
        self.policy = policy
        self.logger = logger
        self.path = path
        self.agent = agent

    def run_experiment_meta_online1_1(self, cfg, is_eval):
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        #222,50,8
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)

        train_iters = cfg.train_iters

        #ppo logger
        logger_ppo = self.logger # used for tensorboard

        #runing rollouts for collection samples for meta training
        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        wind_condition_x = cfg.wind_condition_x
        wind_condition_y = cfg.wind_condition_y
        L = cfg.L

        replay_buffer = ReplayBuffer(cfg.buffer_params)
        
        task_hor = cfg.task_hor
        cfg.ppo_model_params.max_train_steps = train_iters* task_hor
        ppo_agent = PPO_model(cfg.ppo_model_params, logger_ppo, self.path, is_eval).to(TORCH_DEVICE)

        adapt_size = cfg.k_spt
        total_steps = 0
        evaluate_frequency = cfg.evaluate_frequency
        reward_index = 0
        reward_index_log = 0 
        reward_repeat = []
        episode_n = 0
        repeat_eval = False
        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        for i in range(train_iters):
            self.adapt_buffer = dict(obs=[],act=[])
            self.env.wind_controller_x.publish(0.0)
            self.env.wind_controller_y.publish(0.0)
            self.env.set_L(L)

            if i == 1:
                self.policy.model.save_model(0)
        
            times, rewards = [], []
            o1, goal, g_s = self.env.reset()
            O, A, reward_sum, done = [o1], [], 0, False
            top_act_seq = []
            prediction_error = []
            # reward_scaling.reset()

            past_corrected_goals = [np.zeros((1,3))]
            past_traj = [np.zeros((1,3))]
            past_traj_error = [np.zeros((1,3))]

            if i>0:
                #normalize goal input in observation
                O_t = torch.from_numpy(o1).cuda().float()
                g_normalize_s = g_s.copy()
                g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
                g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
                g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
                g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                # print("old:",past_traj[-1])
                g_normalize_past = past_traj[-1]
                g_normalize_past[:,0] = g_normalize_past[:,0]/self.env.max_x
                g_normalize_past[:,1] = g_normalize_past[:,1]/self.env.max_y
                g_normalize_past[:,2] = g_normalize_past[:,2]/self.env.max_z
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
            
            self.adapt_buffer['obs'].append(o1)
            self.policy.model.fast_adapted_params = None

            self.policy.reset()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                self.env.wind_controller_x.publish(wind_condition_x)
                self.env.wind_controller_y.publish(wind_condition_y)
                for t in range(task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    
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
                    
                    self.adapt_buffer['obs'].append(obs)
                    self.adapt_buffer['act'].append(A[t])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                    g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                    # print("old1:",past_traj[-1])
                    g_normalize_past1 = past_traj[-1]
                    g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
                    g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
                    g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
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
                for t in range(task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                
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
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
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

                        if done or t == task_hor-1:
                            dw = True
                        else:
                            dw = False

                    if i>0:
                        # print(replay_buffer.count)
                        replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)
                        s_all = s_all1
                        total_steps+=1

                        if replay_buffer.count == cfg.buffer_params.batch_size:
                            ppo_agent.update(replay_buffer, total_steps)
                            replay_buffer.count = 0

                    prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

                    self.adapt_buffer['obs'].append(obs)
                    self.adapt_buffer['act'].append(A[t])

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


    def run_experiment_meta_online1_1all(self, cfg, is_eval):
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        train_iters = cfg.train_iters

        #ppo logger
        logger_ppo = self.logger # used for tensorboard

        #runing rollouts for collection samples for meta training
        wind_condition_x = cfg.wind_condition_x1
        wind_condition_y = cfg.wind_condition_y1
        L = cfg.L1

        wind_condition_x1 = cfg.wind_condition_x2
        wind_condition_y1 = cfg.wind_condition_y2
        L1 = cfg.L2

        wind_condition_x2 = cfg.wind_condition_x3
        wind_condition_y2 = cfg.wind_condition_y3
        L2 = cfg.L3

        replay_buffer = ReplayBuffer(cfg.buffer_params)

        task_hor = cfg.task_hor
        cfg.ppo_model_params.max_train_steps = train_iters* task_hor
        ppo_agent = PPO_model(cfg.ppo_model_params, logger_ppo, self.path, is_eval).to(TORCH_DEVICE)

        adapt_size = cfg.k_spt
        total_steps = 0
        evaluate_frequency = cfg.evaluate_frequency
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

            self.adapt_buffer = dict(obs=[],act=[])

            if i == 1:
                self.policy.model.save_model(0)
        
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
                #normalize goal input in observation
                O_t = torch.from_numpy(o1).cuda().float()
                g_normalize_s = g_s.copy()
                g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
                g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
                g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
                g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                # print("old:",past_traj[-1])
                g_normalize_past = past_traj[-1]
                g_normalize_past[:,0] = g_normalize_past[:,0]/self.env.max_x
                g_normalize_past[:,1] = g_normalize_past[:,1]/self.env.max_y
                g_normalize_past[:,2] = g_normalize_past[:,2]/self.env.max_z
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
            
            self.adapt_buffer['obs'].append(o1)
            self.policy.model.fast_adapted_params = None

            self.policy.reset()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                if reward_index%3==0:
                    self.env.wind_controller_x.publish(wind_condition_x)
                    self.env.wind_controller_y.publish(wind_condition_y)
                if reward_index%3==1:
                    self.env.wind_controller_x.publish(wind_condition_x1)
                    self.env.wind_controller_y.publish(wind_condition_y1)
                if reward_index%3==2:
                    self.env.wind_controller_x.publish(wind_condition_x2)
                    self.env.wind_controller_y.publish(wind_condition_y2)

                for t in range(task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()

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
                    self.adapt_buffer['obs'].append(obs)
                    self.adapt_buffer['act'].append(A[t])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                    g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                    # print("old1:",past_traj[-1])
                    g_normalize_past1 = past_traj[-1]
                    g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
                    g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
                    g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
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
        
                if episode_n%3==1:
                    self.env.wind_controller_x.publish(wind_condition_x1)
                    self.env.wind_controller_y.publish(wind_condition_y1)

                if episode_n%3==2:
                    self.env.wind_controller_x.publish(wind_condition_x2)
                    self.env.wind_controller_y.publish(wind_condition_y2)

                for t in range(task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
                    if len(self.adapt_buffer['act'])>adapt_size:
                        #transform trajectories into adapt dataset
                        new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                        new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                        new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                        new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
        
                        self.policy.model.adapt(new_train_in, new_train_targs)

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
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
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

                        if done or t == task_hor-1:
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

                        if replay_buffer.count == cfg.buffer_params.batch_size:
                            ppo_agent.update(replay_buffer, total_steps)
                            replay_buffer.count = 0

                    prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

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

    
    def run_experiment_meta_online1_evaluation_full_uav(self, cfg, is_eval): 
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # seed 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        eval_iters = 1

        #ppo logger
        logger_ppo = self.logger # used for tensorboard

        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        #runing rollouts for collection samples for meta training
        wind_condition_x = cfg.wind_condition_x
        wind_condition_y = cfg.wind_condition_y
        L = cfg.L
        task_hor = cfg.task_hor

        adapt_size = cfg.k_spt

        task_num = 4   #testing with obstacles 1,2,3, False means the environment has no obstacles

        if task_num == 1:
            goal_uav1 = torch.tensor([0.3024, 0.0786]).float().cuda()
            goal_uav2 = torch.tensor([-0.294, 0.0662]).float().cuda()
        if task_num == 2:
            goal_uav1 = torch.tensor([0.2164, -0.0198]).float().cuda()
            goal_uav2 = torch.tensor([-0.4932, -0.018]).float().cuda()
        if task_num == 3:
            goal_uav1 = torch.tensor([0.4876, -0.01]).float().cuda()
            goal_uav2 = torch.tensor([-0.5756, -0.0092]).float().cuda()
        if task_num == 4:
            goal_uav1 = torch.tensor([0.2983, 0.0]).float().cuda()
            goal_uav2 = torch.tensor([-0.2983, 0.0]).float().cuda()
        if task_num == 5:
            goal_uav1 = torch.tensor([0.1474, -0.0006]).float().cuda()
            goal_uav2 = torch.tensor([-0.4470, -0.0006]).float().cuda()
        if task_num == 6:
            goal_uav1 = torch.tensor([0.2020, 0.0004]).float().cuda()
            goal_uav2 = torch.tensor([-0.3937, 0.0004]).float().cuda()

        cfg.ppo_model_params.max_train_steps = 1
        ppo_agent = PPO_model(cfg.ppo_model_params, logger_ppo, self.path, is_eval).to(TORCH_DEVICE)
    
        self.adapt_buffer = dict(obs=[],act=[])
        self.env.wind_controller_x.publish(0.0)
        self.env.wind_controller_y.publish(0.0)
        self.env.set_L(L)
    
        times, rewards = [], []
        errorx = []
        errory = []
        errorz = []
        erroruav1 = []
        erroruav2 = []
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
        past_traj = [np.zeros((1,3))]
        past_traj_error = [np.zeros((1, 3))]
        past_traj_error1 = [np.zeros((1, 2))]
        past_traj_error2 = [np.zeros((1, 2))]

        #normalize goal input in observation
        O_t = torch.from_numpy(o1).cuda().float()
        g_normalize_s = g_s.copy()
        g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
        g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
        g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
        g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

        # print("old:",past_traj[-1])
        g_normalize_past = past_traj[-1]
        g_normalize_past[:,0] = g_normalize_past[:,0]/self.env.max_x
        g_normalize_past[:,1] = g_normalize_past[:,1]/self.env.max_y
        g_normalize_past[:,2] = g_normalize_past[:,2]/self.env.max_z
        g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

        g_normalize_past_c = past_corrected_goals[-1]
        g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
        g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
        g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
        g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

        e_normalize_past = past_traj_error[-1]
        e_old = torch.from_numpy(e_normalize_past.reshape(1,-1)).cuda().float()

        e_normalize_past1 = past_traj_error1[-1]
        e_old1 = torch.from_numpy(e_normalize_past1.reshape(1, -1)).cuda().float()
                
        e_normalize_past2 = past_traj_error2[-1]
        e_old2 = torch.from_numpy(e_normalize_past2.reshape(1, -1)).cuda().float()
                
        s_all = torch.cat((O_t[None],goal_uav1+g_current[:,:2], goal_uav2+g_current[:,:2], g_current[:,:3],g_old[:,:3],g_old_correct,e_old,e_old1,e_old2),dim=1)
        assert s_all.shape[1]==12+3+3+3+3+2+2+2+2

        self.adapt_buffer['obs'].append(o1)
        self.policy.model.fast_adapted_params = None

        self.policy.reset()

        self.env.wind_controller_x.publish(wind_condition_x)
        self.env.wind_controller_y.publish(wind_condition_y)
    
        for t in range(task_hor):
            # self.env.wind_controller_x.publish(0.8*np.sin(kx * t))
            # self.env.wind_controller_y.publish(0.8*np.sin(kx * t))
            # if t>100:
            #     self.env.wind_controller_x.publish(0.2)
            #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
            #     self.env.set_L(0.8)
            # break
            start = time.time()
            
            if len(self.adapt_buffer['act'])>adapt_size:
                #transform trajectories into adapt dataset
                new_train_in = np.concatenate([self.policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                new_train_targs = self.policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)

                self.policy.model.adapt(new_train_in, new_train_targs)

            action,act_l,store_top_s,store_bad_s = self.policy.act(O[t], t, goal) #[6,5,2] store top s
            
            start_ppo = time.time()
            a = ppo_agent.evaluate(s_all)
            print("ppo time is: ", time.time()-start_ppo)
            a_correct = a.reshape(1, 3).copy()
            
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

            obs, reward, (done, done_f), (goal, g_s) = self.env.step_obs(A[t], task_num)

            new_error_traj = np.zeros((1,3))
            new_error_traj[:,0] = reward['error_x']
            new_error_traj[:,1] = reward['error_y']
            new_error_traj[:,2] = reward['error_z']
            past_traj_error.append(new_error_traj)

            new_error_traj1 = np.zeros((1,2))
            new_error_traj1[:,0] = reward['error_u1_x']
            new_error_traj1[:,1] = reward['error_u1_y']
            past_traj_error1.append(new_error_traj1)

            new_error_traj2 = np.zeros((1,2))
            new_error_traj2[:,0] = reward['error_u2_x']
            new_error_traj2[:,1] = reward['error_u2_y']
            past_traj_error2.append(new_error_traj2)

            self.adapt_buffer['obs'].append(obs)
            self.adapt_buffer['act'].append(A[t])

            obs1 = self.env.get_uav_obs()[0]
            obs2 = self.env.get_uav_obs()[1]
            obs1_l.append(obs1)
            obs2_l.append(obs2)

            O_t1 = torch.from_numpy(obs).cuda().float()
            g_normalize_s1 = g_s.copy()
            g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
            g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
            g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
            g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

            # print("old1:",past_traj[-1])
            g_normalize_past1 = past_traj[-1]
            g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
            g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
            g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
            g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

            g_normalize_past_c1 = past_corrected_goals[-1]
            g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
            g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
            g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
            g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

            e_normalize_past1 = past_traj_error[-1]
            e_old1 = torch.from_numpy(e_normalize_past1.reshape(1,-1)).cuda().float()

            e_normalize_past2 = past_traj_error1[-1]
            e_old2 = torch.from_numpy(e_normalize_past2.reshape(1, -1)).cuda().float()
                    
            e_normalize_past3 = past_traj_error2[-1]
            e_old3 = torch.from_numpy(e_normalize_past3.reshape(1,-1)).cuda().float()

            s_all1 = torch.cat((O_t1[None],goal_uav1+g_current1[:,:2], goal_uav2+g_current1[:,:2],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1,e_old2,e_old3),dim=1)
            assert s_all1.shape[1]==12+2+2+3+3+3+3+2+2

            s_all = s_all1

            # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
            O.append(obs)
            reward_sum += reward['reward']
            reward_sum += reward['error_uav1']
            reward_sum += reward['error_uav2']

            rewards.append(reward['reward'])
            errorx.append(reward['abs_error_x'])
            errory.append(reward['abs_error_y'])
            errorz.append(reward['abs_error_z'])
          
            erroruav1.append(reward['error_uav1'])
            erroruav2.append(reward['error_uav2'])
            if done:
                break
        
        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))
        print("Rollout reward: ", reward_sum)
        print('UAV1 error: ', np.mean(np.array(erroruav1)))
        print('UAV2 error: ', np.mean(np.array(erroruav2)))
        print('payload error: ', np.mean(np.array(rewards)))

        # savemat(data_path+'/storeElites.mat', mdict={'arr': top_act_seq})
        savemat(self.path+'/storeUAV1.mat', mdict={'arr': obs1_l})
        savemat(self.path+'/storeUAV2.mat', mdict={'arr': obs2_l})
        
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
        "prediction_error": np.array(prediction_error),
        "error_uav1": np.array(erroruav1),
        "error_uav2": np.array(erroruav2)
        }

        if cfg.log_sample_data:
            savemat(self.path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            savemat(self.path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
            savemat(self.path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
            savemat(self.path + '/store_errorz.mat', mdict={'arr': sample["error_z"]})
            if task_num:
                savemat(self.path + '/store_errorUAV1.mat', mdict={'arr': sample['error_uav1']})
                savemat(self.path +'/store_errorUAV2.mat', mdict={'arr': sample['error_uav2']})
            savemat(self.path+'/store_ac.mat', mdict={'arr': sample["ac"]})
            savemat(self.path+'/store_ac_ori.mat', mdict={'arr': sample["ac_ori"]})
            savemat(self.path+'/store_ac_c.mat', mdict={'arr': sample["ac_c"]})
            savemat(self.path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            savemat(self.path+'/store_destraj.mat', mdict={'arr': data['arr']})

        logger_ppo.close()


    def run_experiment_meta_online1_evaluation(self, cfg, is_eval): 
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # seed 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        eval_iters = 1

        #ppo logger
        logger_ppo = self.logger # used for tensorboard

        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        #runing rollouts for collection samples for meta training
        wind_condition_x = cfg.wind_condition_x
        wind_condition_y = cfg.wind_condition_y
        L = cfg.L

        task_hor = cfg.task_hor
        cfg.ppo_model_params.max_train_steps = 1
        ppo_agent = PPO_model(cfg.ppo_model_params, logger_ppo, self.path, is_eval).to(TORCH_DEVICE)

        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        adapt_size = cfg.k_spt
        self.adapt_buffer = dict(obs=[],act=[])
        self.env.wind_controller_x.publish(0.0)
        self.env.wind_controller_y.publish(0.0)
        self.env.set_L(L)
    
        times, rewards = [], []
        errorx = []
        errory = []
        errorz = []
        erroruav1 = []
        erroruav2 = []
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

        #normalize goal input in observation
        O_t = torch.from_numpy(o1).cuda().float()
        g_normalize_s = g_s.copy()
        g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
        g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
        g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
        g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

        # print("old:",past_traj[-1])
        g_normalize_past = past_traj[-1]
        g_normalize_past[:,0] = g_normalize_past[:,0]/self.env.max_x
        g_normalize_past[:,1] = g_normalize_past[:,1]/self.env.max_y
        g_normalize_past[:,2] = g_normalize_past[:,2]/self.env.max_z
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

        self.adapt_buffer['obs'].append(o1)
        self.policy.model.fast_adapted_params = None

        self.policy.reset()

        self.env.wind_controller_x.publish(wind_condition_x)
        self.env.wind_controller_y.publish(wind_condition_y)
    
        for t in range(task_hor):
            # if t>100:
            #     self.env.wind_controller_x.publish(0.2)
            #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
            #     self.env.set_L(0.8)
            # break
            start = time.time()
            
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

            self.adapt_buffer['obs'].append(obs)
            self.adapt_buffer['act'].append(A[t])

            obs1 = self.env.get_uav_obs()[0]
            obs2 = self.env.get_uav_obs()[1]
            obs1_l.append(obs1)
            obs2_l.append(obs2)

            O_t1 = torch.from_numpy(obs).cuda().float()
            g_normalize_s1 = g_s.copy()
            g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
            g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
            g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
            g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

            # print("old1:",past_traj[-1])
            g_normalize_past1 = past_traj[-1]
            g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
            g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
            g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
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

        if cfg.log_sample_data:
            # savemat(data_path+'/storeElites.mat', mdict={'arr': top_act_seq})
            savemat(self.path+'/storeUAV1.mat', mdict={'arr': obs1_l})
            savemat(self.path+'/storeUAV2.mat', mdict={'arr': obs2_l})
            savemat(self.path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            savemat(self.path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
            savemat(self.path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
            savemat(self.path + '/store_errorz.mat', mdict={'arr': sample["error_z"]})
            savemat(self.path+'/store_ac.mat', mdict={'arr': sample["ac"]})
            savemat(self.path+'/store_ac_ori.mat', mdict={'arr': sample["ac_ori"]})
            savemat(self.path+'/store_ac_c.mat', mdict={'arr': sample["ac_c"]})
            savemat(self.path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            savemat(self.path+'/store_destraj.mat', mdict={'arr': data['arr']})

        logger_ppo.close()
    
    def run_experiment_meta_without_online(self, cfg):
        """Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # seed 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)

        #runing rollouts for collection samples for meta training
        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        wind_condition_x = cfg.wind_condition_x
        wind_condition_y = cfg.wind_condition_y
        L = cfg.L

        sample = self.agent.sample(cfg.task_hor, self.policy, wind_condition_x,wind_condition_y, L, adapt_size = cfg.k_spt, log_data=cfg.log_sample_data, data_path=self.path)
        #if path_length of path i is less than k_spt+k_qry, we have to resample it
        
        if cfg.log_sample_data:
            savemat(self.path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            savemat(self.path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
            savemat(self.path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
            savemat(self.path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
            savemat(self.path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            savemat(self.path+'/store_destraj.mat', mdict={'arr': data['arr']})
            # savemat(path+'/storeAcs.mat', mdict={'arr': sample["ac"]})
        
        self.logger.close()

    def run_experiment_meta_online1_1all_full_uav(self, cfg, is_eval):
        """
           Correct actions
           Perform meta experiment.
           we load the offline meta model and without the online training, only one episode adaptation
        """
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        train_iters = cfg.train_iters

        #ppo logger
        logger_ppo = self.logger # used for tensorboard

        #runing rollouts for collection samples for meta training
        wind_condition_x = cfg.wind_condition_x1
        wind_condition_y = cfg.wind_condition_y1
        L = cfg.L1

        wind_condition_x1 = cfg.wind_condition_x2
        wind_condition_y1 = cfg.wind_condition_y2
        L1 = cfg.L2

        wind_condition_x2 = cfg.wind_condition_x3
        wind_condition_y2 = cfg.wind_condition_y3
        L2 = cfg.L3

        replay_buffer = ReplayBuffer(cfg.buffer_params)

        task_hor = cfg.task_hor
        cfg.ppo_model_params.max_train_steps = train_iters* task_hor
        ppo_agent = PPO_model(cfg.ppo_model_params, logger_ppo, self.path, is_eval).to(TORCH_DEVICE)
        
        adapt_size = cfg.k_spt
        total_steps = 0
        evaluate_frequency = cfg.evaluate_frequency
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
                    goal_uav1 = torch.tensor(cfg.uav1_goal1).float().cuda()
                    goal_uav2 = torch.tensor(cfg.uav2_goal1).float().cuda()
                if reward_index%3==1:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L1)
                    goal_uav1 = torch.tensor(cfg.uav1_goal2).float().cuda()
                    goal_uav2 = torch.tensor(cfg.uav2_goal2).float().cuda()
                if reward_index%3==2:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L2)
                    goal_uav1 = torch.tensor(cfg.uav1_goal3).float().cuda()
                    goal_uav2 = torch.tensor(cfg.uav2_goal3).float().cuda()
            else:
                if episode_n%3==0:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L)
                    goal_uav1 = torch.tensor(cfg.uav1_goal1).float().cuda()
                    goal_uav2 = torch.tensor(cfg.uav2_goal1).float().cuda()
                if episode_n%3==1:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L1)
                    goal_uav1 = torch.tensor(cfg.uav1_goal2).float().cuda()
                    goal_uav2 = torch.tensor(cfg.uav2_goal2).float().cuda()
                if episode_n%3==2:
                    self.env.wind_controller_x.publish(0.0)
                    self.env.wind_controller_y.publish(0.0)
                    self.env.set_L(L2)
                    goal_uav1 = torch.tensor(cfg.uav1_goal3).float().cuda()
                    goal_uav2 = torch.tensor(cfg.uav2_goal3).float().cuda()

            self.adapt_buffer = dict(obs=[],act=[])

            if i == 1:
                self.policy.model.save_model(0)
        
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
            past_traj_error = [np.zeros((1, 3))]
            past_traj_error1 = [np.zeros((1, 2))]
            past_traj_error2 = [np.zeros((1, 2))]

            if i>0:
                #normalize goal input in observation
                O_t = torch.from_numpy(o1).cuda().float()
                g_normalize_s = g_s.copy()
                g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
                g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
                g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
                g_current = torch.from_numpy(g_normalize_s.reshape(1,-1)).cuda().float()

                # print("old:",past_traj[-1])
                g_normalize_past = past_traj[-1]
                g_normalize_past[:,0] = g_normalize_past[:,0]/self.env.max_x
                g_normalize_past[:,1] = g_normalize_past[:,1]/self.env.max_y
                g_normalize_past[:,2] = g_normalize_past[:,2]/self.env.max_z
                g_old = torch.from_numpy(g_normalize_past.reshape(1,-1)).cuda().float()

                g_normalize_past_c = past_corrected_goals[-1]
                g_normalize_past_c[:,0] = g_normalize_past_c[:,0]/2.0
                g_normalize_past_c[:,1] = g_normalize_past_c[:,1]/2.0
                g_normalize_past_c[:,2] = g_normalize_past_c[:,2]/2.0
                g_old_correct = torch.from_numpy(g_normalize_past_c.reshape(1,-1)).cuda().float()

                e_normalize_past = past_traj_error[-1]
                e_old = torch.from_numpy(e_normalize_past.reshape(1, -1)).cuda().float()
                
                e_normalize_past1 = past_traj_error1[-1]
                e_old1 = torch.from_numpy(e_normalize_past1.reshape(1, -1)).cuda().float()
                
                e_normalize_past2 = past_traj_error2[-1]
                e_old2 = torch.from_numpy(e_normalize_past2.reshape(1,-1)).cuda().float()

                s_all = torch.cat((O_t[None],goal_uav1+g_current[:,:2], goal_uav2+g_current[:,:2], g_current[:,:3],g_old[:,:3],g_old_correct,e_old,e_old1,e_old2),dim=1)
                assert s_all.shape[1]==12+3+3+3+3+2+2+2+2
            
            self.adapt_buffer['obs'].append(o1)
            self.policy.model.fast_adapted_params = None

            self.policy.reset()
        
            if (episode_n%evaluate_frequency==0 and episode_n!=0) and not repeat_eval:
                if reward_index%3==0:
                    self.env.wind_controller_x.publish(wind_condition_x)
                    self.env.wind_controller_y.publish(wind_condition_y)
                if reward_index%3==1:
                    self.env.wind_controller_x.publish(wind_condition_x1)
                    self.env.wind_controller_y.publish(wind_condition_y1)
                if reward_index%3==2:
                    self.env.wind_controller_x.publish(wind_condition_x2)
                    self.env.wind_controller_y.publish(wind_condition_y2)

                for t in range(task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()

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

                    if reward_index%3==0:
                        obs, reward, (done, done_f), (goal, g_s) = self.env.step_obs(A[t], 1)
                    if reward_index%3==1:
                        obs, reward, (done,done_f), (goal, g_s) = self.env.step_obs(A[t], 2)
                    if reward_index%3==2:
                        obs, reward, (done, done_f), (goal, g_s) = self.env.step_obs(A[t], 3)
                        
                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)

                    new_error_traj1 = np.zeros((1,2))
                    new_error_traj1[:,0] = reward['error_u1_x']
                    new_error_traj1[:,1] = reward['error_u1_y']
                    past_traj_error1.append(new_error_traj1)

                    new_error_traj2 = np.zeros((1,2))
                    new_error_traj2[:,0] = reward['error_u2_x']
                    new_error_traj2[:,1] = reward['error_u2_y']
                    past_traj_error2.append(new_error_traj2)
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    reward_u1 = -reward['error_uav1']
                    reward_u2 = -reward['error_uav2']

                    # reward['reward'] = reward_scaling(reward['reward'])
                    self.adapt_buffer['obs'].append(obs)
                    self.adapt_buffer['act'].append(A[t])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                    g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                    # print("old1:",past_traj[-1])
                    g_normalize_past1 = past_traj[-1]
                    g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
                    g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
                    g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
                    g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                    g_normalize_past_c1 = past_corrected_goals[-1]
                    g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                    g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                    g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                    g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                    e_normalize_past1 = past_traj_error[-1]
                    e_old1 = torch.from_numpy(e_normalize_past1.reshape(1, -1)).cuda().float()
                    
                    e_normalize_past2 = past_traj_error1[-1]
                    e_old2 = torch.from_numpy(e_normalize_past2.reshape(1, -1)).cuda().float()
                    
                    e_normalize_past3 = past_traj_error2[-1]
                    e_old3 = torch.from_numpy(e_normalize_past3.reshape(1,-1)).cuda().float()

                    s_all1 = torch.cat((O_t1[None],goal_uav1+g_current1[:,:2], goal_uav2+g_current1[:,:2],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1,e_old2,e_old3),dim=1)
                    assert s_all1.shape[1]==12+2+2+3+3+3+3+2+2

                    s_all = s_all1

                    # prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))
                    O.append(obs)
                    reward_sum += reward['reward']
                    reward_sum += reward_u1
                    reward_sum += reward_u2
                    rewards.append(reward['reward']+reward_u1+reward_u2)
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
        
                if episode_n%3==1:
                    self.env.wind_controller_x.publish(wind_condition_x1)
                    self.env.wind_controller_y.publish(wind_condition_y1)

                if episode_n%3==2:
                    self.env.wind_controller_x.publish(wind_condition_x2)
                    self.env.wind_controller_y.publish(wind_condition_y2)

                for t in range(task_hor):
                    # if t>100:
                    #     self.env.wind_controller_x.publish(0.2)
                    #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
                    #     self.env.set_L(0.8)
                    # break
                    start = time.time()
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

                    if episode_n%3==0:
                        obs, reward, (done, done_f), (goal, g_s) = self.env.step_obs(A[t], 1)
                    if episode_n%3==1:
                        obs, reward, (done, done_f), (goal, g_s) = self.env.step_obs(A[t], 2)
                    if episode_n%3==2:
                        obs, reward, (done,done_f), (goal, g_s) = self.env.step_obs(A[t], 3)

                    new_error_traj = np.zeros((1,3))
                    new_error_traj[:,0] = reward['error_x']
                    new_error_traj[:,1] = reward['error_y']
                    new_error_traj[:,2] = reward['error_z']
                    past_traj_error.append(new_error_traj)

                    new_error_traj1 = np.zeros((1,2))
                    new_error_traj1[:,0] = reward['error_u1_x']
                    new_error_traj1[:,1] = reward['error_u1_y']
                    past_traj_error1.append(new_error_traj1)

                    new_error_traj2 = np.zeros((1,2))
                    new_error_traj2[:,0] = reward['error_u2_x']
                    new_error_traj2[:,1] = reward['error_u2_y']
                    past_traj_error2.append(new_error_traj2)

                    #reward process
                    reward['reward'] = -reward['reward']
                    reward_u1 = -reward['error_uav1']
                    reward_u2 = -reward['error_uav2']
                    # reward['reward'] = reward_scaling(reward['reward'])

                    if i>0:
                        O_t1 = torch.from_numpy(obs).cuda().float()
                        g_normalize_s1 = g_s.copy()
                        g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                        g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                        g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                        g_current1 = torch.from_numpy(g_normalize_s1.reshape(1,-1)).cuda().float()

                        # print("old1:",past_traj[-1])
                        g_normalize_past1 = past_traj[-1]
                        g_normalize_past1[:,0] = g_normalize_past1[:,0]/self.env.max_x
                        g_normalize_past1[:,1] = g_normalize_past1[:,1]/self.env.max_y
                        g_normalize_past1[:,2] = g_normalize_past1[:,2]/self.env.max_z
                        g_old1 = torch.from_numpy(g_normalize_past1.reshape(1,-1)).cuda().float()

                        g_normalize_past_c1 = past_corrected_goals[-1]
                        g_normalize_past_c1[:,0] = g_normalize_past_c1[:,0]/2.0
                        g_normalize_past_c1[:,1] = g_normalize_past_c1[:,1]/2.0
                        g_normalize_past_c1[:,2] = g_normalize_past_c1[:,2]/2.0
                        g_old_correct1 = torch.from_numpy(g_normalize_past_c1.reshape(1,-1)).cuda().float()

                        e_normalize_past1 = past_traj_error[-1]
                        e_old1 = torch.from_numpy(e_normalize_past1.reshape(1, -1)).cuda().float()
                        
                        e_normalize_past2 = past_traj_error1[-1]
                        e_old2 = torch.from_numpy(e_normalize_past2.reshape(1, -1)).cuda().float()
                        
                        e_normalize_past3 = past_traj_error2[-1]
                        e_old3 = torch.from_numpy(e_normalize_past3.reshape(1, -1)).cuda().float()

                        s_all1 = torch.cat((O_t1[None],goal_uav1+g_current1[:,:2], goal_uav2+g_current1[:,:2],g_current1[:,:3],g_old1[:,:3],g_old_correct1,e_old1,e_old2,e_old3),dim=1)
                        assert s_all1.shape[1]==12+2+2+3+3+3+3+2+2

                        if done or t == task_hor-1:
                            dw = True
                        else:
                            dw = False

                    # print(goal)
                    # print("reward:",reward['reward'])
                    if i > 0:
                        ppo_buffer_time = time.time()
                        replay_buffer.store(s_all, a, a_logprob, reward['reward'] + reward_u1 + reward_u2, s_all1, done, dw)
                        print("save buffer: ", time.time()-ppo_buffer_time)
                        # print("obs:", s_all)
                        # print("next_obs:",s_all1)
                        # print("action:",a)
                        # print("reward:",reward['reward'])

                        s_all = s_all1
                        total_steps+=1

                        if replay_buffer.count == cfg.buffer_params.batch_size:
                            ppo1_time = time.time()
                            ppo_agent.update(replay_buffer, total_steps)
                            mdict_ppo = {'arr': time.time() - ppo1_time}
                            savemat("/home/wawa/catkin_meta/src/MBRL_transport/src/ppo_update_time.mat",mdict_ppo)
                            replay_buffer.count = 0

                    prediction_error.append(self.policy._validate_prediction(O[t],A[t],obs))

                    self.adapt_buffer['obs'].append(obs)
                    self.adapt_buffer['act'].append(A[t])

                    O.append(obs)
                    reward_sum += reward['reward']
                    reward_sum += reward_u1
                    reward_sum += reward_u2
                    rewards.append(reward['reward']+reward_u1+reward_u2)
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

