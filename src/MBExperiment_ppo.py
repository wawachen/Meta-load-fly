from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import savemat
from common.Agent import Agent
import torch 
import numpy as np
import scipy.io as scio
from models.ppo import PPO_model
from common.replay_buffer import ReplayBuffer
import time

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')


class MBExperiment:
    def __init__(self, env, logger, path):
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

        self.env = env
        self.agent = Agent(env)
        self.logger = logger
        self.path = path

    def run_experiment_ppo(self, cfg, is_eval):
        torch.manual_seed(cfg.seed)
        # torch.cuda.manual_seed_all(222)
        np.random.seed(cfg.seed)
        train_iters = cfg.train_iters

        logger_ppo = self.logger # used for tensorboard
      
        #runing rollouts for collection samples for meta training
        #task 1: wind 0.0 L 0.6  
    #     # task 2: wind 0.3 L 1.0 
    #     # task 3: wind 0.5 L 0.8 
    #     # task 4: wind 0.8 L 1.2
    #     # test task 1: wind 1.0 L 0.8
    #     # test task 2: wind 0.6 L 1.4
        wind_condition_x = cfg.wind_condition_x
        wind_condition_y = cfg.wind_condition_y
        L = cfg.L
        task_hor = cfg.task_hor

        replay_buffer = ReplayBuffer(cfg)
        cfg.max_train_steps = train_iters* task_hor
        ppo_agent = PPO_model(cfg,logger_ppo, self.path, is_eval).to(TORCH_DEVICE)

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

            #normalize goal input in observation
            O_t = torch.from_numpy(o1).cuda().float()
            g_normalize_s = g_s.copy()
            # g_normalize_s = np.array([[3.0,0.0,1.2]])
            g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
            g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
            g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
            g_current = torch.from_numpy(g_normalize_s[0,:].reshape(1,-1)).cuda().float()-O_t[None][:,(6,7,8)]

            s_all = torch.cat((O_t[None],g_current),dim=1)
            assert s_all.shape[1]==12+3
        
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
                    
                    a = ppo_agent.evaluate(s_all.float()) 
                    action=a
                    # print(store_top_s)
                    A.append(action)
                    times.append(time.time() - start)

                    obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    if done_f==1:
                        reward['reward']-=200
                    # reward['reward'] = reward_scaling(reward['reward'])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    # g_normalize_s1 = np.array([[3.0,0.0,1.2]])
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                    g_current1 = torch.from_numpy(g_normalize_s1[0,:].reshape(1,-1)).cuda().float()-O_t1[None][:,(6,7,8)]

                    s_all1 = torch.cat((O_t1[None],g_current1),dim=1)
                    assert s_all1.shape[1]==12+3

                    s_all = s_all1

                    O.append(obs)
                    reward_sum += reward['reward']
                    rewards.append(reward['reward'])
        
                    if done:
                        break
                
                if reward_index%3==0 and reward_index!=0:
                    repeat_eval = True

                # print(reward_sum)
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
                    #add ppo here    
                    # print("i:",self.env.trajectory.get_i())
                    # print("s_all:", s_all)

                    a, a_logprob = ppo_agent.choose_action(s_all.float()) 
                    # print("a:",a)

                    action=a
                    # print(store_top_s)
                    A.append(action)
                    times.append(time.time() - start)

                    obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])
                    
                    #reward process
                    reward['reward'] = -reward['reward']
                    if done_f==1:
                        reward['reward']-=200
                    # reward['reward'] = reward_scaling(reward['reward'])

                    O_t1 = torch.from_numpy(obs).cuda().float()
                    g_normalize_s1 = g_s.copy()
                    # g_normalize_s1 = np.array([[3.0,0.0,1.2]])
                    g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
                    g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
                    g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
                    g_current1 = torch.from_numpy(g_normalize_s1[0,:].reshape(1,-1)).cuda().float()-O_t1[None][:,(6,7,8)]

                    s_all1 = torch.cat((O_t1[None],g_current1),dim=1)
                    assert s_all1.shape[1]==12+3

                    if done or t == task_hor-1:
                        dw = True
                    else:
                        dw = False
                   
                    replay_buffer.store(s_all, a, a_logprob, reward['reward'], s_all1, done, dw)

                    s_all = s_all1
                    total_steps+=1

                    if replay_buffer.count == cfg.batch_size:
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

    def run_ppo_evaluation(self, cfg, is_eval):
        # 222,50,8,1,20,45,60,104,165,200
        torch.manual_seed(cfg.seed)
        # torch.cuda.manual_seed_all(222)
        np.random.seed(cfg.seed)
        train_iters = 1

        #ppo logger
        logger_ppo = self.logger # used for tensorboard

        #runing rollouts for collection samples for meta training
        wind_condition_x = cfg.wind_condition_x
        wind_condition_y = cfg.wind_condition_y
        L = cfg.L

        #task 1: wind 0.0 L 0.6  
        # task 2: wind 0.3 L 1.0 
        # task 3: wind 0.5 L 0.8 
        # task 4: wind 0.8 L 1.2
        # test task 1: wind 1.0 L 0.8
        # test task 2: wind 0.6 L 1.4
        task_hor = cfg.task_hor
        cfg.max_train_steps = train_iters* task_hor
        ppo_agent = PPO_model(cfg, logger_ppo, self.path, is_eval).to(TORCH_DEVICE)

        # reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        self.env.wind_controller_x.publish(0.0)
        self.env.wind_controller_y.publish(0.0)
        self.env.set_L(L)
    
        times, rewards = [], []
        errorx = []
        errory = []
        errorz = []
        
        o1, goal, g_s = self.env.reset()
        O, A, reward_sum, done = [o1], [], 0, False
        # reward_scaling.reset()

        obs1_l = []
        obs2_l = []
        obs1 = self.env.get_uav_obs()[0]
        obs2 = self.env.get_uav_obs()[1]
        obs1_l.append(obs1)
        obs2_l.append(obs2)

        #normalize goal input in observation
        O_t = torch.from_numpy(o1)
        g_normalize_s = g_s.copy()
        # g_normalize_s = np.array([[3.0,0.0,1.2]])
        g_normalize_s[:,0] = g_normalize_s[:,0]/self.env.max_x
        g_normalize_s[:,1] = g_normalize_s[:,1]/self.env.max_y
        g_normalize_s[:,2] = g_normalize_s[:,2]/self.env.max_z
        g_current = torch.from_numpy(g_normalize_s[0,:].reshape(1,-1))-O_t[None][:,(6,7,8)]

        s_all = torch.cat((O_t[None],g_current),dim=1)
        assert s_all.shape[1]==12+3
    
        self.env.wind_controller_x.publish(wind_condition_x)
        self.env.wind_controller_y.publish(wind_condition_y)

        for t in range(task_hor):
            # if t>100:
            #     self.env.wind_controller_x.publish(0.2)
            #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
            #     self.env.set_L(0.8)
            # break
            start = time.time()
            
            a = ppo_agent.evaluate(s_all.cuda().float()) 
            action=a
            # print(store_top_s)
            A.append(action)
            times.append(time.time() - start)

            obs, reward, (done,done_f), (goal, g_s) = self.env.step(A[t])

            obs1 = self.env.get_uav_obs()[0]
            obs2 = self.env.get_uav_obs()[1]
            obs1_l.append(obs1)
            obs2_l.append(obs2)
            
            #reward process
            reward['reward'] = -reward['reward']
            if done_f==1:
                reward['reward']-=200
            # reward['reward'] = reward_scaling(reward['reward'])

            O_t1 = torch.from_numpy(obs)
            g_normalize_s1 = g_s.copy()
            # g_normalize_s1 = np.array([[3.0,0.0,1.2]])
            g_normalize_s1[:,0] = g_normalize_s1[:,0]/self.env.max_x
            g_normalize_s1[:,1] = g_normalize_s1[:,1]/self.env.max_y
            g_normalize_s1[:,2] = g_normalize_s1[:,2]/self.env.max_z
            g_current1 = torch.from_numpy(g_normalize_s1[0,:].reshape(1,-1))-O_t1[None][:,(6,7,8)]

            s_all1 = torch.cat((O_t1[None],g_current1),dim=1)
            assert s_all1.shape[1]==12+3

            s_all = s_all1

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
        savemat(self.path+'/storeUAV1.mat', mdict={'arr': obs1_l})
        savemat(self.path+'/storeUAV2.mat', mdict={'arr': obs2_l})
        
        sample = {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "reward_average":-reward_sum/len(A),
            "rewards": np.array(rewards),
            "error_x": np.array(errorx),
            "error_y": np.array(errory),
            "error_z": np.array(errorz),
        }

        if cfg.log_sample_data:
            savemat(self.path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
            savemat(self.path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
            savemat(self.path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
            savemat(self.path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
            savemat(self.path+'/storeObs.mat', mdict={'arr': sample["obs"]})
            data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
            savemat(self.path+'/store_destraj.mat', mdict={'arr': data['arr']})

        self.logger.close()
