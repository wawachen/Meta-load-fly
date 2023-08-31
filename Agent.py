from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import rospy
import torch
from scipy.io import savemat
import sys
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/FAMLE")
import famle
import copy

TORCH_DEVICE = torch.device('cuda')


class Agent:
    """An general class for RL agents.
    """

    def __init__(self, env, meta=False):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        # assert params.get("noisy_actions", False) is False
        self.env = env
        self.meta = meta
        # self.space_3d = rospy.get_param("/firefly/3d_space")
        self.model_3d_in = rospy.get_param("/firefly/model_3d_in")
        self.model_3d_out = rospy.get_param("/firefly/model_3d_out")
        # self.total_steps = 0

        # if isinstance(self.env, DotMap):
        #     raise ValueError("Environment must be provided to the agent at initialization.")

    def sample(self, horizon, policy, wind_test_x,wind_test_y, L, adapt_size=None, log_data=None, data_path=None): 
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
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
        
        o1, goal,_ = self.env.reset()
        O, A, reward_sum, done = [o1], [], 0, False
        top_act_seq = []
        prediction_error = []

        if log_data:
            obs1_l = []
            obs2_l = []
            obs1 = self.env.get_uav_obs()[0]
            obs2 = self.env.get_uav_obs()[1]
            obs1_l.append(obs1)
            obs2_l.append(obs2)

        if self.meta:
            self.adapt_buffer['obs'].append(o1)
            policy.model.fast_adapted_params = None

        policy.reset()
        
        self.env.wind_controller_x.publish(wind_test_x)
        self.env.wind_controller_y.publish(wind_test_y)
        for t in range(horizon):
            # if t>100:
            #     self.env.wind_controller_x.publish(0.2)
            #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
            #     self.env.set_L(0.8)
            # break
            start = time.time()
            if self.meta:
                if len(self.adapt_buffer['act'])>adapt_size:
                    #transform trajectories into adapt dataset
                    new_train_in = np.concatenate([policy.obs_preproc_3d(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1]), np.array(self.adapt_buffer['act'])[-adapt_size:]], axis=-1)

                    new_train_targs = policy.targ_proc(np.array(self.adapt_buffer['obs'])[-adapt_size-1:-1], np.array(self.adapt_buffer['obs'])[-adapt_size:])

                    new_train_in = torch.from_numpy(new_train_in).float().to(TORCH_DEVICE)
                    new_train_targs = torch.from_numpy(new_train_targs).float().to(TORCH_DEVICE)
    
                    policy.model.adapt(new_train_in, new_train_targs)

            # if log_data:
            #     #only log meta model for visualization long time
            #     if self.meta:
            #         policy.model.save_model(t,model_path=data_path)

            action,act_l,store_top_s,store_bad_s = policy.act(O[t], t, goal) #[6,5,2] store top s
            # print(store_top_s)
            self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
            self.env.pub_action_sequence1(store_bad_s)
            A.append(action)
            top_act_seq.append(act_l)
            times.append(time.time() - start)

            obs, reward, (done,done_f), (goal,g_s) = self.env.step(A[t])
            # print(goal)

            prediction_error.append(policy._validate_prediction(O[t],A[t],obs))

            if self.meta:
                self.adapt_buffer['obs'].append(obs)
                self.adapt_buffer['act'].append(A[t])

            if log_data:
                obs1 = self.env.get_uav_obs()[0]
                obs2 = self.env.get_uav_obs()[1]
                obs1_l.append(obs1)
                obs2_l.append(obs2)
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

        if log_data:
            # savemat(data_path+'/storeElites.mat', mdict={'arr': top_act_seq})
            savemat(data_path+'/storeUAV1.mat', mdict={'arr': obs1_l})
            savemat(data_path+'/storeUAV2.mat', mdict={'arr': obs2_l})

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "reward_average":-reward_sum/len(A),
            "rewards": np.array(rewards),
            "error_x": np.array(errorx),
            "error_y": np.array(errory),
            "error_z": np.array(errorz),
            "prediction_error": np.array(prediction_error)
        }

    # def sample_evaluate_ppo(self,args, horizon, policy, wind_test_x,wind_test_y, L, ppo_agent,act_encoder,episode_steps,replay_buffer, adapt_size=None, log_data=None, data_path=None):
    #     a = ppo_agent.evaluate(s)  # We use the deterministic policy during the evaluating
        
    #     while not done:
    #         action = a
    #         s_, r, done, _ = self.env.step(action)

    #         episode_reward += r
    #         s = s_

    def embedding_sample(self, horizon, policy, wind_test_x,wind_test_y, L,log_data=None, data_path=None): 
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
       
        self.env.wind_controller_x.publish(0.0)
        self.env.wind_controller_y.publish(0.0)
        self.env.set_L(L)

        policy.reset()
        
        n_training_tasks = 4
        raw_models = [copy.deepcopy(policy.model) for _ in range(n_training_tasks)]
        models = [copy.deepcopy(policy.model) for _ in range(n_training_tasks)]
        for task_id, m in enumerate(raw_models):
            m.fix_task(task_id)

        for task_id, m in enumerate(models):
            m.fix_task(task_id)

        '''------------------------Test time------------------------------------'''
        task_likelihoods = np.random.rand(n_training_tasks)
        task_index = np.argmax(task_likelihoods)
        policy.model = models[task_index]

        trajectory = []
        times, rewards = [], []
        errorx = []
        errory = []
        errorz = []
        
        o1, goal,_ = self.env.reset()
        O, A, reward_sum, done = [o1], [], 0, False
        top_act_seq = []
        if log_data:
            obs1_l = []
            obs2_l = []
            obs1 = self.env.get_uav_obs()[0]
            obs2 = self.env.get_uav_obs()[1]
            obs1_l.append(obs1)
            obs2_l.append(obs2)
        
        self.env.wind_controller_x.publish(wind_test_x)
        self.env.wind_controller_y.publish(wind_test_y)

        for t in range(horizon):
            # if t>100:
            #     self.env.wind_controller_x.publish(0.2)
            #     self.env.wind_controller_y.publish(0.5) #for tesing the middle fault
            #     self.env.set_L(0.8)
            # if t>100:
            #     self.env.wind_controller.publish(1.2) #for tesing the middle fault
            # break
            start = time.time()

            # if log_data:
            #     #only log meta model for visualization long time
            #     if self.meta:
            #         policy.model.save_model(t,model_path=data_path)

            action,act_l,store_top_s,store_bad_s = policy.act(O[t], t, goal) #[6,5,2] store top s
            # print(store_top_s)
            self.env.pub_action_sequence(store_top_s) #visualize top states in rviz, long traj needs to use stored model
            self.env.pub_action_sequence1(store_bad_s)
            A.append(action)
            top_act_seq.append(act_l)
            times.append(time.time() - start)

            obs, reward, (done,done_f), (goal,g_s) = self.env.step(A[t])
            # print(goal)
            trajectory.append([O[t].copy(), A[t].copy(),
                        obs-O[t]])

            ################################################
            if len(trajectory)>10:
                '''-----------------Compute likelihood before relearning the models-------''' 
                task_likelihoods = famle.compute_likelihood(trajectory, raw_models)

                x, y, high, low = famle.process_data(trajectory)
                task_index = np.argmax(task_likelihoods)

                data_size = len(x)
                models[task_index] = policy.train_model(model=copy.deepcopy(raw_models[task_index]), train_in=x[-data_size::], train_out=y[-data_size::], task_id=task_index)
                policy.model = models[task_index]
                ################################################
            if log_data:
                obs1 = self.env.get_uav_obs()[0]
                obs2 = self.env.get_uav_obs()[1]
                obs1_l.append(obs1)
                obs2_l.append(obs2)
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

        if log_data:
            savemat(data_path+'/storeUAV1.mat', mdict={'arr': obs1_l})
            savemat(data_path+'/storeUAV2.mat', mdict={'arr': obs2_l})

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "reward_average":-reward_sum/len(A),
            "rewards": np.array(rewards),
            "error_x": np.array(errorx),
            "error_y": np.array(errory),
            "error_z": np.array(errorz)
        }
