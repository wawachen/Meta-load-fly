from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import savemat
from tqdm import trange
from common.Agent import Agent
import scipy.io as sio
import torch 
import numpy as np
import scipy.io as scio
from common.train import offline_train_MBRL_pre,offline_train_MBRL_post

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')


class MBExperiment:
    def __init__(self, env, policy, logger):
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
        self.policy = policy
        self.logger = logger


    def run_experiment(self, cfg):
        """Perform model-based experiment.
        """
        task_hor = cfg.task_hor
        log_sample_data = cfg.log_sample_data
        ntrain_iters = cfg.ntrain_iters
        nrollouts_per_iter = cfg.nrollouts_per_iter
        neval = cfg.neval

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)

        wind_condition_x = cfg.wind_x
        wind_condition_y = cfg.wind_y
        L = cfg.L

        if not cfg.is_eval:
            # Perform initial rollouts
            mat_contents = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/all_data/dynamics/firefly_data_3d_wind_x{0}_2agents_L{1}_dt_0.15.mat".format(wind_condition_x,L))
            train_obs = mat_contents['obs'] 
            train_acs = mat_contents['acs']

            #samples [episode, steps,n]
            train_in, train_targs = offline_train_MBRL_pre(self.policy, cfg.epochs, train_obs, train_acs, self.logger)

            # Training loop
            for i in trange(ntrain_iters):
                print("####################################################################")
                print("Starting training iteration %d." % (i + 1))

                samples = []

                #horizon, policy, wind_test_type, adapt_size=None, log_data=None, data_path=None
                #MBRL is baseline no need to log data in agent sampling
                for j in range(max(neval, nrollouts_per_iter)):
                    samples.append(
                            self.agent.sample(
                                task_hor, self.policy, wind_condition_x, wind_condition_y, L
                            )
                        )
                # print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:self.neval]])
                self.logger.add_scalar('Reward', np.mean([sample["reward_average"] for sample in samples[:]]), i)
                samples = samples[:nrollouts_per_iter]

                if i < ntrain_iters - 1:
                    #add new samples into the whole dataset and train the whole dataset
                    train_in, train_targs = offline_train_MBRL_post(
                        self.policy,
                        cfg.epochs,
                        [sample["obs"] for sample in samples],
                        [sample["ac"] for sample in samples],
                        train_in,
                        train_targs,
                        self.logger,
                        i
                    )

            self.logger.close()
        else:
            sample = self.agent.sample(task_hor, self.policy, wind_condition_x, wind_condition_y, L)
            #if path_length of path i is less than k_spt+k_qry, we have to resample it
            
            if log_sample_data:
                print("start logging")
                savemat(cfg.log_sample_path+'/storeReward.mat', mdict={'arr': sample["rewards"]})
                savemat(cfg.log_sample_path+'/store_errorx.mat', mdict={'arr': sample["error_x"]})
                savemat(cfg.log_sample_path+'/store_errory.mat', mdict={'arr': sample["error_y"]})
                savemat(cfg.log_sample_path+'/store_errorz.mat', mdict={'arr': sample["error_z"]})
                savemat(cfg.log_sample_path+'/storeObs.mat', mdict={'arr': sample["obs"]})
                data = scio.loadmat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat') 
                savemat(cfg.log_sample_path+'/store_destraj.mat', mdict={'arr': data['arr']})
            
            self.logger.close()

