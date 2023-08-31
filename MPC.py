from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tkinter import N
import numpy as np
from scipy.io import savemat
import sys
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/FAMLE")
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/PETS")
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/VI")

from optimizers import CEMOptimizer
from PETS import PETS_model
from meta import Meta
from tqdm import trange
from windNShot import WindNShot
# from windNShot_test import WindNShot_test
# from windNShot_obstacle import WindNShot_obs
import famle
from task_generator_em import generate_task_data

import torch
import rospy
import copy
import math
from torch.nn import functional as F
import glob
from dotmap import DotMap
from pytorch_lightning import seed_everything
from occupancy_predictor_2d import Predictor_Model_2d
from latent_env_spec import LatentEnvSpec
# from windNShot_VI import WindNShot_VI
from latent_model import LatentModel
from variation_inference import LatentTrainer

TORCH_DEVICE = torch.device('cuda')

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, log_pred_data=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, params, env, meta_params=None,obs=False):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        if env == None:
            print("please insert env")
        else:
            self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
            self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
            self.max_x = env.max_x
            self.max_z = env.max_z
            self.max_y = env.max_y
        # self.update_fns = params.update_fns
        self.per = params.per

        self.prop_mode = params.prop_mode
        self.npart = params.npart #num of particles for cem
        self.ign_var = params.ign_var  

        self.opt_mode = params.opt_mode
        self.plan_hor = params.plan_hor
        self.num_nets = params.num_nets  #emsemble models
        self.epsilon = params.epsilon,
        self.alpha = params.alpha
        self.epochs = params.epochs
        self.max_iters = params.max_iters
        self.popsize = params.popsize
        self.num_elites = params.num_elites

        # self.model_in = params.model_in
        # self.model_out = params.model_out

        self.model_3d_in = params.model_3d_in
        self.model_3d_out = params.model_3d_out
        self.obs = obs

        self.metaParams = meta_params

        self.save_all_models = False
        self.log_traj_preds = False
        self.log_particles = False
        self.has_obstacle_sig = env.has_obstacle

        # Perform argument checks
        assert self.opt_mode == 'CEM'
        assert self.prop_mode == 'TSinf' #'only TSinf propagation mode is supported'
        assert self.npart % self.num_nets == 0, "Number of particles must be a multiple of the ensemble size."

        # Create action sequence optimizer
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            cost_function=self._compile_cost,
            epsilon = self.epsilon,
            alpha = self.alpha,
            max_iters = self.max_iters ,
            popsize = self.popsize,
            num_elites = self.num_elites
        )

        # Controller state variables
        self.has_been_trained =  False
        self.ac_buf = np.array([]).reshape(0, self.dU)
        #sol: [act_dim*plan_hor,]
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

        # Set up pytorch model

        if meta_params is None:
            self.load_model = params.load_model
            self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc_3d(np.zeros([1, self.dO])).shape[-1])
            self.train_targs = np.array([]).reshape(0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1])
            
            self.model = self.nn_constructor(self.num_nets,self.model_3d_in,self.model_3d_out)
            self.epoch_sum = 0
        else:
            if not meta_params.embedding:
                self.meta_task_num = meta_params.m_task_num
                self.n_way = meta_params.n_way
                self.k_spt = meta_params.k_spt
                self.k_qry = meta_params.k_qry
                self.meta_epoch = meta_params.m_epoch
                self.inter=False
                self.db_train = WindNShot(50, self.meta_task_num, 501, self.n_way, self.k_spt, self.k_qry,integrated=self.inter,sequential=True)
                # self.db_train = WindNShot_test(10, self.meta_task_num, 501, self.n_way, self.k_spt, self.k_qry,integrated=self.inter,sequential=True,task_num=3)
                # self.db_train = WindNShot_obs(50, self.meta_task_num, 501, 3, self.k_spt, self.k_qry,integrated=self.inter,sequential=True)

                self.model = self.meta_nn_constructor(self.model_3d_in,self.model_3d_out)
                if self.has_obstacle_sig:
                    self.occupancy_predictor_nn_constructor()
                    #add obstacle here
                    self.obs_points = env.obs_p
                    self.obs_points1 = env.obs_p1
                    self.obs_pos = env.obs_p_pos
                    self.obs_pos1 = env.obs_p1_pos

                self.meta_epoch_sum = 0
                self.meta_epoch_running = meta_params.m_epoch_running
                self.embded = meta_params.embedding
            else:
                if meta_params.VI:
                    self.VI_nn_constructor()
                else:
                    self.model = self.embedding_nn_constructor(self.model_3d_in, self.model_3d_out)
                    self.embded = meta_params.embedding


    def train(self, obs_trajs, acs_trajs, logger):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.

        Returns: None.
        """

        if not self.has_been_trained:
            #in this part, actions are not the true action, is the normalized reference goal position
            new_train_in, new_train_targs = [], []
            
            #action is real relative distance not normalized one
            new_train_in.append(np.concatenate([self.obs_preproc_3d(obs_trajs[:-1]), acs_trajs], axis=-1))

            new_train_targs.append(self.targ_proc(obs_trajs[:-1], obs_trajs[1:]))
            self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
            self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

            ###########################
            # Train the pytorch model
            self.model.net.fit_input_stats(self.train_in)

            idxs = np.random.randint(self.train_in.shape[0], size=[self.model.net.num_nets, self.train_in.shape[0]])

            epochs = self.epochs

            # TODO: double-check the batch_size for all env is the same
            batch_size = 100

            epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
            num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

            for i in epoch_range:
                train_loss = 0 
                validate_loss = 0

                for batch_num in range(num_batch):
                    batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

                    loss = 0.01 * (self.model.net.max_logvar.sum() - self.model.net.min_logvar.sum())
                    loss += self.model.net.compute_decays()

                    # TODO: move all training data to GPU before hand
                    train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                    train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()

                    mean, logvar = self.model.net(train_in, ret_logvar=True)
                    inv_var = torch.exp(-logvar)

                    train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                    train_losses = train_losses.mean(-1).mean(-1).sum()
                    # Only taking mean over the last 2 dimensions
                    # The first dimension corresponds to each model in the ensemble

                    loss += train_losses
                    train_loss += train_losses.item()

                    self.model.optim.zero_grad()
                    loss.backward()
                    self.model.optim.step()

                logger.add_scalar('Train_iter_offline/Training loss', train_loss/num_batch, i)
                print('Offline: step:', i, '\ttraining acc:', train_loss/num_batch)

                idxs = shuffle_rows(idxs)

                val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
                val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

                mean, _ = self.model.net(val_in)
                mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
                validate_loss += mse_losses.item()
                
                logger.add_scalar('Validation_iter_offline/Validation loss', validate_loss, i)
                print('Offline: step:', i, '\ttest acc:', validate_loss)

                # if i%10:
                #     self.model.save_model(i)
            ###########################

            self.has_been_trained = True
            return 
        else:
            # Construct new training points and add to training set
            #true action, normalized observations
            new_train_in, new_train_targs = [], []
            for obs, acs in zip(obs_trajs, acs_trajs):
                new_train_in.append(np.concatenate([self.obs_preproc_3d(obs[:-1]), acs], axis=-1))
                new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
            self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
            self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.has_been_trained = True

        # Train the pytorch model
        self.model.net.fit_input_stats(self.train_in)

        idxs = np.random.randint(self.train_in.shape[0], size=[self.model.net.num_nets, self.train_in.shape[0]])

        epochs = self.epochs

        # TODO: double-check the batch_size for all env is the same
        batch_size = 100

        epoch_range = epochs
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for i in range(self.epoch_sum, self.epoch_sum+epoch_range):
            train_loss = 0 
            validate_loss = 0

            for batch_num in range(num_batch):
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

                loss = 0.01 * (self.model.net.max_logvar.sum() - self.model.net.min_logvar.sum())
                loss += self.model.net.compute_decays()

                # TODO: move all training data to GPU before hand
                train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()

                mean, logvar = self.model.net(train_in, ret_logvar=True)
                inv_var = torch.exp(-logvar)

                train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                train_losses = train_losses.mean(-1).mean(-1).sum()
                # Only taking mean over the last 2 dimensions
                # The first dimension corresponds to each model in the ensemble

                loss += train_losses
                train_loss += train_losses.item()

                self.model.optim.zero_grad()
                loss.backward()
                self.model.optim.step()
            
            logger.add_scalar('Train_iter_online/Training loss', train_loss/num_batch, i)

            idxs = shuffle_rows(idxs)

            val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
            val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

            mean, _ = self.model.net(val_in)
            mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
            validate_loss += mse_losses.item()
           
            logger.add_scalar('Validation_iter_online/Validation loss', validate_loss, i)

            if i%10==0:
                self.model.save_model(i)

        self.epoch_sum+=epoch_range

    # def offline_train_MBRL(self,obs_trajs, acs_trajs, logger):
    #     #in this part, actions are not the true action, is the normalized reference goal position
    #     new_train_in, new_train_targs = [], []
        
    #     new_train_in.append(np.concatenate([self.obs_preproc_3d(obs_trajs[:-1]), acs_trajs], axis=-1))

    #     new_train_targs.append(self.targ_proc(obs_trajs[:-1], obs_trajs[1:]))
    #     self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
    #     self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

    #     # Train the pytorch model
    #     self.model.net.fit_input_stats(self.train_in)

    #     idxs = np.random.randint(self.train_in.shape[0], size=[self.model.net.num_nets, self.train_in.shape[0]])

    #     epochs = self.epochs

    #     # TODO: double-check the batch_size for all env is the same
    #     batch_size = 100

    #     epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
    #     num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

    #     for i in epoch_range:
    #         train_loss = 0 
    #         validate_loss = 0

    #         for batch_num in range(num_batch):
    #             batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

    #             loss = 0.01 * (self.model.net.max_logvar.sum() - self.model.net.min_logvar.sum())
    #             loss += self.model.net.compute_decays()

    #             # TODO: move all training data to GPU before hand
    #             train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
    #             train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()

    #             mean, logvar = self.model.net(train_in, ret_logvar=True)
    #             inv_var = torch.exp(-logvar)

    #             train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
    #             train_losses = train_losses.mean(-1).mean(-1).sum()
    #             # Only taking mean over the last 2 dimensions
    #             # The first dimension corresponds to each model in the ensemble

    #             loss += train_losses
    #             train_loss += train_losses.item()

    #             self.model.optim.zero_grad()
    #             loss.backward()
    #             self.model.optim.step()

    #         logger.add_scalar('Train_iter/Training loss', train_loss/num_batch, i)
    #         print('step:', i, '\ttraining acc:', train_loss/num_batch)

    #         idxs = shuffle_rows(idxs)

    #         val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
    #         val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

    #         mean, _ = self.model.net(val_in)
    #         mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
    #         validate_loss += mse_losses.item()
            
    #         logger.add_scalar('Validation_iter/Validation loss', validate_loss, i)
    #         print('step:', i, '\ttest acc:', validate_loss)

    #         if i%10:
    #             self.model.save_model(i)
    def offline_test(self,logger):
        accs = []
        for _ in range(10):
            # test
            x_spt, y_spt, x_qry, y_qry = self.db_train.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                            torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = self.model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)

        # [b, update_step+1]
        accs1 = np.array(accs).mean(axis=0).astype(np.float16)
        std_value = np.std(np.array(accs)[:,-1])
        print('Test acc:', accs1[-1])
        print('Test std:', std_value)
            

    def offline_train(self, logger):
        #first offline META training 
        rospy.loginfo("Start offline training...")
        for step in range(self.meta_epoch):

            x_spt, y_spt, x_qry, y_qry = self.db_train.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                            torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

            # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
            accs = self.model(x_spt, y_spt, x_qry, y_qry, step)

            if step % 50 == 0:
                print('step:', step, '\ttraining acc:', accs)
                train_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
                logger.add_scalars('offline train accuracy', train_data_dic, step)

            if step % 500 == 0:
                accs = []
                for _ in range(1000//self.meta_task_num):
                    # test
                    x_spt, y_spt, x_qry, y_qry = self.db_train.next('test')
                    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                                    torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        test_acc = self.model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append(test_acc)

                # [b, update_step+1]
                accs = np.array(accs).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
                test_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
                logger.add_scalars('offline test accuracy', test_data_dic, step)

        #start online rollout and training
        rospy.loginfo("Offline training is finished, start online rollouts...")
        self.meta_epoch_sum += self.meta_epoch

    ############# For emdedding NN

    def train_meta(self, obs_trajs, acs_trajs, logger, num_i):
        self.db_train.add_roll_outs(obs_trajs,acs_trajs)

        self.has_been_trained = True
        
        if self.inter:
            train_epoch = 25
        else:
            train_epoch = max(int(0.8*self.db_train.running_samples_num/(self.meta_task_num*self.n_way*self.k_spt)),1)
        
        for step in range(self.meta_epoch_sum,self.meta_epoch_sum+train_epoch):

            x_spt, y_spt, x_qry, y_qry = self.db_train.meta_next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                            torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

            # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
            accs = self.model(x_spt, y_spt, x_qry, y_qry,step)

            if step % 50 == 0:
                print('step:', step, '\ttraining acc:', accs)
                train_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
                logger.add_scalars('online train accuracy', train_data_dic, step)

            if step % 500 == 0:
                accs = []
                for _ in range(1000//self.meta_task_num):
                    # test
                    x_spt, y_spt, x_qry, y_qry = self.db_train.meta_next('test')
                    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                                    torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        test_acc = self.model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append(test_acc)

                # [b, update_step+1]
                accs = np.array(accs).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
                test_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
                logger.add_scalars('online test accuracy', test_data_dic, step)

        self.meta_epoch_sum += train_epoch


    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()

        # for update_fn in self.update_fns:
        #     update_fn()


    def act(self, obs, t, goal, log_pred_data=False):
        """Returns the action that this controller would take at time t given observation obs.
           for trajectory tracking, we have to iter the goals

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        # if not self.has_been_trained:
        #     return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            # print("action_buffer",self.ac_buf)
            print("action:", action)
            return action,self.soln_elites,self.store_top_s,self.store_bad_s

        self.sy_cur_obs = obs

        # print("current pos", self.sy_cur_obs[0]*self.max_x, self.sy_cur_obs[1], self.sy_cur_obs[2]*self.max_z)
        # print("goal:", goal.shape)
        #[soldim,] [10,soldim]
        soln, self.soln_elites,self.store_top_s,self.store_bad_s = self.optimizer.obtain_solution(self.prev_sol, self.init_var, goal)
        # print("solutions",soln)
        # print(self.store_top_s.shape)
        
        assert(self.store_top_s.shape[0]==self.plan_hor+1 and self.store_top_s.shape[1]==5 and self.store_top_s.shape[2]==3)
        #zeros part may be replaced by the (self.act_high+self.act_low)/2
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        #only store one solution, thus will update each time
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)
        # print("action_buffer",self.ac_buf)

        return self.act(obs, t,goal)

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        # TODO: implement saving model for pytorch
        # self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []


    # template mpc cost function. used for sampling action sequences
    def default_mpc_cost_fn(self, obs_seq, goal_seq):
        assert obs_seq.shape == goal_seq.shape  # (N, obsdim)
        # this std is some scaled version of the observation standard deviation
        # std = model_out_seq.next_obs_sigma[:, :1, 0]  # (N, 1, obsdim) TODO non deterministic
        obs_seq1 = obs_seq.clone().detach()
        
        obs_seq1[:,0] = obs_seq1[:,0]*self.max_x
        obs_seq1[:,1] = obs_seq1[:,1]*self.max_y
        obs_seq1[:,2] = obs_seq1[:,2]*self.max_z

        normalized = torch.abs(obs_seq1 - goal_seq)
        return normalized.sum(1)  # (N,)
    
    def default_mpc_cost_fn_obs(self, obs_seq, goal_seq):
        assert obs_seq.shape == goal_seq.shape  # (N, obsdim)
        # this std is some scaled version of the observation standard deviation
        # std = model_out_seq.next_obs_sigma[:, :1, 0]  # (N, 1, obsdim) TODO non deterministic
        obs_seq1 = obs_seq.clone().detach()
        
        obs_seq1[:,0] = obs_seq1[:,0]*self.max_x
        obs_seq1[:,1] = obs_seq1[:,1]*self.max_y

        normalized = torch.abs(obs_seq1 - goal_seq)
        return normalized.sum(1)  # (N,)
    
    def set_obs(self,num):
        self.obs = num
    
    @torch.no_grad()
    def _compile_cost(self, ac_seqs, goal):

        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        #  After, ac seqs has dimension (400, 25, 2)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 2)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 2)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 2)

        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)
        # Then, (25, 8000, 2)
        goal = goal.float().to(TORCH_DEVICE)
        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        costs = torch.zeros(nopt, self.npart, device=TORCH_DEVICE)
        
        store_states = [cur_obs[:,(6,7,8)].view(-1,self.npart,3).mean(dim=1).clone().detach().cpu().numpy()]

        if self.obs:
            if self.obs == 1:
                dx1 = 0.3024
                dy1 = 0.0786

                dx2 = -0.294
                dy2 = 0.0662

            if self.obs == 2:
                dx1 = 0.2164
                dy1 = -0.0198

                dx2 = -0.4932
                dy2 = -0.018

            if self.obs == 3:
                dx1 = 0.4876
                dy1 = -0.01

                dx2 = -0.5756
                dy2 = -0.0092

            d1_arr = np.zeros((goal[0,:,:].shape[0],2))
            d1_arr[:,0] = np.ones(goal[0,:,:].shape[0])*dx1
            d1_arr[:,1] = np.ones(goal[0,:,:].shape[0])*dy1
            d1_arr = torch.from_numpy(d1_arr).cuda()

            d2_arr = np.zeros((goal[0,:,:].shape[0],2))
            d2_arr[:,0] = np.ones(goal[0,:,:].shape[0])*dx2
            d2_arr[:,1] = np.ones(goal[0,:,:].shape[0])*dy2
            d2_arr = torch.from_numpy(d2_arr).cuda()

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            #[8000, dim1] 
            # cost = self.default_mpc_cost_fn(next_obs[:,:3], goal[0,t,:]) + self.ac_cost_fn(cur_acs)

            #[nopt * npart, 2]->[nopt * npart]
            if self.obs:
                cost = self.default_mpc_cost_fn(next_obs[:,(6,7,8)], goal[t,:,:]) + self.default_mpc_cost_fn_obs(next_obs[:,(0,1)], goal[t,:,:2]+d1_arr) + self.default_mpc_cost_fn_obs(next_obs[:,(3,4)], goal[t,:,:2]+d2_arr) + self.ac_cost_fn_3d(cur_acs) #+ self.obstacle_cost_fn_3d(next_obs[:,(6,7)])
            else:
                cost = self.default_mpc_cost_fn(next_obs[:,(6,7,8)], goal[t,:,:]) + self.ac_cost_fn_3d(cur_acs)
            # if self.has_obstacle_sig:
            #     if t == (self.plan_hor-1):
            #         cost = cost+self.occpuancy_cost_fn(next_obs[:,(1,2,4,5,7,8)], self.obs_points,self.obs_points1,self.obs_pos,self.obs_pos1)
            ####

            s_t = next_obs[:,(6,7,8)].view(-1,self.npart,3).mean(dim=1).clone().detach().cpu().numpy() #[400,20,3]
            assert(s_t.shape[0]==nopt and s_t.shape[1]==3)

            store_states.append(s_t) #[400,2]
            ####
            #[nopt,npart]
            cost = cost.view(-1, self.npart)

            costs += cost
            cur_obs = next_obs

        # Replace nan with high cost
        costs[costs != costs] = 1e6

        #return [nopt,]

        return costs.mean(dim=1).detach().cpu().numpy(), store_states

    def _validate_prediction(self,obs,acs,next_obs):
        obs = torch.from_numpy(obs).float().to(TORCH_DEVICE)
        obs = obs[None]
        acs = torch.from_numpy(acs).float().to(TORCH_DEVICE)
        acs = acs[None]
        next_obs = torch.from_numpy(next_obs).float().to(TORCH_DEVICE)

        proc_obs = self.obs_preproc_3d(obs)
        
        inputs = torch.cat((proc_obs, acs), dim=-1)

        if self.metaParams is None:
            mean, var = self.model.net(inputs)
            predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()
        else:
            if not self.embded:
                #meta policy is determinstic nets
                if self.model.fast_adapted_params == None:
                    predictions = self.model.pre_forward(inputs)
                else:
                    predictions = self.model.post_forward(inputs)
            else:
                predictions = self.model.predict_tensor(inputs)
        
        prediction_s = self.obs_postproc(obs, predictions)
        loss = F.mse_loss(prediction_s, next_obs)

        return loss.item() 

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc_3d(obs)

        assert self.prop_mode == 'TSinf'

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        if self.metaParams is None:
            mean, var = self.model.net(inputs)
            predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()
        else:
            if not self.embded:
                #meta policy is determinstic nets
                if self.model.fast_adapted_params == None:
                    predictions = self.model.pre_forward(inputs)
                else:
                    predictions = self.model.post_forward(inputs)
            else:
                predictions = self.model.predict_tensor(inputs)

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions)

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [8000, 5] in case of proc_obs
        if self.metaParams is None:
            #[400,1,20,5]
            reshaped = mat.view(-1, self.model.net.num_nets, self.npart // self.model.net.num_nets, dim)
        else:
            #[400,1,20,5]
            reshaped = mat.view(-1, 1, self.npart // 1, dim)
        
        transposed = reshaped.transpose(0, 1)
        # After, [1, 400, 20, 5]

        if self.metaParams is None:
            reshaped = transposed.contiguous().view(self.model.net.num_nets, -1, dim)
        else:
            reshaped = transposed.contiguous().view(1, -1, dim)
        # After. [1, 8000, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        if self.metaParams is None:
            reshaped = ts_fmt_arr.view(self.model.net.num_nets, -1, self.npart // self.model.net.num_nets, dim)
        else:
            reshaped = ts_fmt_arr.view(1, -1, self.npart // 1, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped

#################
    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
           return np.concatenate([np.sin(obs[:, 3]).reshape(-1,1), np.cos(obs[:, 3]).reshape(-1,1), obs[:, :3]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, 3].sin().reshape(-1,1),
                obs[:, 3].cos().reshape(-1,1),
                obs[:, :3],
            ], dim=1)

    @staticmethod
    def obs_preproc_3d(obs):
        if isinstance(obs, np.ndarray): 
           return np.concatenate([obs[:, :9],np.sin(obs[:, 9]).reshape(-1,1), np.cos(obs[:, 9]).reshape(-1,1),np.sin(obs[:, 10]).reshape(-1,1), np.cos(obs[:, 10]).reshape(-1,1),np.sin(obs[:, 11]).reshape(-1,1), np.cos(obs[:, 11]).reshape(-1,1)], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, :9],
                obs[:, 9].sin().reshape(-1,1),
                obs[:, 9].cos().reshape(-1,1),
                obs[:, 10].sin().reshape(-1,1),
                obs[:, 10].cos().reshape(-1,1),
                obs[:, 11].sin().reshape(-1,1),
                obs[:, 11].cos().reshape(-1,1),
            ], dim=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def ac_cost_fn(acs):
        #this cost is to constraint reference goal is not too far from the load
        return 0.25 * (acs**2).sum(dim=1)

    @staticmethod
    def ac_cost_fn_3d(acs):
        #this cost is to constraint reference goal is not too far from the load
        return 0.25 * (acs**2).sum(dim=1)

    def obstacle_cost_fn_3d(self, obs):
        #this cost is to constraint reference goal is not too far from the load
        obs_pos =torch.tensor([[3.0,0]]).to(TORCH_DEVICE).float()
        current_obs = obs.clone().detach()
        current_obs[:,0] = current_obs[:,0]*self.max_x
        current_obs[:,1] = current_obs[:,1]*self.max_y

        # over_value = list(map(self.overlap_area, current_obs))

        # over_value = torch.tensor(over_value)

        # return over_value*15
        # print(obs.shape)
        dis = torch.sqrt(((current_obs-obs_pos)**2).sum(dim=1))
        dis_r = dis<0.75
        # print(dis_r.shape)
        return dis_r*15

    def occupancy_predictor(self, state):
        #for online trajectory planner
        current_p = state[6:8].copy()
        current_p[0] = current_p[0]*4.0
        current_p[1] = current_p[1]*2.0
        d = np.sqrt(((current_p-self.obs_pos)**2).sum(axis=1))[0]
        d1 = np.sqrt(((current_p-self.obs_pos1)**2).sum(axis=1))[0]
        # print("dis: ",d)
        sample_obs_points = 300
        cur_obs = torch.from_numpy(state).float().to(TORCH_DEVICE)
        state_e = cur_obs[None]
        state_expand = state_e.expand(sample_obs_points,-1)
        
        if d<1.0:
            #########################
            obs_num = self.obs_points.shape[0]
            obs_num_shuffle = np.random.permutation(obs_num)
            obs_index = obs_num_shuffle[:sample_obs_points]
            obs_points_e = torch.from_numpy(self.obs_points[obs_index,:]).float().to(TORCH_DEVICE)
            #########################
           
            input_c = torch.cat((obs_points_e,state_expand),1)
            sdf_output = self.occupancy_model.model(input_c)
            sdf_output_np = sdf_output.detach().cpu().numpy()
            # sdf_output_np_repulsive = sdf_output_np.copy()

            sdf_np_index = np.where(sdf_output_np>0.050)[0]
            sdf_output_np[sdf_np_index] = 0
            sdf_np_p_index = np.where((sdf_output_np>0)&(sdf_output_np<=0.050))[0]
            sdf_output_np[sdf_np_p_index] = -0.1
            sdf_output_cost = np.sum(sdf_output_np)

            cost_sum =  sdf_output_cost.copy()
            threshold = 0
            if cost_sum<threshold:
                cost1 = 1 
            else:
                cost1 = 0
        else:
            cost1 = 0
       
        if d1<1.0:
            obs_num1 = self.obs_points1.shape[0]
            obs_num_shuffle1 = np.random.permutation(obs_num1)
            obs_index1 = obs_num_shuffle1[:sample_obs_points]
            obs_points_e1 = torch.from_numpy(self.obs_points1[obs_index1,:]).float().to(TORCH_DEVICE)
            #########################
           
            input_c1 = torch.cat((obs_points_e1,state_expand),1)
            sdf_output1 = self.occupancy_model.model(input_c1)
            sdf_output_np1 = sdf_output1.detach().cpu().numpy()
            # sdf_output_np_repulsive = sdf_output_np.copy()

            sdf_np_index1 = np.where(sdf_output_np1>0.050)[0]
            sdf_output_np1[sdf_np_index1] = 0
            sdf_np_p_index1 = np.where((sdf_output_np1>0)&(sdf_output_np1<=0.050))[0]
            sdf_output_np1[sdf_np_p_index1] = -0.1
            sdf_output_cost1 = np.sum(sdf_output_np1)

            cost_sum1 =  sdf_output_cost1.copy()
            threshold1 = 0
            if cost_sum1<threshold1:
                cost2 = 1 
            else:
                cost2 = 0
        else:
            cost2 = 0

        return cost1 or cost2
        

    def occpuancy_cost_fn(self, state, obs_points, obs_points1, obs_pos,obs_pos1):
        # print(self.sy_cur_obs[6:8])
        # d1 = np.sqrt(((self.sy_cur_obs[6:8]-obs_pos1)**2).sum(axis=1))[0]
        # print("dis: ",d)
        # print("wawa")
        obs_l = []
        obs_num = obs_points.shape[0]
        state_e = state[None,:,:]
        state_expand = state_e.expand(obs_num,-1,-1).contiguous().view(-1,state.shape[1])

        #########################
        obs_num_shuffle = np.random.permutation(obs_num)
        obs_index = obs_num_shuffle
        obs_points_e = torch.from_numpy(obs_points[obs_index,:]).float().to(TORCH_DEVICE)
        #########################
        # obs_points_e = torch.from_numpy(obs_points).float().to(TORCH_DEVICE)
        obs_points_ee = obs_points_e[:,None,:]
        obs_points_expand = obs_points_ee.expand(-1,state.shape[0],-1)
        obs_p_final = obs_points_expand.contiguous().view(-1,2)
        assert(state_expand.shape[0]==obs_p_final.shape[0])

        input_c = torch.cat((obs_p_final,state_expand),1)
        sdf_output = self.occupancy_model.model(input_c)
        
        sdf_output1_np = sdf_output.detach().cpu().numpy()
        # print(sdf_output1_np)

        sdf_np_index1 = np.where(sdf_output1_np<=0.01)[0]
        print(len(sdf_np_index1))
        sdf_np_index2 = np.where(sdf_output1_np>0.01)[0]
        sdf_output1_np[sdf_np_index1] = 1
        sdf_output1_np[sdf_np_index2] = 0

        sdf_output1_np_t = sdf_output1_np.reshape(-1,state.shape[0])
        sdf_ouput_cost = np.sum(sdf_output1_np_t,axis=0)

        # print(sdf_ouput_cost)

        # sdf_np_rep_index = np.where(sdf_output_np_repulsive<=0.10)[0]
        # sdf_output_np_repulsive[sdf_np_rep_index] = 0
        # sdf_output_a_repulsive = sdf_output_np_repulsive.reshape(-1,state.shape[0])
        # # sdf_output_cost_rep = np.sum(sdf_output_a_repulsive,axis=0)
        # state_e1 = state[None,:,:]
        # state_expand1 = state_e1.expand(obs_points1.shape[0],-1,-1).contiguous().view(-1,state.shape[1])

        # obs_points_e1 = torch.from_numpy(obs_points1).float().to(TORCH_DEVICE)
        # obs_points_ee1 = obs_points_e1[:,None,:]
        # obs_points_expand1 = obs_points_ee1.expand(-1,state.shape[0],-1)
        # obs_p_final1 = obs_points_expand1.contiguous().view(-1,2)
        # assert(state_expand1.shape[0]==obs_p_final1.shape[0])

        # input_c1 = torch.cat((obs_p_final1,state_expand1),1)
        # sdf_output1 = self.occupancy_model.model(input_c1)
        # sdf_output1_np = sdf_output1.detach().cpu().numpy()
       
        # sdf_np_index1 = np.where(sdf_output1_np<=0.0)[0]
        # sdf_np_index2 = np.where(sdf_output1_np>0.0)[0]
        # sdf_output1_np[sdf_np_index1] = 1
        # sdf_output1_np[sdf_np_index2] = 0

        # sdf_output1_np_t = sdf_output1_np.reshape(-1,state.shape[0])
        # sdf_ouput_cost = np.sum(sdf_output1_np_t,axis=0)

        cost_sum = sdf_ouput_cost.copy()

        cost_sum = torch.from_numpy(cost_sum).float().to(TORCH_DEVICE)
        # print(cost_sum)

        return cost_sum
    

    # def overlap_area(self, b):
    #     a = torch.tensor([3.0,0])
    #     r1 = 0.2
    #     r2 = 0.6
    #     d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    #     if d >= r1+r2:
    #         return 0

    #     if r2 - r1 >= d:
    #         return np.pi*r1*r1

    #     ang1=math.acos((r1*r1+d*d-r2*r2)/(2*r1*d))
    #     ang2=math.acos((r2*r2+d*d-r1*r1)/(2*r2*d))
    #     return ang1*r1*r1 + ang2*r2*r2 - r1*d*np.sin(ang1)
        

    # definitions of different neural network models used in MPC
    def nn_constructor(self, num_nets, model_in, model_out):

        #initialize nerworks here
        ensemble_size = num_nets

        model = PETS_model(ensemble_size, model_in, model_out,load_model=self.load_model)
        # * 2 because we output both the mean and the variance

        # model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


    def meta_nn_constructor(self, model_in, model_out):
        #3 hidden layers of 512 units with ReLU
        #config [out,in]
        config = [
            ('linear', [512, model_in]),
            ('relu', [True]),
            ('linear', [512, 512]),
            ('relu', [True]),
            ('linear', [model_out, 512]),
        ]

        maml = Meta(self.metaParams, config).to(TORCH_DEVICE)

        tmp = filter(lambda x: x.requires_grad, maml.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(maml)
        print('Total trainable tensors:', num)

        return maml 

    def embedding_nn_constructor(self,model_in, model_out):
        fm = famle.Embedding_NN(dim_in=model_in, hidden=[200, 200, 100], dim_out=model_out, embedding_dim=5,
                           num_tasks=4, CUDA=True, SEED=None, output_limit=None, dropout=0.0)

        return fm

    def embedding_meta_train(self, logger):
        # Generate data for meta training
        dataset = generate_task_data()
        tasks_in, tasks_out = dataset.generate_data()

        # Meta train the model + embedding, and save it
        famle.train_meta(self.model, tasks_in, tasks_out, logger, meta_iter=20000, inner_iter=10, inner_step=0.0001, meta_step=0.3, minibatch=32, inner_sample_size=500)
        # self.model.save("/home/wawa/catkin_meta/src/MBRL_transport/FAMLE_model/model.pt")
    
    def load_embedding_model(self):
        self.model = famle.load_model("/home/wawa/catkin_meta/src/MBRL_transport/FAMLE_model/model.pt",device=torch.device('cuda'))

    def train_model(self, model, train_in, train_out, task_id):
        cloned_model = copy.deepcopy(model)
        famle.train(cloned_model,
                    train_in,
                    train_out,
                    task_id,
                    inner_iter=20,
                    inner_lr=1e-4,
                    minibatch=32)
        return cloned_model

    def variation_inference_train(self,logger):
        self.trainer.run(logger)
    
    def VI_nn_constructor(self):
        obs_dim = 3
        act_dim = 3
        OBS_HISTORY_LENGTH = 10
        ACT_HISTORY_LENGTH = 10
        NUM_LATENT_CLASSES = 4
        HORIZON = 5
        batch_size = 100
        NUM_NETS = 1
        PROBABILISTIC = False
        LATENT_DIM = 1
        DEFAULT_LATENT_MU = None
        DEFAULT_LATENT_LOG_SIGMA = None
        LATENT_TRAIN_EVERY_N = 1

        MODEL_IN = obs_dim * (1 + OBS_HISTORY_LENGTH) + act_dim * (1 + ACT_HISTORY_LENGTH) + LATENT_DIM
        MODEL_OUT = obs_dim * 2 if PROBABILISTIC else obs_dim
 
        names_shapes_limits_dtypes=[
                ('obs', (obs_dim,), (0, 1), np.float32),
                ('prev_obs', (OBS_HISTORY_LENGTH, obs_dim), (0, 1), np.float32),
                ('prev_act', (ACT_HISTORY_LENGTH, act_dim), (0, 1), np.float32),
                ('latent', (1,), (0, NUM_LATENT_CLASSES - 1), np.int),

                ('next_obs', (obs_dim,), (0, 1), np.float32),
                ('next_obs_sigma', (obs_dim,), (0, np.inf), np.float32),

                ('goal_obs', (HORIZON+1, obs_dim), (0, 1), np.float32),

                ('act', (act_dim,), (-1, 1), np.float32)]

        env_spec = LatentEnvSpec(names_shapes_limits_dtypes)
        dataset_train = WindNShot_VI("train", batch_size, HORIZON, OBS_HISTORY_LENGTH, ACT_HISTORY_LENGTH, env_spec)
        dataset_holdout = WindNShot_VI("holdout", batch_size, HORIZON, OBS_HISTORY_LENGTH, ACT_HISTORY_LENGTH, env_spec)
    
        model_params = DotMap()
        model_params.num_nets = NUM_NETS
        model_params.is_probabilistic = PROBABILISTIC
        model_params.deterministic_sigma_multiplier = 0.01  # default sigma_obs uncertainty multiplier

        latent_object1 = DotMap()
        latent_object1.num_latent_classes=NUM_LATENT_CLASSES,
        latent_object1.latent_dim=LATENT_DIM,
        latent_object1.known_latent_default_mu=DEFAULT_LATENT_MU,
        latent_object1.known_latent_default_log_sigma=DEFAULT_LATENT_LOG_SIGMA,
        latent_object1.beta_kl=.1

        model_params.latent_object = latent_object1
        model = LatentModel(model_params, env_spec, dataset_train.get_sigma_obs(), MODEL_IN, MODEL_OUT)

        trainer_params = DotMap()
        trainer_params.dynamics_learning_rate=5e-4
        trainer_params.latent_learning_rate=5e-4
        trainer_params.latent_train_every_n_steps=LATENT_TRAIN_EVERY_N
        trainer_params.sample_every_n_steps=0
        trainer_params.train_every_n_steps=1
        trainer_params.holdout_every_n_steps=500
        trainer_params.max_steps=1e5
        trainer_params.max_train_data_steps=0
        trainer_params.max_holdout_data_steps=0
        trainer_params.log_every_n_steps=10
        trainer_params.save_every_n_steps=1e3
        trainer_params.save_checkpoints=True
     
        self.trainer = LatentTrainer(trainer_params,
                                    model,
                                    dataset_train,
                                    dataset_holdout)

    def occupancy_predictor_nn_constructor(self):
        checkpoint_filepath = "/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_1/lightning_logs/version_0/checkpoints"
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
        cfg = DotMap()
        cfg.seed = 1
        cfg.lr = 0.00005 # more_layers: 0.00005, one layer: 0.0001
        cfg.if_cuda = True
        cfg.gamma = 0.5
        cfg.log_dir = 'logs'
        cfg.num_workers = 8
        cfg.model_name = 'Occupancy_predictor'
        cfg.lr_schedule = [400,800]
        cfg.num_gpus = 1
        cfg.epochs = 1000
        cfg.dof = 6
        cfg.coord_system = 'cartesian'
        cfg.tag = '2agents_all'
        seed(cfg)
        seed_everything(cfg.seed)

        log_dir = '/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_{0}'.format(cfg.seed)

        self.occupancy_model = Predictor_Model_2d(lr=cfg.lr,
                                dof=cfg.dof,
                                if_cuda=cfg.if_cuda,
                                if_test=True,
                                gamma=cfg.gamma,
                                log_dir=log_dir,
                                num_workers=cfg.num_workers,
                                coord_system=cfg.coord_system,
                                lr_schedule=cfg.lr_schedule)

        ckpt = torch.load(checkpoint_filepath)
        self.occupancy_model.load_state_dict(ckpt['state_dict'])
        self.occupancy_model = self.occupancy_model.to('cuda')
        self.occupancy_model.eval()
        self.occupancy_model.freeze()

        


        



