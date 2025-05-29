from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from common.optimizers import CEMOptimizer
import torch

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')

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

    def act(self, obs, t):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, params, mode, env, model):
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
       
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.max_x = env.max_x
        self.max_z = env.max_z
        self.max_y = env.max_y
      
        self.per = params.per
        self.prop_mode = params.prop_mode
        self.npart = params.npart #num of particles for cem 
        self.opt_mode = params.opt_mode
        self.plan_hor = params.plan_hor
        self.num_nets = params.num_nets  #emsemble models
        self.epsilon = params.epsilon,
        self.alpha = params.alpha
        self.max_iters = params.max_iters
        self.popsize = params.popsize
        self.num_elites = params.num_elites
        self.mode = mode
        self.model = model

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
        self.ac_buf = np.array([]).reshape(0, self.dU)
        #sol: [act_dim*plan_hor,]
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart))

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()

    def act(self, obs, t, goal):
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

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            #[8000, dim1] 
            cost = self.default_mpc_cost_fn(next_obs[:,(6,7,8)], goal[t,:,:]) + self.ac_cost_fn_3d(cur_acs)

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

        return costs.mean(dim=1).detach().cpu().numpy(), store_states

    def _validate_prediction(self,obs,acs,next_obs):
        obs = torch.from_numpy(obs).float().to(TORCH_DEVICE)
        obs = obs[None]
        acs = torch.from_numpy(acs).float().to(TORCH_DEVICE)
        acs = acs[None]
        next_obs = torch.from_numpy(next_obs).float().to(TORCH_DEVICE)

        proc_obs = self.obs_preproc_3d(obs)
        
        inputs = torch.cat((proc_obs, acs), dim=-1)

        
        if self.mode == "MBRL":
            mean, var = self.model.net(inputs)
            predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()
        
        if self.mode == "Meta":
            #meta policy is determinstic nets
            if self.model.fast_adapted_params == None:
                predictions = self.model.pre_forward(inputs)
            else:
                predictions = self.model.post_forward(inputs)

        if self.mode == "FAMLE":
            predictions = self.model.predict_tensor(inputs)
        
        prediction_s = self.obs_postproc(obs, predictions)

        return prediction_s.detach().cpu().numpy(), next_obs.detach().cpu().numpy()

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc_3d(obs)

        assert self.prop_mode == 'TSinf'

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        if self.mode == "MBRL":
            mean, var = self.model.net(inputs)
            predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()
        
        if self.mode == "Meta":
            #meta policy is determinstic nets
            if self.model.fast_adapted_params == None:
                predictions = self.model.pre_forward(inputs)
            else:
                predictions = self.model.post_forward(inputs)

        if self.mode == "FAMLE":
            predictions = self.model.predict_tensor(inputs)

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions)

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [8000, 5] in case of proc_obs
        if self.mode == "MBRL":
            #[400,1,20,5]
            reshaped = mat.view(-1, self.model.net.num_nets, self.npart // self.model.net.num_nets, dim)
        else:
            #[400,1,20,5]
            reshaped = mat.view(-1, 1, self.npart // 1, dim)
        
        transposed = reshaped.transpose(0, 1)
        # After, [1, 400, 20, 5]

        if self.mode == "MBRL":
            reshaped = transposed.contiguous().view(self.model.net.num_nets, -1, dim)
        else:
            reshaped = transposed.contiguous().view(1, -1, dim)
        # After. [1, 8000, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        if self.mode == "MBRL":
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
    

        


        



