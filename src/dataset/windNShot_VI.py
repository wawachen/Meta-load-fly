import  numpy as np
import pickle
import torch

def fill_n_prev(mat, n_prev, initial_zero=True):
    if initial_zero:
        prev_filled_mat = np.zeros((mat.shape[0], n_prev * mat.shape[1]))  # prev starts at initial value
    else:
        prev_filled_mat = np.tile(mat, (1, n_prev))  # prev starts at initial value
    for i in range(n_prev):
        # fill in column block with previous row (from rows i+1 down)
        if i + 1 < mat.shape[0]:
            start_col = i * mat.shape[1]
            end_col = (i + 1) * mat.shape[1]
            # block of size [B - (i+1), N]
            prev_filled_mat[i + 1:, start_col:end_col] = mat[:-(i + 1)]

    return prev_filled_mat

def split_dim_np(np_in, axis, new_shape):
    sh = list(np_in.shape)
    assert axis < len(sh)
    assert sh[axis] == np.prod(new_shape)
    new_shape = sh[:axis] + list(new_shape) + sh[axis + 1:]
    return np_in.reshape(new_shape)


class WindNShot_VI:

    def __init__(self, flag, batch_size, horizon, obs_history_length, acs_history_length, env_spec):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        
        # self._all_names = ['obs', 'prev_obs', 'prev_act', 'latent', 'act', 'next_obs', 'next_obs_sigma','goal_obs','done']
        self._batch_size = batch_size  # number of samples to get per training epoch
        self._planning_horizon = horizon  # obs sequence length per batch item
        self._obs_history_length = obs_history_length  # obs history in model input
        self._acs_history_length = acs_history_length  # action history in model input
        # self._save_every_n_steps = params.save_every_n_steps
        # assert self._save_every_n_steps == 0 or self._output_file

        self._all_names = env_spec.observation_names + \
                          env_spec.action_names + \
                          env_spec.output_observation_names + \
                          ['done']

        self._data_len = 0
        self._num_episodes = 0
        self._split_indices = np.array([])

        self._env_spec = env_spec
        o_shape = self._env_spec.names_to_shapes['obs']
        self._sigma_obs = np.ones(o_shape)
        self._mu_obs = np.zeros(o_shape)

        # Here we prepare the dataset for offline meta training
        with open('/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/VI/train_hold_data.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        if flag == "train":
            obs_all = loaded_dict["train_obs"]
            acs_all = loaded_dict["train_acs"]
            latent_all = loaded_dict["train_latent"]
        else:
            obs_all = loaded_dict["hold_obs"]
            acs_all = loaded_dict["hold_acs"]
            latent_all = loaded_dict["hold_latent"]

        mu_obs = np.mean(np.concatenate(obs_all,axis=0), axis=0)
        sigma_obs = np.std(np.concatenate(obs_all,axis=0), axis=0)

        pre_remove = 0
        n_acs = self._acs_history_length
        n_obs = self._obs_history_length

        obs_start_list = []
        acs_start_list = []
        next_obs_list = []
        latent_start_list = []
        prev_obs_start_list = []
        prev_acs_start_list = []
        done_list = []

        for i in range(len(obs_all)):
            obs = obs_all[i]
            acs = acs_all[i]
            latent = latent_all[i]

            ### Observation sequences
            ### Action histories
            prev_acs = split_dim_np(fill_n_prev(acs, n_acs), axis=1, new_shape=[n_acs] + list(acs.shape[1:]))
            acs_start = acs[pre_remove:-self._planning_horizon]
            prev_acs_start = prev_acs[pre_remove:-self._planning_horizon]

            ### Observation histories

            # we don't use initial_zero=False here bc we remove the first pre_remove anyways
            prev_obs = split_dim_np(fill_n_prev(obs, n_obs), axis=1, new_shape=[n_obs] + list(obs.shape[1:]))
            obs_start = obs[pre_remove:-self._planning_horizon]
            next_obs = obs[pre_remove + 1:-self._planning_horizon + 1]
            
            latent_start = latent[pre_remove:-self._planning_horizon].astype(int)
            
            prev_obs_start = prev_obs[pre_remove:-self._planning_horizon]

            obs_start_list.append(obs_start)  # (N x dO)
            acs_start_list.append(acs_start)  # (N x dU)
            next_obs_list.append(next_obs)  # (N x dO)
            latent_start_list.append(latent_start)  # (N x 1)
            prev_obs_start_list.append(prev_obs_start)  # (N x nobs x dO)
            prev_acs_start_list.append(prev_acs_start)  # (N x nacs x dO)
            done_list.append(np.array([False for _ in range(obs_start.shape[0] - 1)] + [True], dtype=np.bool))

        # some input statistics
        delta_obs = np.concatenate(next_obs_list, axis=0) - np.concatenate(obs_start_list, axis=0)
        mu_delta_obs = np.mean(delta_obs, axis=0)
        sigma_delta_obs = np.std(delta_obs, axis=0)
        next_obs_sigma_list = [np.tile(sigma_delta_obs[None], (next_obs.shape[0], 1)) for next_obs in next_obs_list]

        return_dict = {
        'mu_obs': mu_obs,
        'sigma_obs': sigma_obs,
        'mu_delta_obs': mu_delta_obs,
        'sigma_delta_obs': sigma_delta_obs,
        'done': done_list,
        'obs_full': obs_all,
        'act_full': acs_all,
        # 'obs_seq': obs_seq_list,
        'latent': latent_start_list,
        'obs': obs_start_list,
        'act': acs_start_list,
        'prev_obs': prev_obs_start_list,
        'prev_act': prev_acs_start_list,
        'next_obs': next_obs_list,
        'next_obs_sigma': next_obs_sigma_list,
        }

        self._mu_obs = return_dict['mu_obs']
        self._sigma_obs = return_dict['sigma_obs']
        self._mu_delta_obs = return_dict['mu_delta_obs']
        self._sigma_delta_obs = return_dict['sigma_delta_obs']

        self._num_episodes += len(return_dict['done'])
        
        # logger.debug('Dataset length: {}'.format(self._data_len))

        local_dict = {}    
        for key in self._all_names:
            # turn list into np array with the correct type
            local_dict[key] = np.concatenate(return_dict[key], axis=0).astype(self._env_spec.names_to_dtypes[key])
        
        self._data_len += local_dict['obs'].shape[0]
        
        self._datadict = local_dict

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")


    def get_sigma_obs(self):
        return self._sigma_obs

    def get_batch(self, indices=None, min_idx=0):
        # TODO fix this
        if indices is None:
            assert 0 <= min_idx < self._data_len
            batch = min(self._data_len - min_idx, self._batch_size)
            indices = np.random.choice(self._data_len - min_idx, batch, replace=False)
            indices += min_idx  # base index to consider in dataset

        # get current batch
        # sampled_datadict = self._datadict[indices]

        inputs = {}
        outputs = {}
        
        for key in self._env_spec.observation_names:
            inputs[key] = torch.from_numpy(self._datadict[key][indices]).to(self.device)
        for key in self._env_spec.action_names:
            inputs[key] = torch.from_numpy(self._datadict[key][indices]).to(self.device)
        for key in self._env_spec.output_observation_names:
            outputs[key] = torch.from_numpy(self._datadict[key][indices]).to(self.device)

        outputs["done"] = torch.from_numpy(self._datadict["done"][indices]).to(self.device)

        return inputs, outputs  # shape is (batch, horizon, name_dim...)


