import sys
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/FAMLE")
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/PETS")
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/baselines/VI")
sys.path.insert(0, "/home/wawa/catkin_meta/src/MBRL_transport/src/dataset")

from PETS import PETS_model
from models.meta import Meta
import famle
from latent_env_spec import LatentEnvSpec
from windNShot_VI import WindNShot_VI
from latent_model import LatentModel
from variation_inference import LatentTrainer
from dotmap import DotMap
import torch
import numpy as np

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')

# definitions of different neural network models used in MPC
def nn_constructor(model_cfg, path):
    model = PETS_model(model_cfg, path)
    return model

def meta_nn_constructor(model_cfg, path):
    #3 hidden layers of 512 units with ReLU
    #config [out,in]
    config = [
        ('linear', [512, model_cfg.model_in]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('relu', [True]),
        ('linear', [model_cfg.model_out, 512]),
    ]

    maml = Meta(model_cfg, config, path).to(TORCH_DEVICE)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    return maml
    
def embedding_nn_constructor(model_cfg, seed):
    fm = famle.Embedding_NN(dim_in=model_cfg.model_in, hidden=model_cfg.hidden, dim_out=model_cfg.model_out, embedding_dim=model_cfg.embedding_dim,
                        num_tasks=model_cfg.num_tasks, CUDA=True, SEED=seed, output_limit=None, dropout=model_cfg.dropout)

    return fm

def VI_nn_constructor():
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
    
    trainer = LatentTrainer(trainer_params,
                                model,
                                dataset_train,
                                dataset_holdout)
    return trainer

# def occupancy_predictor_nn_constructor():
#     checkpoint_filepath = "/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_1/lightning_logs/version_0/checkpoints"
#     checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
#     cfg = DotMap()
#     cfg.seed = 1
#     cfg.lr = 0.00005 # more_layers: 0.00005, one layer: 0.0001
#     cfg.if_cuda = True
#     cfg.gamma = 0.5
#     cfg.log_dir = 'logs'
#     cfg.num_workers = 8
#     cfg.model_name = 'Occupancy_predictor'
#     cfg.lr_schedule = [400,800]
#     cfg.num_gpus = 1
#     cfg.epochs = 1000
#     cfg.dof = 6
#     cfg.coord_system = 'cartesian'
#     cfg.tag = '2agents_all'
#     seed(cfg)
#     seed_everything(cfg.seed)

#     log_dir = '/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_{0}'.format(cfg.seed)

#     self.occupancy_model = Predictor_Model_2d(lr=cfg.lr,
#                             dof=cfg.dof,
#                             if_cuda=cfg.if_cuda,
#                             if_test=True,
#                             gamma=cfg.gamma,
#                             log_dir=log_dir,
#                             num_workers=cfg.num_workers,
#                             coord_system=cfg.coord_system,
#                             lr_schedule=cfg.lr_schedule)

#     ckpt = torch.load(checkpoint_filepath)
#     self.occupancy_model.load_state_dict(ckpt['state_dict'])
#     self.occupancy_model = self.occupancy_model.to('cuda')
#     self.occupancy_model.eval()
#     self.occupancy_model.freeze()