from pointcloud_datamodule import PointcloudModule
from occupancy_predictor_2d import Predictor_Model_2d
from pytorch_lightning import Trainer, seed_everything
import torch
from dotmap import DotMap
from pytorch_lightning.strategies.ddp import DDPStrategy
import glob
import os
from pytorch_lightning import loggers as pl_loggers

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

if __name__ == '__main__':
    cfg = DotMap()
    cfg.seed = 500
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
    cfg.tag = '2d_movement'

    seed(cfg)
    seed_everything(cfg.seed)
    log_dir = '/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_{0}'.format(cfg.seed)

    # im_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir,name='im_logs')
    
    is_predict = 1
    dm = PointcloudModule()

    if not is_predict:
        
        model = Predictor_Model_2d(lr=cfg.lr,
                                dof=cfg.dof,
                                if_cuda=cfg.if_cuda,
                                if_test=False,
                                gamma=cfg.gamma,
                                log_dir=log_dir,
                                num_workers=cfg.num_workers,
                                coord_system=cfg.coord_system,
                                lr_schedule=cfg.lr_schedule)

        dm.setup('fit')
        
        trainer = Trainer(gpus=cfg.num_gpus,
                        max_epochs=cfg.epochs,
                        deterministic=True,
                        strategy=DDPStrategy(find_unused_parameters=False),
                        amp_backend='native',
                        default_root_dir=log_dir,check_val_every_n_epoch=10)

        trainer.fit(model, datamodule=dm)
    else:
        log_dir_test = '/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_{0}'.format(cfg.seed)
        dm.setup('test')
        
        checkpoint_filepath = "/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall1_1/lightning_logs/version_0/checkpoints"
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]

        model_r = Predictor_Model_2d(lr=cfg.lr,
                                dof=cfg.dof,
                                if_cuda=cfg.if_cuda,
                                if_test=True,
                                gamma=cfg.gamma,
                                log_dir=log_dir_test,
                                num_workers=cfg.num_workers,
                                coord_system=cfg.coord_system,
                                lr_schedule=cfg.lr_schedule)

        model = model_r.load_from_checkpoint(checkpoint_path=checkpoint_filepath)
        
        trainer = Trainer(gpus=1, limit_test_batches=0.45,
                        default_root_dir=log_dir_test)
        trainer.test(model=model, datamodule=dm)