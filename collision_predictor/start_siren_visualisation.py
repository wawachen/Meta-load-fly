import os
import sys
import glob
import yaml
import time
import torch
from occupancy_predictor_2d import Predictor_Model_2d
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from dotmap import DotMap
import json
import numpy as np
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def eval():
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
    cfg.lr_schedule = [10000000]
    cfg.num_gpus = 1
    cfg.epochs = 10
    cfg.dof = 6
    cfg.coord_system = 'cartesian'
    cfg.tag = '2agents_all'
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '/home/wawa/catkin_meta/src/MBRL_transport/logs_Occupancy_predictor_2d_movementall_{0}'.format(cfg.seed)

    model = Predictor_Model_2d(lr=cfg.lr,
                             dof=cfg.dof,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             num_workers=cfg.num_workers,
                             coord_system=cfg.coord_system,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()
    
    data_filepath1_train = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.0_y0.0_2agents_L0.6/preprocess/validation"
    data_filepath5 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x1.0_y0.0_2agents_L0.8/preprocess"
    data_filepath6 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.6_y0.0_2agents_L1.4/preprocess"
    # data_filepath5 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.0_y0.0_2agents_L0.6/preprocess/train"
    # data_filepath6 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.3_y0.0_2agents_L1.0/preprocess/train"    

    test_data_filepath = [data_filepath5] # data_filepath1_train,data_filepath5,data_filepath6
    # get test file ids
    filelist = []
    for i in range(len(test_data_filepath)):
        DIR = test_data_filepath[i]
        for name in os.listdir(DIR):
            filelist.append(os.path.join(DIR, name))

    val_loss_all = []

    for idx in tqdm(filelist):
        fig, axes = plt.subplots(1,4, figsize=(30,6))
        fig.tight_layout()
        plt.subplots_adjust(wspace =0.08, hspace =0)
        ax_titles = ['', '', '','']
        # get testing robot states
        data = scipy.io.loadmat(idx)
        robot_state = data['configuration'] 
        sel_robot_state = np.array(robot_state).reshape(1, -1)
        ground_truth_sdf = data['sdf']

        N=401
        # max_batch=64 ** 2
        start = time.time()
        
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [0, 2]
        voxel_size = 4.0 / (N - 1)

        overall_index = torch.arange(0, N ** 2, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 2, 3)

        # transform first 2 columns to be the x, y index
        samples[:, 0] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N

        # transform first 3 columns to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
        samples[:, 1] = -(samples[:, 1] * voxel_size) + voxel_origin[1]
    
        num_samples = N ** 2
        samples.requires_grad = False

        max_x = 4
        max_y = 2
        sample_test = samples[:,:2].cuda()
        sample_test[:,0] = sample_test[:,0]/max_x
        sample_test[:,1] = sample_test[:,1]/max_y

        # print(sel_robot_state)
        print(sel_robot_state[0,(0,1,3,4,6,7)])

        final_robot_states = np.tile(sel_robot_state[0,(0,1,3,4,6,7)], (sample_test.shape[0], 1))
        final_robot_states = torch.from_numpy(final_robot_states).float().cuda()
        sample_set = torch.cat((sample_test, final_robot_states), dim=1)
        samples[:, 2] = (model.model(sample_set).squeeze().detach().cpu())

        sdf_values = samples[:, 2]

        sdf_values_np_1 = sdf_values.detach().cpu().numpy()
        sdf_values_im_1 = sdf_values_np_1.reshape(N, N)
        img_1 = sdf_values_im_1

        axes[0].set_axis_off()
        axes[0].imshow(img_1)
        axes[0].set_title(ax_titles[0], fontsize=25)

        img_2 = ground_truth_sdf.reshape(N,N)
        
        axes[1].set_axis_off()
        axes[1].imshow(img_2)
        axes[1].set_title(ax_titles[1], fontsize=25)

        # print(sdf_values)
        sdf_off_index = np.where(sdf_values>0.01)[0]
        sdf_on_index = np.where(sdf_values<=0.01)[0]
        sdf_values[sdf_on_index] = 1
        sdf_values[sdf_off_index] = 0

        end = time.time()
        print("sampling takes: %f" % (end - start))

        # mdict = {"sdf":sdf_values.detach().cpu().numpy()}    
        # scipy.io.savemat(ply_filename + ".mat", mdict)
        # extract_points_on_surface(sdf_values,0.0001,voxel_origin,voxel_size,ply_filename)
        sdf_values_np = sdf_values.detach().cpu().numpy()
        sdf_values_im = sdf_values_np.reshape(N, N)
        img_3 = sdf_values_im
        
        axes[2].set_axis_off()
        axes[2].imshow(img_3)
        axes[2].set_title(ax_titles[2], fontsize=25)

        sdf_off_index = np.where(ground_truth_sdf>0)[0]
        sdf_on_index = np.where(ground_truth_sdf<=0)[0]
        ground_truth_sdf[sdf_on_index] = 1
        ground_truth_sdf[sdf_off_index] = 0
        img_4 = ground_truth_sdf.reshape(N,N)
        
        axes[3].set_axis_off()
        axes[3].imshow(img_4)
        axes[3].set_title(ax_titles[3], fontsize=25)

        plt.show()
        
        # sdf_predict = sdf_values_np.reshape(-1,1)
        # # print(ground_truth_sdf.shape)
        # val_loss = ((sdf_predict - ground_truth_sdf)**2).mean()
        # val_loss_all.append(val_loss)
        # sample_index = samples[:,:2].detach().cpu().numpy()
        # mdic = {"val_loss":val_loss,"x":sample_index[:,0].reshape(N,N),"y":sample_index[:,1].reshape(N,N),"predicted_map":img,"groundt_map":img1,"configuration":robot_state,"top":points}
        # s_p = '/'.join(idx.split('/')[:-2])
        # fld = idx.split("/")[-1]
        # if eval_type == "test_data":
        #     if not os.path.exists(s_p+'/predictions_test_data'):
        #         os.makedirs(s_p+'/predictions_test_data')
        
        #     scipy.io.savemat(s_p+'/predictions_test_data/'+fld,mdic)
        # else:
        #     if not os.path.exists(s_p+'/predictions_val_data'):
        #         os.makedirs(s_p+'/predictions_val_data')
           
        #     scipy.io.savemat(s_p+'/predictions_val_data/'+fld,mdic)

    # mdic1 = {"loss_all":val_loss_all}
    # if eval_type == "test_data":
    #     scipy.io.savemat("/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/all_test_loss.mat", mdic1)
    # else:
    #     scipy.io.savemat("/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations/all_val_loss.mat", mdic1)

if __name__ == '__main__':
   eval()