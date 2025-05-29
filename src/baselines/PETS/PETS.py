from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
import numpy as np
from models.neural_nets import PtModel
import os
from time import localtime, strftime
import torch
import rospy

TORCH_DEVICE = torch.device('cuda')

class PETS_model(nn.Module):
    def __init__(self, cfg, log_path):
        super(PETS_model,self).__init__()
        self.net = PtModel(cfg.ensemble_size, cfg.model_in, cfg.model_out * 2).to(TORCH_DEVICE)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.model_path = log_path

        if cfg.load_model:
            load_path = cfg.load_model_path
            if os.path.exists(load_path + '/params.pkl'):
                self.initialise_networks(load_path+'/params.pkl')
                rospy.logwarn('Agent successfully loaded PETS_network: {}'.format(load_path + '/params.pkl'))
            else:
                rospy.logwarn('Failed to load model from PETS')

    def initialise_networks(self, path):
        checkpoint = torch.load(path) # load the torch data

        self.net.load_state_dict(checkpoint['PETS_params'])    # actor parameters
        self.optim.load_state_dict(checkpoint['PETS_optim_params']) # critic optimiser state
        
    def save_model(self, train_step):
        num = str(train_step)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'PETS_params' : self.net.state_dict(),
                    'PETS_optim_params' : self.optim.state_dict()}

        torch.save(save_dict, self.model_path + '/models/' + num + '_params.pkl') 
