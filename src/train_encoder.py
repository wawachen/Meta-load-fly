from neural_nets import Conv_autoencoder
import torch
from torch import nn as nn
from torch.nn import functional as F
from depth_dataset import Depth_dataset
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.io import savemat

EPOCH=100
BATCH_SIZE=128
LR=0.001
train_from_stratch = 0

all_path = []
path1 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.0_y0.0_2agents_L0.6"
path2 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.3_y0.0_2agents_L1.0"
path3 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.5_y0.0_2agents_L0.8"
path4 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.8_y0.0_2agents_L1.2"
all_path.append(path1)
all_path.append(path2)
all_path.append(path3)
all_path.append(path4)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

if train_from_stratch:
    train_dataset = Depth_dataset("train", all_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

    model = Conv_autoencoder().to(device)

    # training
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    loss_func=nn.MSELoss()
    
    loss_train = []
    model.train()
    for epoch in range(EPOCH):
        for step, batch_data in enumerate(train_loader):
            x = batch_data.cuda().float()
            y = batch_data.cuda().float()
            decoded=model(x)
            loss=loss_func(decoded,y)
            loss_train.append(loss.item())
            print("Epoch:{0},step:{1},loss:{2}".format(epoch,step,loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # fig, axes = plt.subplots(1,2, figsize=(30,6))
        # ax_titles = ['Predicted output', 
        #             'Ground Truth']
        # save_recon = decoded.clone().detach().cpu().numpy()
        # save_dep = y.clone().detach().cpu().numpy()
        # axes[0].set_axis_off()
        # axes[0].imshow(np.squeeze(save_recon[0,:,:]))
        # axes[0].set_title(ax_titles[0], fontsize=25)

        # axes[1].set_axis_off()
        # axes[1].imshow(np.squeeze(save_dep[0,:,:]))
        # axes[1].set_title(ax_titles[1], fontsize=25)
        
        # plt.show()

        model.save_encoder()
        model.save_decoder()

    save_path = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/validate_folder"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    loss_dic = {"loss_step":loss_train}
    savemat(save_path+"/train_loss.mat",loss_dic)
else:
    test_dataset = Depth_dataset("test", all_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=True)
    model = Conv_autoencoder().to(device)
    model.load_encoder()    
    model.load_decoder()    
    loss_func=nn.MSELoss()

    save_path = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/validate_folder"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_test = []
    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(test_loader):
            x = batch_data.cuda().float()
            outputs = model(x)
            loss = loss_func(outputs,x)
            loss_test.append(loss.item())
            save_recon = outputs.clone().detach().cpu().numpy()
            save_dep = x.clone().detach().cpu().numpy()
            dic_d ={"recons_depth":np.squeeze(save_recon),"depth":np.squeeze(save_dep)}
            savemat(save_path+"/{0}.mat".format(step),dic_d)

            # fig, axes = plt.subplots(1,2, figsize=(30,6))
            # ax_titles = ['Predicted output', 
            #             'Ground Truth']
            # axes[0].set_axis_off()
            # axes[0].imshow(np.squeeze(save_recon))
            # axes[0].set_title(ax_titles[0], fontsize=25)

            # axes[1].set_axis_off()
            # axes[1].imshow(np.squeeze(save_dep))
            # axes[1].set_title(ax_titles[1], fontsize=25)
            
            # plt.show()
        print("test loss:{0}".format(np.mean(np.array(loss_test))))


   
    