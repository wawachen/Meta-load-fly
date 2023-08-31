import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
import numpy as np

from neural_nets import Learner
from copy import deepcopy
import argparse
import os
from time import localtime, strftime
import rospy

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.fast_adapted_params = None

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        
        self.model_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/model_3d",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        # self.online_train_meta = args.online_train_meta

        if args.load_model:            
            test_model_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/model_3d")
            if os.path.exists(test_model_path + '/params.pkl'):
                self.initialise_networks(test_model_path+'/params.pkl')
                print('Agent successfully loaded meta_network: {}'.format(test_model_path + '/params.pkl'))


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry, i_ter):
        """

        :param x_spt:   [b, setsz, nninput]
        :param y_spt:   [b, setsz, output]
        :param x_qry:   [b, querysz, nninput]
        :param y_qry:   [b, querysz, output]
        :return:
        """
        task_num, setsz, input_dim = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]  # record accuracy
        # print(self.update_step)

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)

            #self.loss = tf.reduce_mean(tf.square(self.delta_ph - self.delta_pred))
            loss = F.mse_loss(logits, y_spt[i]) #change to mean squared error
            grad = torch.autograd.grad(loss, self.net.parameters())
            #separate weights to not influence meta update
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # print(fast_weights)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[0] += loss_q

                correct = loss_q.item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                # print(x_qry[i])
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry[i])
                # print(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                correct = loss_q.item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    correct = loss_q.item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        if i_ter%100==0:
            self.save_model(i_ter)

        accs = np.array(corrects) / task_num

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        in meta test, we firstly adapt our model into the new scenario and use this model to do the control part
        :param x_spt:   [setsz, nn_input]
        :param y_spt:   [setsz, nn_output]
        :param x_qry:   [querysz, nn_input]
        :param y_qry:   [querysz, nn_output]
        :return:
        """
        assert len(x_spt.shape) == 2

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        # here we just use the default parameter to forward test of NN
        logits = net(x_spt)
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # scalar
            correct = F.mse_loss(logits_q, y_qry).item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # scalar
            correct = F.mse_loss(logits_q, y_qry).item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)

            with torch.no_grad():
                correct = loss_q.item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects)

        return accs

    def pre_forward(self, x):
        #input [8000,dim] 8000 is the sum of the sampled particles
        #for meta test time, the meta offline model will be used firstly before using the adapted model 
        with torch.no_grad():
            predictions = self.net(x)

        return predictions

    # def adapt(self,x,y):
    #     #[k_shot, dim]
    #     # losses = [0 for _ in range(self.update_adapt_step + 1)]  # record loss on steo i
    #     # corrects = [0 for _ in range(self.update_adapt_step + 1)]  # record accuracy
    #     # self.fast_adapted_params = None
        
    #     #firstly we only consider one step update for efficiency
    #     logits = self.net(x, vars=self.fast_adapted_params, bn_training=True)
    #     # logits = self.net(x, vars=None, bn_training=True)
    #     loss = F.mse_loss(logits, y)

    #     # grad = torch.autograd.grad(loss, self.net.parameters())
    #     if self.fast_adapted_params == None:
    #         grad = torch.autograd.grad(loss, self.net.parameters())
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
    #     else:
    #         grad = torch.autograd.grad(loss, self.fast_adapted_params)
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.fast_adapted_params)))

    #     #separate weights to not influence meta update
    #     # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

    #     for k in range(1, self.update_step):
    #         logits = self.net(x, fast_weights, bn_training=True)
    #         loss = F.mse_loss(logits, y)
    #         # 2. compute grad on theta_pi
    #         grad = torch.autograd.grad(loss, fast_weights)
    #         # 3. theta_pi = theta_pi - train_lr * grad
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

    #     self.fast_adapted_params = fast_weights

    def adapt(self,x,y):
        #[k_shot, dim]
        # losses = [0 for _ in range(self.update_adapt_step + 1)]  # record loss on steo i
        # corrects = [0 for _ in range(self.update_adapt_step + 1)]  # record accuracy
        self.fast_adapted_params = None
        
        #firstly we only consider one step update for efficiency
        logits = self.net(x, vars=None, bn_training=True)
        loss = F.mse_loss(logits, y)

        grad = torch.autograd.grad(loss, self.net.parameters())
        
        #separate weights to not influence meta update
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        for k in range(1, self.update_step):
            logits = self.net(x, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        self.fast_adapted_params = fast_weights

    def post_forward(self, x):
        #[k_shot, dim]
        assert not(self.fast_adapted_params==None)
        #input [8000,dim] 8000 is the sum of the sampled particles
        with torch.no_grad():
            logits_q = self.net(x, self.fast_adapted_params, bn_training=True)
            
        return logits_q

########################################################
# warning! this function is only for testing in the offline_meta.py
    def single_tunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        in meta test, we firstly adapt our model into the new scenario and use this model to do the control part
        :param x_spt:   [setsz, nn_input]
        :param y_spt:   [setsz, nn_output]
        :param x_qry:   [querysz, nn_input]
        :param y_qry:   [querysz, nn_output]
        :return:
        """
        assert len(x_spt.shape) == 2

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # net = deepcopy(self.net)

        spt = 15
        qry = 15

        # 1. run the i-th task and compute loss for k=0
        # here we just use the default parameter to forward test of NN
        logits = self.net(x_spt[:spt])
        loss = F.mse_loss(logits, y_spt[:spt])
        grad = torch.autograd.grad(loss, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = self.net(x_qry[:qry], self.net.parameters(), bn_training=True)
            # scalar
            correct = F.mse_loss(logits_q, y_qry[:qry]).item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = self.net(x_qry[:qry], fast_weights, bn_training=True)
            # scalar
            correct = F.mse_loss(logits_q, y_qry[:qry]).item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = self.net(x_spt[:spt], fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt[:spt])
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = self.net(x_qry[:qry], fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry[:qry])

            with torch.no_grad():
                correct = loss_q.item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        # del net

        accs = np.array(corrects)

        return accs

########################################################
    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.net.load_state_dict(checkpoint['meta_params'])    # actor parameters
        self.meta_optim.load_state_dict(checkpoint['meta_optim_params']) # critic optimiser state
        
    def save_model(self, train_step, model_path=None):
        num = str(train_step)

        if model_path == None:
            model_path = self.model_path
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_dict = {'meta_params' : self.net.state_dict(),
                    'meta_optim_params' : self.meta_optim.state_dict()}

        torch.save(save_dict, model_path + '/' + num + '_params.pkl')


def main():
    #3 hidden layers of 512 units with ReLU
    obs_dim = 4 
    act_dim = 2

    #config [out,in]
    config = [
        ('linear', [512, obs_dim+1+act_dim]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('relu', [True]),
        ('linear', [obs_dim, 512]),
    ]

    device = torch.device('cpu')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    maml = Meta(args, config).to(device)
    print(maml)


if __name__ == '__main__':
    main()
