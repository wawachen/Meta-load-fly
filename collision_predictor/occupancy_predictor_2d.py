import os
import torch
import numpy as np
from torch import nn
# from utils import common
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from pointcloud_dataset import DataModel
from neural_nets import OccupancyMLPQueryModel
from utils import gradient_f
from collections import OrderedDict


class Predictor_Model_2d(pl.LightningModule):

    def __init__(self,
                 lr: float=5e-5,
                 dof: int=5,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 num_workers: int=8,
                 coord_system: str='cartesian',
                 lr_schedule: list=[100000]) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}

        self.__build_model()

    def __build_model(self):
        # model
        self.model = OccupancyMLPQueryModel(in_channels=int(2+self.hparams.dof), out_channels=1, hidden_features=256)

        # loss
        self.loss_func = self.siren_sdf_loss

    def siren_sdf_loss(self, model_output, gt):
        gt_sdf = gt['sdf'].reshape(-1, 1)

        pred_sdf = model_output['model_out']

        loss = ((pred_sdf - gt_sdf)**2).mean()
        return loss
    
    def train_forward(self, data):
        data['coords'] = data['coords'].reshape(-1, 2)
        coords_org = data['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        states = data['states'].reshape(-1, self.hparams.dof)
        output = self.model(torch.cat((coords, states), dim=1))
        pred = {'model_in': coords_org, 'model_out': output}
        return pred

    def training_step(self, batch, batch_idx):
        data, target = batch
        
        pred = self.train_forward(data)
        train_loss = self.loss_func(pred, target)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return train_loss

    def validation_step(self,batch, batch_idx):
        data, target = batch
        
        pred = self.train_forward(data)

        if batch_idx == 10:
            N = 401
            val_im_pre = pred['model_out'].clone().detach().cpu().numpy().reshape(N,N)
            val_im_gt = target['sdf'].reshape(-1, 1).clone().detach().cpu().numpy().reshape(N,N)
            img_pre = np.reshape(val_im_pre, (1, N, N))
            im_gt = np.reshape(val_im_gt, (1, N, N))
            
            self.logger.experiment.add_image('pre_images', img_pre, self.current_epoch)
            self.logger.experiment.add_image('gt_images', im_gt, self.current_epoch)
        #     # mdic = {"image_pre":img_pre,"image_gt":im_gt}
        #     # scipy.io.savemat("/home/wawa/catkin_meta/src/MBRL_transport/val_im_epoch{0}.mat".format(self.current_epoch),mdic)

        val_loss = self.loss_func(pred, target)
        self.log('val_loss', val_loss)

    def test_step(self,batch, batch_idx):
        data, target = batch
        
        pred = self.train_forward(data)
        test_loss = self.loss_func(pred, target)
        self.log('test_loss', test_loss, on_step=True, on_epoch=False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


class MAML(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module # The module who's weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static': 
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter': # As proposed in "Meta-SGD".
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, context_dict):
        """Specializes the model"""
        x = context_dict.get('x').cuda()
        y = context_dict.get('y').cuda()

        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            # First, replicate the initialization for each batch item.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the 
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions)

                if loss > prev_loss:
                    print('inner lr too high?')
                
                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, meta_batch, **kwargs):
        # The meta_batch conists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)
        context = meta_batch['context']
        query_x = meta_batch['query']['x'].cuda()

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(context)

        # Compute the final outputs.
        model_output = self.hypo_module(query_x, params=fast_params)
        out_dict = {'model_out':model_output, 'intermed_predictions':intermed_predictions}

        return out_dict,fast_params

