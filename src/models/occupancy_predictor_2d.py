import torch
import numpy as np
import pytorch_lightning as pl
from neural_nets import OccupancyMLPQueryModel


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
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        
        return [optimizer], []
