import pytorch_lightning as pl
from pointcloud_dataset import DataModel_2d
import torch

class PointcloudModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        data_filepath1_train = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.0_y0.0_2agents_L0.6/preprocess/train"
        data_filepath2_train = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.3_y0.0_2agents_L1.0/preprocess/train"
        data_filepath3_train = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.5_y0.0_2agents_L0.8/preprocess/train"
        data_filepath4_train = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.8_y0.0_2agents_L1.2/preprocess/train"

        data_filepath1_val = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.0_y0.0_2agents_L0.6/preprocess/validation"
        data_filepath2_val = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.3_y0.0_2agents_L1.0/preprocess/validation"
        data_filepath3_val = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.5_y0.0_2agents_L0.8/preprocess/validation"
        data_filepath4_val = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.8_y0.0_2agents_L1.2/preprocess/validation"
        #test data
        data_filepath5 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x1.0_y0.0_2agents_L0.8/preprocess"
        data_filepath6 = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d_wind_x0.6_y0.0_2agents_L1.4/preprocess"
        
        self.train_data_filepath = [data_filepath1_train,data_filepath2_train,data_filepath3_train,data_filepath4_train]
        self.val_data_filepath = [data_filepath1_val,data_filepath2_val,data_filepath3_val,data_filepath4_val]
        self.test_data_filepath = [data_filepath1_val] #data_filepath1_val,data_filepath5,data_filepath6

        if_cuda = 1
        self.kwargs = {'num_workers': 8, 'pin_memory': True} if if_cuda else {}

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = DataModel_2d(flag='train', path=self.train_data_filepath)
            self.val_dataset = DataModel_2d(flag="validation",path=self.val_data_filepath)
        if stage == 'test':
            self.test_dataset = DataModel_2d(flag='test', path=self.test_data_filepath)

    def train_dataloader(self):
        # batch size cannot be more than 3
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                       batch_size=55,
                                                       shuffle=True,
                                                       **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       **self.kwargs)
        return test_loader


