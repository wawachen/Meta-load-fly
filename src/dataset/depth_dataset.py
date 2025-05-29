from torch.utils.data import Dataset
import os
import numpy as np
import scipy

class Depth_dataset(Dataset):
    def __init__(self,mode,path):
        super().__init__()

        self.flag = mode
        self.depth_folder = path
        # self.on_surface_points = on_surface_points
        self.all_filelist = self.get_all_filelist(self.depth_folder)
        
    def get_all_filelist(self, file_path):
        # path joining version for other paths
        if self.flag == "train":
            DIR = file_path[0]
            file_num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

            DIR1 = file_path[1]
            file_num1 = len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])

            DIR2 = file_path[2]
            file_num2 = len([name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))])

            DIR3 = file_path[3]
            file_num3 = len([name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))])

            filelist = []
            # split data
            train_num = int(file_num*0.9)
            file_list = list(np.random.permutation(file_num)+510)
            train_file_list = file_list[:train_num]
            test_file_list = file_list[train_num:]

            train_num1 = int(file_num1*0.9)
            file_list1 = list(np.random.permutation(file_num1)+510)
            train_file_list1 = file_list1[:train_num1]
            test_file_list1 = file_list1[train_num1:]

            train_num2 = int(file_num2*0.9)
            file_list2 = list(np.random.permutation(file_num2)+510)
            train_file_list2 = file_list2[:train_num2]
            test_file_list2 = file_list2[train_num2:]

            train_num3 = int(file_num3*0.9)
            file_list3 = list(np.random.permutation(file_num3)+510)
            train_file_list3 = file_list3[:train_num3]
            test_file_list3 = file_list3[train_num3:]

            mdic = {"train":train_file_list, "test":test_file_list,"train1":train_file_list1, "test1":test_file_list1,\
            "train2":train_file_list2, "test2":test_file_list2,"train3":train_file_list3, "test3":test_file_list3}
            scipy.io.savemat("/home/wawa/catkin_meta/src/MBRL_transport/depth_images/split_file_list.mat",mdic)

        if self.flag == "train":
            id_lst_all = [train_file_list,train_file_list1,train_file_list2,train_file_list3]
        if self.flag == "test":
            filelist = []
            list_f = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/depth_images/split_file_list.mat")
            id_lst_all = [list_f["test"],list_f["test1"],list_f["test2"],list_f["test3"]]

        for i, id_lst in enumerate(id_lst_all):
            if self.flag == "train":
                for idx in id_lst:
                    filepath = os.path.join(file_path[i], f'{idx}.mat')
                    filelist.append(filepath)
            else:
                for idx in range(id_lst.shape[1]):
                    filepath = os.path.join(file_path[i], f'{id_lst[0,idx]}.mat')
                    filelist.append(filepath)

        return filelist

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):

        # =====> sdf
        data = scipy.io.loadmat(self.all_filelist[idx])
        depth = data["depth"]
        
        return depth[None]

if __name__ == '__main__':
    all_path = []
    path1 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.0_y0.0_2agents_L0.6"
    path2 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.3_y0.0_2agents_L1.0"
    path3 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.5_y0.0_2agents_L0.8"
    path4 = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images/wind_x0.8_y0.0_2agents_L1.2"
    all_path.append(path1)
    all_path.append(path2)
    all_path.append(path3)
    all_path.append(path4)

    data = Depth_dataset("train", all_path)
    print(f'data size is : {len(data)}')

    print(type(data[1])) 
