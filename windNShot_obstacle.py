import  os.path
from matplotlib import transforms
import  numpy as np
import scipy.io as sio
import rospy


class WindNShot_obs:

    def __init__(self, batch_num, batchsz, path_length, n_way, k_shot, k_query, sequential=False,integrated=False):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        self.sequential = sequential
        # Here we prepare the dataset for offline meta training
        
        mat_contents_x1 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.0_2agents_L0.6_dt_0.15_obstacle.mat")
        data_obs_x1 = mat_contents_x1['obs']  # [10001,4]
        data_acs_x1 = mat_contents_x1['acs']  # [10000,2]

        # for each dataset this property is the same
        data_all_num = data_obs_x1.shape[0]
        assert data_all_num == 2501

        mat_contents_x2 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x1.0_2agents_L0.8_dt_0.15_obstacle.mat")
        data_obs_x2 = mat_contents_x2['obs']  # [10001,4]
        data_acs_x2 = mat_contents_x2['acs']  # [10000,2]

        mat_contents_x3 = sio.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/firefly_data_3d_wind_x0.6_2agents_L1.4_dt_0.15_obstacle.mat")
        data_obs_x3 = mat_contents_x3['obs']  # [10001,4]
        data_acs_x3 = mat_contents_x3['acs']  # [10000,2]

        #we split 10001 sampling points into  (M-1)/(N-1) trajectories, M is total points, N is the path_length
        l = path_length # k_shot+k_query should be equal or less than l
        num_of_trajectories = int((data_all_num-1)/(l-1)) 

        trajs1_obs = []
        trajs1_acs = []  
        for i in range(num_of_trajectories):
            trajs1_obs.append(data_obs_x1[(l-1)*i:(l-1)*i+l])
            trajs1_acs.append(data_acs_x1[(l-1)*i:(l-1)*(i+1)])

        new_train_in1, new_train_targs1 = [], []
        for obs, acs in zip(trajs1_obs, trajs1_acs):
            new_train_in1.append(np.concatenate([self.obs_preproc_3d(obs[:-1]), acs], axis=-1))
            new_train_targs1.append(self.targ_proc(obs[:-1], obs[1:]))
        train_in1 = np.array(new_train_in1)
        train_targs1 = np.array(new_train_targs1)

        trajs2_obs = []
        trajs2_acs = []  
        for i in range(num_of_trajectories):
            trajs2_obs.append(data_obs_x2[(l-1)*i:(l-1)*i+l])
            trajs2_acs.append(data_acs_x2[(l-1)*i:(l-1)*(i+1)])

        new_train_in2, new_train_targs2 = [], []
        for obs, acs in zip(trajs2_obs, trajs2_acs):
            new_train_in2.append(np.concatenate([self.obs_preproc_3d(obs[:-1]), acs], axis=-1))
            new_train_targs2.append(self.targ_proc(obs[:-1], obs[1:]))
        train_in2 = np.array(new_train_in2)
        train_targs2 = np.array(new_train_targs2)

        trajs3_obs = []
        trajs3_acs = []  
        for i in range(num_of_trajectories):
            trajs3_obs.append(data_obs_x3[(l-1)*i:(l-1)*i+l])
            trajs3_acs.append(data_acs_x3[(l-1)*i:(l-1)*(i+1)])

        new_train_in3, new_train_targs3 = [], []
        for obs, acs in zip(trajs3_obs, trajs3_acs):
            new_train_in3.append(np.concatenate([self.obs_preproc_3d(obs[:-1]), acs], axis=-1))
            new_train_targs3.append(self.targ_proc(obs[:-1], obs[1:]))
        train_in3 = np.array(new_train_in3)
        train_targs3 = np.array(new_train_targs3)

        x_all = np.concatenate([train_in1,train_in2,train_in3], axis=0) #[trajectory_num,path length, dim]
        y_all = np.concatenate([train_targs1,train_targs2,train_targs3], axis=0)

        #all conditions are permutated
        #we shuffle different paths here because we regard each path as different tasks, and k_shot and k_query are sampled from each path
        perm_all = np.random.permutation(x_all.shape[0])
        x_all = x_all[perm_all]
        y_all = y_all[perm_all]

        self.NN_input = x_all.shape[2]
        self.NN_output = y_all.shape[2]

        train_index = int(x_all.shape[0]*0.8)
        
        self.x_train, self.x_test = x_all[:train_index], x_all[train_index:]
        self.y_train, self.y_test = y_all[:train_index], y_all[train_index:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = x_all.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.l = l
        self.batch_num = batch_num
        self.integrated = integrated

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test, "train_label":self.y_train, "test_label":self.y_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        #different from datasets, its path_length is varied
        self.running_indexes = {"train": 0, "test": 0}
        self.running_datasets = dict(train=[],test=[],train_label=[],test_label=[])
        self.has_been_updated = False 
        self.running_cache = dict(train=0,test=0)
        self.running_samples_num = 0

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"],self.datasets["train_label"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"],self.datasets["test_label"])}


    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
    def add_roll_outs(self, trajs1_obs, trajs1_acs):
        #[[path_length,dim],]
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(trajs1_obs, trajs1_acs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
            self.running_samples_num += acs.shape[0]
        #as the path_length may be different

        index = len(new_train_in)
        train_index = int(0.8*index)

        print("adding samples: train:{0},test:{1}".format(train_index,index-train_index))
        #add new data into running dataset
        self.running_datasets['train'] = self.running_datasets['train'] + new_train_in[:train_index]
        self.running_datasets['train_label'] = self.running_datasets['train_label'] + new_train_targs[:train_index]
        self.running_datasets['test'] = self.running_datasets['test'] + new_train_in[train_index:]
        self.running_datasets['test_label'] = self.running_datasets['test_label'] + new_train_targs[train_index:]
        
        #we can add rate here to control the update of the running dataset
        if not self.integrated:
            batch_r_num = max(int((self.running_samples_num*0.8)/(self.batchsz*self.n_way*(self.k_shot+self.k_query))),1)
            batch_r_num1 = max(int((self.running_samples_num*0.2)/(self.batchsz*self.n_way*(self.k_query+self.k_query))),1)
            self.running_cache = {"train":self.load_running_data_cache(self.running_datasets['train'],self.running_datasets['train_label'],batch_r_num), "test":self.load_running_data_cache(self.running_datasets['test'],self.running_datasets['test_label'],batch_r_num1)}
        else:
            self.running_cache = {"train":self.load_running_data_cache(self.running_datasets['train'],self.running_datasets['train_label'],data_pack=self.datasets["train"],data_pack_label=self.datasets["train_label"]), "test":self.load_running_data_cache(self.running_datasets['test'],self.running_datasets['test_label'],data_pack=self.datasets["test"],data_pack_label=self.datasets["test_label"])}

    def load_running_data_cache(self, data_pack_run, data_pack_label_run, running_batch_num=None, data_pack = None, data_pack_label=None):
        #  [[path_len,dim],[path_len1,dim]]
        # will be running when new data are added
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        if running_batch_num == None:
            running_batch_num = 20

        for sample in range(running_batch_num):  # num of episodes
            
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                
                if not self.integrated:
                    # print(len(data_pack_run))
                    assert len(data_pack_run)>self.n_way
                    selected_cls = np.random.choice(len(data_pack_run), self.n_way, False)
                else:
                    assert len(data_pack_run)+data_pack.shape[0]>self.n_way
                    selected_cls = np.random.choice(len(data_pack_run)+data_pack.shape[0], self.n_way, False)
        
                if self.sequential:
                    idx_batch = np.zeros(self.n_way,dtype=int)
                    for index in range(self.n_way):
                        if not self.integrated:
                            idx_batch[index] = np.random.randint(self.k_shot, data_pack_run[selected_cls[index]].shape[0] - self.k_query)
                        else:
                            if selected_cls[index] >= len(data_pack_run):
                                idx_batch[index] = np.random.randint(self.k_shot, data_pack[selected_cls[index]-len(data_pack_run)].shape[0] - self.k_query)
                            else:
                                idx_batch[index] = np.random.randint(self.k_shot, data_pack_run[selected_cls[index]].shape[0] - self.k_query)

                for j, cur_class in enumerate(selected_cls):

                    #k_shot and k_query is sequentially connected or not
                    if not self.sequential:
                        if not self.integrated:
                            selected_traj = np.random.choice(data_pack_run[cur_class].shape[0], self.k_shot + self.k_query, False)
                            # meta-training and meta-test
                            x_spt.append(data_pack_run[cur_class][selected_traj[:self.k_shot]])
                            x_qry.append(data_pack_run[cur_class][selected_traj[self.k_shot:]])
                            y_spt.append(data_pack_label_run[cur_class][selected_traj[:self.k_shot]])
                            y_qry.append(data_pack_label_run[cur_class][selected_traj[self.k_shot:]])
                        else:
                            if cur_class>=len(data_pack_run):
                                selected_traj = np.random.choice(data_pack[cur_class-len(data_pack_run)].shape[0], self.k_shot + self.k_query, False)
                                # meta-training and meta-test
                                x_spt.append(data_pack[cur_class-len(data_pack_run)][selected_traj[:self.k_shot]])
                                x_qry.append(data_pack[cur_class-len(data_pack_run)][selected_traj[self.k_shot:]])
                                y_spt.append(data_pack_label[cur_class-len(data_pack_run)][selected_traj[:self.k_shot]])
                                y_qry.append(data_pack_label[cur_class-len(data_pack_run)][selected_traj[self.k_shot:]])
                            else:
                                selected_traj = np.random.choice(data_pack_run[cur_class].shape[0], self.k_shot + self.k_query, False)
                                # meta-training and meta-test
                                x_spt.append(data_pack_run[cur_class][selected_traj[:self.k_shot]])
                                x_qry.append(data_pack_run[cur_class][selected_traj[self.k_shot:]])
                                y_spt.append(data_pack_label_run[cur_class][selected_traj[:self.k_shot]])
                                y_qry.append(data_pack_label_run[cur_class][selected_traj[self.k_shot:]])

                    else:
                        # meta-training and meta-test
                        if not self.integrated:
                            x_spt.append(data_pack_run[cur_class][idx_batch[j]-self.k_shot:idx_batch[j]])
                            x_qry.append(data_pack_run[cur_class][idx_batch[j]:idx_batch[j]+self.k_query])
                            y_spt.append(data_pack_label_run[cur_class][idx_batch[j]-self.k_shot:idx_batch[j]])
                            y_qry.append(data_pack_label_run[cur_class][idx_batch[j]:idx_batch[j]+self.k_query])
                        else:
                            if cur_class>=len(data_pack_run):
                                # print(cur_class-len(data_pack_run),idx_batch[j]-self.k_shot)
                                x_spt.append(data_pack[cur_class-len(data_pack_run)][idx_batch[j]-self.k_shot:idx_batch[j]])
                                x_qry.append(data_pack[cur_class-len(data_pack_run)][idx_batch[j]:idx_batch[j]+self.k_query])
                                y_spt.append(data_pack_label[cur_class-len(data_pack_run)][idx_batch[j]-self.k_shot:idx_batch[j]])
                                y_qry.append(data_pack_label[cur_class-len(data_pack_run)][idx_batch[j]:idx_batch[j]+self.k_query])
                            else:
                                x_spt.append(data_pack_run[cur_class][idx_batch[j]-self.k_shot:idx_batch[j]])
                                x_qry.append(data_pack_run[cur_class][idx_batch[j]:idx_batch[j]+self.k_query])
                                y_spt.append(data_pack_label_run[cur_class][idx_batch[j]-self.k_shot:idx_batch[j]])
                                y_qry.append(data_pack_label_run[cur_class][idx_batch[j]:idx_batch[j]+self.k_query])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.NN_input)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, self.NN_output)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.NN_input)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, self.NN_output)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, dim]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.NN_input)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batchsz, setsz, self.NN_output)
            # [b, qrysz, dim]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.NN_input)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.NN_output)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def meta_next(self,mode='train'):
        # update cache if indexes is larger cached num
        #################
        # update cache if indexes is larger cached num
        if self.integrated:
            if self.running_indexes[mode] >= len(self.running_cache[mode]):
                self.running_indexes[mode] = 0
                mode_label = mode+"_label"
                self.running_cache[mode] = self.load_running_data_cache(self.running_datasets[mode],self.running_datasets[ mode_label],data_pack=self.datasets[mode],data_pack_label=self.datasets[mode_label])
        else:
            batch_r_num = max(int((self.running_samples_num*0.8)/(self.batchsz*self.n_way*(self.k_shot+self.k_query))),1)
            batch_r_num1 = max(int((self.running_samples_num*0.2)/(self.batchsz*self.n_way*(self.k_query+self.k_query))),1)

            if self.running_indexes[mode] >= len(self.running_cache[mode]):
                self.running_indexes[mode] = 0
                mode_label = mode+"_label"
                if mode == 'train':
                    self.running_cache[mode] = self.load_running_data_cache(self.running_datasets[mode],self.running_datasets[mode_label],batch_r_num)
                else:
                    self.running_cache[mode] = self.load_running_data_cache(self.running_datasets[mode],self.running_datasets[mode_label],batch_r_num1)

        next_batch = self.running_cache[mode][self.running_indexes[mode]]
        self.running_indexes[mode] += 1
        
        return next_batch


    def load_data_cache(self, data_pack, data_pack_label):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [4*num_traj*p, l-1, 5+2]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 3 way 1 shot as example: 3 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []
        len_path = data_pack.shape[1] 
        assert len_path>self.k_shot+self.k_query

        for sample in range(self.batch_num):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                if self.sequential:
                    idx_batch = np.random.randint(self.k_shot, len_path - self.k_query+1, size=self.n_way)

                for j, cur_class in enumerate(selected_cls):
                    assert data_pack.shape[1] == (self.l-1)

                    #k_shot and k_query is sequentially connected or not
                    if not self.sequential:
                        selected_traj = np.random.choice(data_pack.shape[1], self.k_shot + self.k_query, False)
                        # meta-training and meta-test
                        x_spt.append(data_pack[cur_class][selected_traj[:self.k_shot]])
                        x_qry.append(data_pack[cur_class][selected_traj[self.k_shot:]])
                        y_spt.append(data_pack_label[cur_class][selected_traj[:self.k_shot]])
                        y_qry.append(data_pack_label[cur_class][selected_traj[self.k_shot:]])
                    else:
                        # meta-training and meta-test
                        x_spt.append(data_pack[cur_class][idx_batch[j]-self.k_shot:idx_batch[j]])
                        x_qry.append(data_pack[cur_class][idx_batch[j]:idx_batch[j]+self.k_query])
                        y_spt.append(data_pack_label[cur_class][idx_batch[j]-self.k_shot:idx_batch[j]])
                        y_qry.append(data_pack_label[cur_class][idx_batch[j]:idx_batch[j]+self.k_query])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.NN_input)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, self.NN_output)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.NN_input)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, self.NN_output)[perm]

                #[b, setsz, dim]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, dim]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.NN_input)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batchsz, setsz, self.NN_output)
            # [b, qrysz, dim]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.NN_input)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.NN_output)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            mode_label = mode+"_label"
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode],self.datasets[mode_label])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
           return np.concatenate([np.sin(obs[:, 3]).reshape(-1,1), np.cos(obs[:, 3]).reshape(-1,1), obs[:, :3]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, 3].sin().reshape(-1,1),
                obs[:, 3].cos().reshape(-1,1),
                obs[:, :3],
            ], dim=1)

    @staticmethod
    def obs_preproc_3d(obs):
        if isinstance(obs, np.ndarray): 
           return np.concatenate([obs[:, :9],np.sin(obs[:, 9]).reshape(-1,1), np.cos(obs[:, 9]).reshape(-1,1),np.sin(obs[:, 10]).reshape(-1,1), np.cos(obs[:, 10]).reshape(-1,1),np.sin(obs[:, 11]).reshape(-1,1), np.cos(obs[:, 11]).reshape(-1,1)], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, :9],
                obs[:, 9].sin().reshape(-1,1),
                obs[:, 9].cos().reshape(-1,1),
                obs[:, 10].sin().reshape(-1,1),
                obs[:, 10].cos().reshape(-1,1),
                obs[:, 11].sin().reshape(-1,1),
                obs[:, 11].cos().reshape(-1,1),
            ], dim=1)


    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs




if __name__ == '__main__':

    import  time
    import  torch

    db = WindNShot(50, 22, 1001, 4, 50, 50)

    for i in range(10):
        x_spt, y_spt, x_qry, y_qry = db.next('train')

        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, input_dim = x_spt.size()
        # print(y_spt.shape)
        # print(y_qry.shape)

        print("batch",batchsz,"setsize",setsz, "input",input_dim)
        time.sleep(5)

