import torch
import numpy as np
from tqdm import trange
from baselines.FAMLE import famle
from dataset.task_generator_em import generate_task_data
import copy

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]

def offline_train_meta(model, dataset, cfg, logger):
    #first offline META training 
    for step in range(cfg.meta_epoch):

        x_spt, y_spt, x_qry, y_qry = dataset.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                        torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = model(x_spt, y_spt, x_qry, y_qry, step)

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)
            train_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
            logger.add_scalars('offline train accuracy', train_data_dic, step)

        if step % 500 == 0:
            accs = []
            for _ in range(1000//cfg.meta_task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = dataset.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                                torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            test_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
            logger.add_scalars('offline test accuracy', test_data_dic, step)

def offline_test_meta(model, db_train):
    accs = []
    for _ in range(10):
        # test
        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                        torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

        # split to single task each time
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            test_acc = model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            accs.append(test_acc)

    # [b, update_step+1]
    accs1 = np.array(accs).mean(axis=0).astype(np.float16)
    std_value = np.std(np.array(accs)[:,-1])
    print('Test acc:', accs1[-1])
    print('Test std:', std_value)

def offline_train_MBRL_pre(policy, epochs, obs_trajs, acs_trajs, logger):
    """Trains the internal model of this controller. Once trained,
    this controller switches from applying random actions to using MPC.

    Arguments:
        obs_trajs: A list of observation matrices, observations in rows.
        acs_trajs: A list of action matrices, actions in rows.

    Returns: None.
    """
    
    #action is real relative distance not normalized one
    train_in = np.concatenate([policy.obs_preproc_3d(obs_trajs[:-1]), acs_trajs], axis=-1)
    train_targs = policy.targ_proc(obs_trajs[:-1], obs_trajs[1:])

    ###########################
    # Train the pytorch model
    policy.model.net.fit_input_stats(train_in)

    idxs = np.random.randint(train_in.shape[0], size = [policy.model.net.num_nets, train_in.shape[0]])

    epochs = epochs

    # TODO: double-check the batch_size for all env is the same
    batch_size = 100

    epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
    num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

    for i in epoch_range:
        train_loss = 0 
        validate_loss = 0

        for batch_num in range(num_batch):
            batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

            loss = 0.01 * (policy.model.net.max_logvar.sum() - policy.model.net.min_logvar.sum())
            loss += policy.model.net.compute_decays()

            # TODO: move all training data to GPU before hand
            train_in1 = torch.from_numpy(train_in[batch_idxs]).to(TORCH_DEVICE).float()
            train_targ1 = torch.from_numpy(train_targs[batch_idxs]).to(TORCH_DEVICE).float()

            mean, logvar = policy.model.net(train_in1, ret_logvar=True)
            inv_var = torch.exp(-logvar)

            train_losses = ((mean - train_targ1) ** 2) * inv_var + logvar
            train_losses = train_losses.mean(-1).mean(-1).sum()
            # Only taking mean over the last 2 dimensions
            # The first dimension corresponds to each model in the ensemble

            loss += train_losses
            train_loss += train_losses.item()

            policy.model.optim.zero_grad()
            loss.backward()
            policy.model.optim.step()

        logger.add_scalar('Train_iter_offline/Training loss', train_loss/num_batch, i)
        print('Offline: step:', i, '\ttraining acc:', train_loss/num_batch)

        idxs = shuffle_rows(idxs)

        val_in = torch.from_numpy(train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
        val_targ = torch.from_numpy(train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

        mean, _ = policy.model.net(val_in)
        mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
        validate_loss += mse_losses.item()
        
        logger.add_scalar('Validation_iter_offline/Validation loss', validate_loss, i)
        print('Offline: step:', i, '\ttest acc:', validate_loss)

    return train_in, train_targs
    

def offline_train_MBRL_post(policy, epochs, obs_trajs, acs_trajs, train_in_all, train_targs_all, logger, index):
    """Trains the internal model of this controller. Once trained,
    this controller switches from applying random actions to using MPC.

    Arguments:
        obs_trajs: A list of observation matrices, observations in rows.
        acs_trajs: A list of action matrices, actions in rows.

    Returns: None.
    """
    # Construct new training points and add to training set
    #true action, normalized observations
    new_train_in, new_train_targs = [], []
    for obs, acs in zip(obs_trajs, acs_trajs):
        new_train_in.append(np.concatenate([policy.obs_preproc_3d(obs[:-1]), acs], axis=-1))
        new_train_targs.append(policy.targ_proc(obs[:-1], obs[1:]))
    train_in = np.concatenate([train_in_all] + new_train_in, axis=0)
    train_targs = np.concatenate([train_targs_all] + new_train_targs, axis=0)

    # Train the pytorch model
    policy.model.net.fit_input_stats(train_in)

    idxs = np.random.randint(train_in.shape[0], size=[policy.model.net.num_nets, train_in.shape[0]])

    epochs = epochs

    # TODO: double-check the batch_size for all env is the same
    batch_size = 100

    epoch_range = epochs
    num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

    for i in range(epoch_range):
        train_loss = 0 
        validate_loss = 0

        for batch_num in range(num_batch):
            batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

            loss = 0.01 * (policy.model.net.max_logvar.sum() - policy.model.net.min_logvar.sum())
            loss += policy.model.net.compute_decays()

            # TODO: move all training data to GPU before hand
            train_in1 = torch.from_numpy(train_in[batch_idxs]).to(TORCH_DEVICE).float()
            train_targ1 = torch.from_numpy(train_targs[batch_idxs]).to(TORCH_DEVICE).float()

            mean, logvar = policy.model.net(train_in1, ret_logvar=True)
            inv_var = torch.exp(-logvar)

            train_losses = ((mean - train_targ1) ** 2) * inv_var + logvar
            train_losses = train_losses.mean(-1).mean(-1).sum()
            # Only taking mean over the last 2 dimensions
            # The first dimension corresponds to each model in the ensemble

            loss += train_losses
            train_loss += train_losses.item()

            policy.model.optim.zero_grad()
            loss.backward()
            policy.model.optim.step()
        
        logger.add_scalar('Train_iter_online/Training loss', train_loss/num_batch, i+index*epoch_range)

        idxs = shuffle_rows(idxs)

        val_in = torch.from_numpy(train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
        val_targ = torch.from_numpy(train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

        mean, _ = policy.model.net(val_in)
        mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
        validate_loss += mse_losses.item()
        
        logger.add_scalar('Validation_iter_online/Validation loss', validate_loss, i+index*epoch_range)

        if (i+index*epoch_range)%10==0:
            policy.model.save_model(i+index*epoch_range)
        
    return train_in, train_targs

############# For emdedding NN

def train_meta(self, obs_trajs, acs_trajs, logger, num_i):
    self.db_train.add_roll_outs(obs_trajs,acs_trajs)

    self.has_been_trained = True
    
    if self.inter:
        train_epoch = 25
    else:
        train_epoch = max(int(0.8*self.db_train.running_samples_num/(self.meta_task_num*self.n_way*self.k_spt)),1)
    
    for step in range(self.meta_epoch_sum,self.meta_epoch_sum+train_epoch):

        x_spt, y_spt, x_qry, y_qry = self.db_train.meta_next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                        torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = self.model(x_spt, y_spt, x_qry, y_qry,step)

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)
            train_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
            logger.add_scalars('online train accuracy', train_data_dic, step)

        if step % 500 == 0:
            accs = []
            for _ in range(1000//self.meta_task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = self.db_train.meta_next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(TORCH_DEVICE), torch.from_numpy(y_spt).to(TORCH_DEVICE), \
                                                torch.from_numpy(x_qry).to(TORCH_DEVICE), torch.from_numpy(y_qry).to(TORCH_DEVICE)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = self.model.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            test_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
            logger.add_scalars('online test accuracy', test_data_dic, step)

    self.meta_epoch_sum += train_epoch

def embedding_meta_train(model, cfg, logger, path):
    # Generate data for meta training
    dataset = generate_task_data()
    tasks_in, tasks_out = dataset.generate_data()

    # Meta train the model + embedding, and save it
    famle.train_meta(model, cfg, tasks_in, tasks_out, logger, path)
    # self.model.save("/home/wawa/catkin_meta/src/MBRL_transport/FAMLE_model/model.pt")

def train_model(model, train_in, train_out, task_id):
    cloned_model = copy.deepcopy(model)
    famle.train(cloned_model,
                train_in,
                train_out,
                task_id,
                inner_iter=20,
                inner_lr=1e-4,
                minibatch=32)
    return cloned_model

def variation_inference_train(trainer, logger):
    trainer.run(logger)