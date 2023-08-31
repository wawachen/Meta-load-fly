import torch, os
import numpy as np
import argparse
from meta import Meta
from windNShot import WindNShot
from time import localtime, strftime
from tensorboardX import SummaryWriter

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_meta_offline_evaluation_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
    os.makedirs(log_path, exist_ok=True)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    print(args)

    # log_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_meta_evaluation_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
    # os.makedirs(log_path, exist_ok=True)
    # logger = SummaryWriter(logdir=log_path) # used for tensorboard

    config = [
        ('linear', [512, args.model_in]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('relu', [True]),
        ('linear', [args.model_out, 512]),
        ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # db_train = WindNShot(50, args.task_num, 1001, args.n_way, args.k_spt, args.k_qry)
    db_train = WindNShot(50, args.task_num, 1001, args.n_way, args.k_spt, args.k_qry, integrated=True,sequential=True)

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry, step)

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)
            train_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
            logger.add_scalars('offline train accuracy', train_data_dic, step)

        if step % 500 == 0:
            accs = []
            accs1 = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

                    test_acc1 = maml.single_tunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs1.append(test_acc1)

            # [b, update_step+1]
            # can use accs to plot error bar here
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            test_data_dic = {"s%d"%i:accs[i] for i in range(accs.shape[0])}
            logger.add_scalars('offline test accuracy', test_data_dic, step)

            accs1 = np.array(accs1).mean(axis=0).astype(np.float16)
            print('single Test acc:', accs1)
            test_data_dic1 = {"s%d"%i:accs1[i] for i in range(accs1.shape[0])}
            logger.add_scalars('offline test single accuracy', test_data_dic1, step)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=15)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--model_in', type=int, help='input of neural network', default=7)
    argparser.add_argument('--model_out', type=int, help='output of neural network', default=4)
    argparser.add_argument('--load_model', type=int, help='output of neural network', default=0)

    args = argparser.parse_args()

    main(args)
