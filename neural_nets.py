import torch
from torch import nn as nn
from torch.nn import functional as F
from utils import swish, get_affine_params
import numpy as np
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
from time import localtime, strftime

TORCH_DEVICE = torch.device('cuda')


class PtModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, 200)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, 200, 200)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, 200, 200)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, 200, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros([1,in_features]), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros([1,in_features]), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)
        

    def compute_decays(self):

        lin0_decays = 0.0001 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)


# class load_model(nn.Module):

#     network=SequentialParams([
#                 LayerParams("linear", in_features=MODEL_IN, out_features=200), LayerParams('relu'),
#                 LayerParams("linear", in_features=200, out_features=200), LayerParams('relu'),
#                 LayerParams("linear", in_features=200, out_features=200), LayerParams('relu'),
#                 LayerParams("linear", in_features=200, out_features=NUM_NETS * MODEL_OUT),
#             ]),


class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            
            if name is 'linear':
                # we use torch.nn.functional.linear which the parameter is [out,in] y = xw'+b
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                #torch.nn.init.xavier_uniform(w) in learn to adapt paper
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                # we add zero bias here
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError


    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [size, nn_input]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        #we can import weights and bias for custom update. Otherwise use the default ones.
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


# adapt from https://github.com/BoyuanChen/visual-selfmodeling
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

class OccupancyMLPQueryModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_features=256):
        super(OccupancyMLPQueryModel, self).__init__()

        half_hidden_features = int(hidden_features / 2)
        # A 2d spatial query point: coordinate network
        self.layerq1 = SirenLayer(2, half_hidden_features, is_first=True)

        # layers for states network
        self.layers1 = SirenLayer(in_channels-2, half_hidden_features, is_first=True)
        self.layers2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers3 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers4 = SirenLayer(half_hidden_features, half_hidden_features)

        # Intermediate layers
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        return x

    def state_encoder(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        return x

    def forward(self, x):
        # data structure: [N, 2+]
        query_feat = self.query_encoder(x[:, :2])
        state_feat = self.state_encoder(x[:, 2:])
        x = torch.cat((query_feat, state_feat), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


# from metasdf
class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

def init_weights_normal(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)   

class MetaFC(MetaModule):
    '''A fully connected neural network that allows swapping out the weights, either via a hypernetwork
    or via MAML.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features),
            nn.ReLU(inplace=True)
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features),
                nn.ReLU(inplace=True)
            ))

        if outermost_linear:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features),
            ))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features),
                nn.ReLU(inplace=True)
            ))

        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, coords, params=None, **kwargs):
        '''Simple forward pass without computation of spatial gradients.'''
        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output

def l2_loss(prediction, gt):
    return ((prediction - gt)**2).mean()


#PPO corrective policy
#state contains robot state, offline trajecotory, online trajectory and 2d occupancy map generated by occupancy predictor whose resolution is 256x256
#we need a encoder based on CNN for grid map 80x80--> 60
class Conv_autoencoder(nn.Module):
    def __init__(self):
        super(Conv_autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=0),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # 32x(80-2*3)x(74) = 194688
            nn.Linear(175232,60),
            nn.ReLU()
            # nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(60,175232),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 74, 74)),
            nn.ConvTranspose2d(32, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self,x):
        en_x = self.encoder(x)
        de_x = self.decoder(en_x)
        return de_x

    def encoder_forward(self,x):
        return self.encoder(x)

    def save_encoder(self):
        return torch.save(self.encoder.state_dict(),'/home/wawa/catkin_meta/src/MBRL_transport/depth_images/encoder_model_weights.pth')

    def load_encoder(self):
        self.encoder.load_state_dict(torch.load('/home/wawa/catkin_meta/src/MBRL_transport/depth_images/encoder_model_weights.pth'))

    def save_decoder(self):
        return torch.save(self.decoder.state_dict(),'/home/wawa/catkin_meta/src/MBRL_transport/depth_images/decoder_model_weights.pth')

    def load_decoder(self):
        self.decoder.load_state_dict(torch.load('/home/wawa/catkin_meta/src/MBRL_transport/depth_images/decoder_model_weights.pth'))

    
class Actor(nn.Module):
    def __init__(self, dim_in, dim_out, max_action):
        super(Actor,self).__init__()
        self.max_action = max_action

        self.layer_merge = nn.Linear(dim_in, 256)
        self.h_layer1 = nn.Linear(256,256)
        self.h_layer2 = nn.Linear(256,256)
        # self.h_layer3 = nn.Linear(256,128)
        self.layer_output = nn.Linear(256,dim_out)
        self.log_std = nn.Parameter(torch.zeros(1, dim_out))

    def forward(self,s):
        out1 = F.relu(self.layer_merge(s))
        out2 = F.relu(self.h_layer1(out1))
        out3 = F.relu(self.h_layer2(out2))
        # out4 = F.relu(self.h_layer3(out3))

        mean = self.max_action * torch.tanh(self.layer_output(out3))
        return mean

    def get_dist(self,s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self,dim_in):
        super(Critic,self).__init__()
        
        self.layer_merge = nn.Linear(dim_in, 256)

        self.h_layer1 = nn.Linear(256,256)
        self.h_layer2 = nn.Linear(256,256)
        # self.h_layer3 = nn.Linear(128,128)
        self.layer_output = nn.Linear(256,1)

    def forward(self,s):
        out1 = F.relu(self.layer_merge(s))
        out2 = F.relu(self.h_layer1(out1))
        out3 = F.relu(self.h_layer2(out2))
        # out4 = F.relu(self.h_layer3(out3))

        return self.layer_output(out3)


class PPO_model(nn.Module):
    def __init__(self,args,logger):
        super(PPO_model,self).__init__()
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.model_path = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        self.model_load_path = "/home/wawa/catkin_meta/src/MBRL_transport/PPO_model"
        self.logger = logger
        self.optimize_step = 0
        self.evaluate_s = args.evaluate_s

        self.actor = Actor(args.state_dim,args.action_dim,args.max_action)
        self.critic = Critic(args.state_dim)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        if self.evaluate_s:
            self.initialise_networks()

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            a = self.actor(s).detach().cpu().numpy().flatten()
        return a

    def choose_action(self, s):
        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def save_network(self,train_step):
        num = str(train_step)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'PPO_actor_params' : self.actor.state_dict(),
                    'PPO_actor_optim_params' : self.optimizer_actor.state_dict(),
                    'PPO_critic_params' : self.critic.state_dict(),
                    'PPO_critic_optim_params' : self.optimizer_critic.state_dict()}

        torch.save(save_dict, self.model_path + '/' + num + '_params.pkl')

    def initialise_networks(self):
        
        checkpoint = torch.load(self.model_load_path + '/' +'params.pkl') # load the torch data

        self.actor.load_state_dict(checkpoint['PPO_actor_params'])    # actor parameters
        self.critic.load_state_dict(checkpoint['PPO_critic_params'])    # actor parameters
        self.optimizer_actor.load_state_dict(checkpoint['PPO_actor_optim_params']) # critic optimiser state
        self.optimizer_critic.load_state_dict(checkpoint['PPO_critic_optim_params']) # critic optimiser state
    
    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).cuda()
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                self.logger.add_scalar('actor loss', actor_loss.mean().item(), self.optimize_step)
                self.logger.add_scalar('entropy', dist_entropy.mean().item(), self.optimize_step)
                self.logger.add_scalar('KL', ratios.mean().item(), self.optimize_step)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.logger.add_scalar('critic loss', critic_loss.item(), self.optimize_step)

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                self.optimize_step+=1

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now




