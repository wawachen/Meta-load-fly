import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from time import localtime, strftime
import os
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')


class Actor(nn.Module):
    def __init__(self, dim_in, dim_out, max_action):
        super(Actor,self).__init__()
        self.max_action = max_action

        self.layer_merge = nn.Linear(dim_in, 128)
        self.h_layer1 = nn.Linear(128,128)
        # self.h_layer2 = nn.Linear(256,256)
        # self.h_layer3 = nn.Linear(256,128)
        self.layer_output = nn.Linear(128,dim_out)
        self.log_std = nn.Parameter(torch.zeros(1, dim_out))

    def forward(self,s):
        out1 = F.relu(self.layer_merge(s))
        out2 = F.relu(self.h_layer1(out1))
        # out3 = F.relu(self.h_layer2(out2))
        # out4 = F.relu(self.h_layer3(out3))

        mean = self.max_action * torch.tanh(self.layer_output(out2))
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
        
        self.layer_merge = nn.Linear(dim_in, 128)

        self.h_layer1 = nn.Linear(128,128)
        # self.h_layer2 = nn.Linear(256,256)
        # self.h_layer3 = nn.Linear(128,128)
        self.layer_output = nn.Linear(128,1)

    def forward(self,s):
        out1 = F.relu(self.layer_merge(s))
        out2 = F.relu(self.h_layer1(out1))
        # out3 = F.relu(self.h_layer2(out2))
        # out4 = F.relu(self.h_layer3(out3))

        return self.layer_output(out2)


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
        self.model_path = "/home/wawa/catkin_meta/src/MBRL_transport/PPO_model"
        self.logger = logger
        self.optimize_step = 0

        self.actor = Actor(args.state_dim,args.action_dim,args.max_action)
        self.critic = Critic(args.state_dim)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        # if self.evaluate_s:
        #     self.initialise_networks()

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s.cuda()).detach().cpu().numpy().flatten()
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
        
        checkpoint = torch.load(self.model_path + '/' +'params.pkl') # load the torch data

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

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            # print(s)
            s = state_norm(s[0], update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _, _= env.step(action)
            # print(s_)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode

    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    log_path_ppo = os.path.join("/home/wawa/catkin_meta/src/MBRL_transport/log_PPO_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
    os.makedirs(log_path_ppo, exist_ok=True)
    logger_ppo = SummaryWriter(log_dir=log_path_ppo) # used for tensorboard

    replay_buffer = ReplayBuffer(args)
    agent = PPO_model(args,logger_ppo).to(TORCH_DEVICE)
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s[0])
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        # print("wawa")
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(torch.from_numpy(s.reshape(1,-1)).float().cuda())  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ ,_= env.step(action)
            # if done == 1:
            #     print("wawa")
            # s_ = env.step(action)
            # print(done)
            print("Episode{0}:{1}".format(total_steps,r))

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(torch.from_numpy(s).float().cuda(), a, a_logprob, r, torch.from_numpy(s_).float().cuda(), dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                logger_ppo.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                # if evaluate_num % args.save_freq == 0:
                #     np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    env_index = 3
    main(args, env_name=env_name[env_index], number=1, seed=10)