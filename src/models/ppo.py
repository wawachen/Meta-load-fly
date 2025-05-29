import torch
from torch import nn as nn
from models.neural_nets import Actor, Critic
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn import functional as F

class PPO_model(nn.Module):
    def __init__(self,args,logger, path, is_eval):
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
        self.model_path = path
        self.model_load_path = args.load_model_path
        self.logger = logger
        self.optimize_step = 0
        self.evaluate_s = is_eval

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
        
        p1 = self.model_path + '/meta_ppo_models/'
        if not os.path.exists(p1):
            os.makedirs(p1)

        torch.save(save_dict, p1 + num + '_params.pkl')

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