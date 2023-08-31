import torch
import numpy as np

cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if cuda else 'cpu')

class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s.cpu().detach().numpy()
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_.cpu().detach().numpy()
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(TORCH_DEVICE)
        a = torch.tensor(self.a, dtype=torch.float).to(TORCH_DEVICE)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(TORCH_DEVICE)
        r = torch.tensor(self.r, dtype=torch.float).to(TORCH_DEVICE)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(TORCH_DEVICE)
        dw = torch.tensor(self.dw, dtype=torch.float).to(TORCH_DEVICE)
        done = torch.tensor(self.done, dtype=torch.float).to(TORCH_DEVICE)

        return s, a, a_logprob, r, s_, dw, done
