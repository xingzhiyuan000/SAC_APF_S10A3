import torch
import torch.nn as nn
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf=np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf=np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf=np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf=np.zeros((size, 1), dtype=np.float32)
        self.done_buf=np.zeros((size, 1), dtype=np.float32)

        self.max_size=size
        self.ptr=0
        self.size=0
        self.device=device

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr]=obs
        self.next_obs_buf[self.ptr]=next_obs
        self.act_buf[self.ptr]=act
        self.rew_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done

        self.ptr= (self.ptr+1)% self.max_size
        self.size=min(self.size+1, self.max_size)

    def sample(self, batch_size):
        idx=np.random.randint(0, self.size, size=batch_size)
        batch=dict(
            obs=torch.tensor(self.obs_buf[idx], device=self.device),
            act=torch.tensor(self.act_buf[idx], device=self.device),
            rew=torch.tensor(self.rew_buf[idx], device=self.device),
            next_obs=torch.tensor(self.next_obs_buf[idx], device=self.device),
            done=torch.tensor(self.done_buf[idx], device=self.device)
        )

        return batch