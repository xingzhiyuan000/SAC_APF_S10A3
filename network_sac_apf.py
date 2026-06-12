import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ENV.mpParams import mpParams

params = mpParams()             # 通用参数

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,obs,act):
        x = torch.cat([obs, act], dim=-1)
        q = self.net(x)
        # q = torch.tanh(q)
        return q

class GaussianPolicy(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dim,
                 log_std_min,
                 log_std_max):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer=nn.Linear(hidden_dim, act_dim)
        self.log_std_layer=nn.Linear(hidden_dim,act_dim)

        self.log_std_min=log_std_min
        self.log_std_max=log_std_max

        # action_high=torch.as_tensor(action_space.high, dtype=torch.float32)
        # action_low=torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high=torch.as_tensor(params.max_action_3, dtype=torch.float32)
        action_low=torch.as_tensor(-params.max_action_3, dtype=torch.float32)

        self.register_buffer("action_scale", (action_high-action_low)/2)
        self.register_buffer("action_bias", (action_high+action_low)/2)


    def forward(self,obs):
        h=self.backbone(obs)
        mean=self.mean_layer(h)
        log_std=self.log_std_layer(h)
        log_std=torch.clamp(log_std,self.log_std_min,self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std=self.forward(obs)
        std = log_std.exp()

        normal=Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t*self.action_scale+self.action_bias

        log_prob=normal.log_prob(x_t)
        log_prob-=torch.log(self.action_scale * (1-y_t.pow(2))+1e-6)
        log_prob = log_prob.sum(dim=-1,keepdim=True)

        mean_action=torch.tanh(mean)*self.action_scale + self.action_bias

        return action, log_prob, mean_action

