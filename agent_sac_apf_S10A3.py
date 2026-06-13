import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from network_sac_apf import GaussianPolicy, QNetwork
import config


#更新方法
@torch.no_grad()
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#软更新
@torch.no_grad()
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1 - tau)
        target_param.data.add_(tau * param.data)

class SACAgent:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dim,
                 device,
                 log_std_min,
                 log_std_max,
                 gamma,
                 tau,
                 alpha_init,
                 actor_lr,
                 critic_lr,
                 alpha_lr):
        super().__init__()

        self.device=device
        self.gamma=gamma
        self.tau=tau

        self.actor=GaussianPolicy(
            obs_dim,
            act_dim,
            hidden_dim,
            log_std_min,
            log_std_max,
        ).to(device)

        self.q1=QNetwork(obs_dim,act_dim,hidden_dim).to(device)
        self.q2=QNetwork(obs_dim,act_dim,hidden_dim).to(device)

        self.q1_target=QNetwork(obs_dim,act_dim,hidden_dim).to(device)
        self.q2_target=QNetwork(obs_dim,act_dim,hidden_dim).to(device)

        self.actor_optimizer=Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer=Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer=Adam(self.q2.parameters(), lr=critic_lr)

        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)

        #初始化温度系数
        self.log_alpha=torch.tensor(np.log(alpha_init),
                                    dtype=torch.float32,
                                    requires_grad=True,
                                    device=device)

        self.alpha_optimizer=Adam([self.log_alpha], lr=alpha_lr)

        self.target_entrogy=-float(act_dim)  # 目标熵


    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs, evaluate=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, mean_action= self.actor.sample(obs_t)

        if evaluate:
            return mean_action.cpu().numpy()[0]
        return action.cpu().numpy()[0]

    # Critic网络更新
    def update_critic(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_prob, next_mean_action = self.actor.sample(next_obs)
            q1_next_action=self.q1_target(next_obs,next_action)
            q2_next_action=self.q2_target(next_obs,next_action)
            min_q_next_target=torch.min(q1_next_action,q2_next_action)

            next_value = min_q_next_target - self.alpha.detach() * next_log_prob

        target_q = rew + (1-done) * self.gamma * next_value

        q1_pred=self.q1(obs, act)
        q2_pred=self.q2(obs, act)

        q1_loss=F.mse_loss(q1_pred,target_q)
        q2_loss=F.mse_loss(q2_pred,target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        return {"q1 loss": q1_loss.item(),
                "q2 loss": q2_loss.item()}

    # actor 和 温度系数更新
    def update_actor_and_alpha(self, batch):
        obs = batch["obs"]
        action, log_prob, mean_action = self.actor.sample(obs)
        q1_val=self.q1(obs, action)
        q2_val=self.q2(obs, action)
        min_q_val = torch.min(q1_val, q2_val)

        actor_loss = (self.alpha.detach()*log_prob-min_q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha update
        alpha_loss=-(self.log_alpha*(log_prob+self.target_entrogy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha_loss_val=alpha_loss.item()

        return {"actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.alpha.item()}

    # Target_Critic网络更新
    def update_targets(self):
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

    # 总更新
    def update(self, batch):

        critic_info=self.update_critic(batch)
        actor_alpha_info=self.update_actor_and_alpha(batch)
        self.update_targets()
        info ={}
        info.update(critic_info)
        info.update(actor_alpha_info)

        return info








