import torch
import os
import time
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import sys


sys.path.append("..") # 把上一级目录添加到 Python 的模块搜索路径中

from ENV.mpParams import mpParams
import ENV.Tools as tools
import ENV.FK_LH4500 as fk

import config
from agent_sac_apf_S10A3 import SACAgent
from env_LH4500_APF_Mill_S10A3 import ENV_APF
from replay_buffer import ReplayBuffer


current_path=os.path.dirname(os.path.realpath(__file__))
model = current_path+"/models/"
image = current_path+"/images/"
data = current_path+"/data/"
timestamp=time.strftime("%Y%m%d%H%M%S")
params = mpParams() # 通用参数


device=torch.device(config.DEVICE)  # 训练设备
Env = ENV_APF(params)               # 场景环境

PLOT_REWARD=True                    #是否绘图
NUM_EPISODE = params.num_episode    #玩多少局 500
NUM_STEP = params.num_step          #每局最多步数

EPSILON_START = params.EPSILON_START
EPSILON_END = params.EPSILON_END
EPSILON_DECAY = NUM_EPISODE*NUM_STEP * params.EPSILON_DECAY_RATE  # 探索衰减
best_reward = -1e10

STATE_DIM = 10              # 输入状态维度: 7个当前关节驱动量+位置误差+姿态误差+是否成功
ACTION_DIM = 3              # 输出动作维度: 3排斥力权重大小

NUM_EXPLORE = 10            # 每步探索次数


REWARD_BUFFER = []          # 每局奖励数组方便绘图
REP_MEAN_BUFFER_Q4 = []     # 排斥力1均值-q4
REP_STD_BUFFER_Q4 = []      # 排斥力1方差-q4
REP_MEAN_BUFFER_Q5 = []     # 排斥力2均值-q5
REP_STD_BUFFER_Q5 = []      # 排斥力2方差-q5
REP_MEAN_BUFFER_Q6 = []     # 排斥力3均值-q6
REP_STD_BUFFER_Q6 = []      # 排斥力3方差-q6

ACTOR_LOSS=[]   # actor网络损失
CRTIC_LOSS=[]   # critic网络损失
ALPHA_LOSS=[]   # 温度系数损失
ALPHA_BUFFER=[] # 温度系数

STEP_REWARD_BUFFER=[]   # 每一步及时奖励
Q_REWARD_BUFFER=[]      # 每一步长远奖励

SUCCESS_BUFFER = np.empty(shape=NUM_EPISODE)

# 实例化Agent
agent=SACAgent(
    obs_dim=STATE_DIM,
    act_dim=ACTION_DIM,
    hidden_dim=config.HIDDEN_DIM,
    device=device,
    log_std_min=config.LOG_STD_MIN,
    log_std_max=config.LOG_STD_MAX,
    gamma=config.GAMMA,
    tau=config.TAU,
    alpha_init=0.1,
    actor_lr=config.ACTOR_LR,
    critic_lr=config.CRITIC_LR,
    alpha_lr=config.ALPHA_LR
)

#经验池
buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, config.BUFFER_SIZE, device)

best_eposide_reward = -1e10  # 单局最大奖励
best_avg_step_reward = -1e10  # 平均每步奖励


success_count = 0  # 成功次数
for episode_i in range(NUM_EPISODE):

    REP_BUFFER_Q4 = []                  # q4排斥力权重
    REP_BUFFER_Q5 = []                  # q5排斥力权重
    REP_BUFFER_Q6 = []                  # q5排斥力权重

    state = Env.reset()                 # 初始化环境
    episode_reward = 0                  # 每局的回报
    progress_reward_sum = 0.0
    tcp2target_reward_sum = 0.0
    orientation_reward_sum = 0.0
    distance_reward_sum = 0.0
    collision_reward_sum = 0.0
    success_reward_sum = 0.0
    energy_reward_sum = 0.0
    still_reward_sum = 0.0
    alignment_reward_sum = 0.0
    step_reward_sum = 0.0
    collision_count = 0                 # 碰撞次数
    success_flag = 0
    reach_flag = 0
    dtw_reward_sum = 0                  # DTW 动态时间规划
    actor_loss_sum = 0                  # actor loss
    critic_loss_sum = 0                 # critic loss
    alpha_loss_sum = 0                  # alpha loss
    alpha_sum = 0                       # alpha
    k_step=1

    # mu = (q456_action_arr[:,1]-q456_action_arr[:,1])/2;
    # sigma = np.full(ACTION_DIM,5.0); #初始大范围探索
    # alpha = 0.5; #均值更新步长
    # beta = 0.95; #方差收缩率（衰减系数）
    q456_action_arr = np.array([[-10, 10], [-10, 10], [-10, 10]])
    q4_action_arr = np.array([-10, 0])
    q5_action_arr = np.array([-10, 10])
    q6_action_arr = np.array([-10, 10])
    random_count = 0  # 随机动作次数
    for step_i in range(NUM_STEP):

        epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            random_count += 1
            best_explor_reward = -1e10  # 探索最大奖励
            for explorer_i in range(NUM_EXPLORE):
                # ε-贪心探索策略
                action = np.zeros(3)
                # 均匀随机探索
                # action=np.random.uniform(low=-params.max_action_3, high=params.max_action_3, size=ACTION_DIM)

                action[0] = np.random.uniform(low=q4_action_arr[0],high=q4_action_arr[1])
                action[1] = np.random.uniform(low=q5_action_arr[0],high=q5_action_arr[1])
                action[2] = np.random.uniform(low=q6_action_arr[0],high=q6_action_arr[1])

                # 高斯策略收缩探索
                # action = np.random.normal(loc=mu, scale=sigma)
                # np.clip(action, q456_action_arr[:,0], q456_action_arr[:,1])

                # 环境交互
                next_state, reward, done, info = Env.step(action, episode_i, step_i, False)
                buffer.add(state, action, reward, next_state, float(done))

                # 找最优action;  q_ltr长远收益, 环境返回的是眼前收益reward
                q_ltr = agent.q1(torch.FloatTensor(state).to(device), torch.FloatTensor(action).to(device))
                q_ltr = q_ltr.detach().cpu().numpy()

                r_ = 0.6 * reward + 0.4 * q_ltr
                # r_ = reward


                STEP_REWARD_BUFFER.append(reward)
                Q_REWARD_BUFFER.append(q_ltr)

                if r_ > best_explor_reward:
                    best_explor_reward=r_
                    best_action = action

                    # 探索范围更改
                    if action[0]>(q4_action_arr[1]+q4_action_arr[0])/2:
                        q4_action_arr[0]=action[0]-(q4_action_arr[1]-action[0])
                    else:
                        q4_action_arr[1] = action[0] + (action[0]-q4_action_arr[0])

                    if action[1] > (q5_action_arr[1] + q5_action_arr[0]) / 2:
                        q5_action_arr[0] = action[1] - (q5_action_arr[1] - action[1])
                    else:
                        q5_action_arr[1] = action[1] + (action[1] - q5_action_arr[0])

                    if action[2] > (q6_action_arr[1] + q6_action_arr[0]) / 2:
                        q6_action_arr[0] = action[2] - (q6_action_arr[1] - action[2])
                    else:
                        q6_action_arr[1] = action[2] + (action[2] - q6_action_arr[0])


                #     mu = mu + alpha * (action - mu);
                #     sigma *= beta;  # 方差变小，区间逐渐逼近点点
                # else:
                #     sigma = np.minimum(5.0, sigma * 1.02)

        else:
            best_action = agent.select_action(state, evaluate=False)


        # 最后真正更新环境状态
        next_state, reward, done, info = Env.step(best_action, episode_i, step_i, True)
        buffer.add(state, best_action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward     # 每局累积奖励
        k_step = info["k_step"]
        tcp2target_reward_sum += info["reward_tcp2target"]
        orientation_reward_sum += info["reward_orientation"]
        collision_reward_sum += info["reward_collision"]  # 碰撞奖励
        collision_count += info["collision_done"]  # 碰撞次数
        success_reward_sum += info["reward_success"]
        energy_reward_sum += info["reward_energy"]
        step_reward_sum += info["reward_step"]
        dtw_reward_sum += info["reward_dtw"]

        success_flag = max(success_flag, int(info["success"]))
        reach_flag = max(reach_flag, int(info["reach"]))
        REP_BUFFER_Q4.append(info["res_rep_q4"])  # 每局q4步长更新权重
        REP_BUFFER_Q5.append(info["res_rep_q5"])  # 每局q5步长更新权重
        REP_BUFFER_Q6.append(info["res_rep_q6"])  # 每局q6步长更新权重

        batch = buffer.sample(config.BATCH_SIZE)
        if buffer.size>=config.BATCH_SIZE:
            info = agent.update(batch)
            actor_loss_sum += info["actor_loss"]
            critic_loss_sum += info["q1 loss"]
            alpha_loss_sum += info["alpha_loss"]
            alpha_sum += info["alpha"]

        if done:
            break

    avg_step_reward=episode_reward/(step_i+1) # 平均每步奖励

    REWARD_BUFFER.append(episode_reward)
    SUCCESS_BUFFER[episode_i] = success_flag

    # 计算每局排斥力均值和方差
    mean_rep_q4 = np.mean(REP_BUFFER_Q4)
    std_rep_q4 = np.std(REP_BUFFER_Q4)
    mean_rep_q5 = np.mean(REP_BUFFER_Q5)
    std_rep_q5 = np.std(REP_BUFFER_Q5)
    mean_rep_q6 = np.mean(REP_BUFFER_Q6)
    std_rep_q6 = np.std(REP_BUFFER_Q6)

    # 保存到数组用于绘图
    REP_MEAN_BUFFER_Q4.append(mean_rep_q4)
    REP_STD_BUFFER_Q4.append(std_rep_q4)
    REP_MEAN_BUFFER_Q5.append(mean_rep_q5)
    REP_STD_BUFFER_Q5.append(std_rep_q5)
    REP_MEAN_BUFFER_Q6.append(mean_rep_q6)
    REP_STD_BUFFER_Q6.append(std_rep_q6)

    ACTOR_LOSS.append(actor_loss_sum)
    CRTIC_LOSS.append(critic_loss_sum)
    ALPHA_LOSS.append(alpha_loss_sum)
    ALPHA_BUFFER.append(alpha_sum)

    # 成功显示标识
    if success_flag:
        success_star = '*'
    else:
        success_star = ''

    print(
        f"{success_star}回合:{episode_i + 1}, 【回合奖励】:{episode_reward:.2f}, "
        # f"DTW奖惩/步:{dtw_reward_sum / (step_i + 1):.2f}, "
        f"位置奖惩/步:{tcp2target_reward_sum / (step_i + 1):.2f}, "
        f"姿态奖惩/步:{orientation_reward_sum / (step_i + 1):.2f}, "
        f"碰撞奖惩/步:{collision_reward_sum / (step_i + 1):.2f}, "
        f"【碰撞次数】:{collision_count}, "
        f"成功奖惩:{success_reward_sum:.2f}, "
        f"能量损耗/步:{energy_reward_sum / (step_i + 1):.2f}, "
        f"步数奖惩:{step_reward_sum:.2f}, "
        f"每局步数:{step_i + 1}, "
        f"是否达到:{reach_flag}, "
        f"是否成功:{success_flag}, "
        f"q4更新均值和标准差:[{mean_rep_q4:.2f}, {std_rep_q4:.2f}], "
        f"q5更新均值和标准差:[{mean_rep_q5:.2f}, {std_rep_q5:.2f}], "
        f"q6更新均值和标准差:[{mean_rep_q6:.2f}, {std_rep_q6:.2f}], "
        f"每局奖惩/步:{avg_step_reward:.2f}"
    )

    # 平均每步奖励最大
    if avg_step_reward > best_reward:
        best_reward = avg_step_reward

    if  success_flag and random_count==0:
        torch.save(agent.actor.state_dict(), model + f"sac_apf_actor_S10A1_{timestamp}.pth")
        print(f"...saving best model reward:{round(best_reward, 2)}")

    print(
        f"--------随机探索次数: {random_count},"
        f"训练成功率: {round(success_count/NUM_EPISODE, 2)},"
        f"平均每步最大奖励: {best_reward:.2f},--------")

    if collision_count == 0 and success_flag == 1:
        success_count += 1


print(f"训练成功率:{round(success_count/NUM_EPISODE, 2)}")

# 【回合奖励】导出为EXCEL
df = pd.DataFrame(REWARD_BUFFER)                # 转成 DataFrame
df.to_excel(data+f"Reward-sac-apf-S10A3-{timestamp}.xlsx", index=False, header=False)  # 导出 Excel

# 【阶段成功】导出为EXCEL
df_success = pd.DataFrame(SUCCESS_BUFFER)
df_success.to_excel(data+f"success-sac-apf-S10A3-{timestamp}.xlsx", index=False, header=False)

# 奖励曲线绘图并保存
if PLOT_REWARD:
    # ================= Total Reward =================
    plt.plot(np.arange(len(REWARD_BUFFER)), REWARD_BUFFER, color='purple', alpha=0.5, label='Reward')
    plt.plot(np.arange(len(REWARD_BUFFER)), gaussian_filter1d(REWARD_BUFFER, sigma=5), color='red', linewidth=2)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig(image + f"Reward-sac-apf-S10A3-{timestamp}.png", format='png')

    # ================= Loss 绘图 =================
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    # ===== 上图：CRTIC LOSS =====
    axs[0].plot(np.arange(len(CRTIC_LOSS)), CRTIC_LOSS, color='purple', alpha=0.5, label='critic_loss')
    axs[0].plot(np.arange(len(CRTIC_LOSS)), gaussian_filter1d(CRTIC_LOSS, sigma=5), color='red', linewidth=2)
    axs[0].set_ylabel("critic_loss")
    axs[0].set_title("critic_loss")
    axs[0].legend()
    axs[0].grid(True)
    # ===== 中图：ACTOR LOSS =====
    axs[1].plot(np.arange(len(CRTIC_LOSS)), CRTIC_LOSS, color='purple', alpha=0.5, label='actor_loss')
    axs[1].plot(np.arange(len(CRTIC_LOSS)), gaussian_filter1d(CRTIC_LOSS, sigma=5), color='red', linewidth=2)
    axs[1].set_ylabel("actor_loss")
    axs[1].set_title("actor_loss")
    axs[1].legend()
    axs[1].grid(True)
    # ===== 下图：ALPHA LOSS =====
    axs[2].plot(np.arange(len(ALPHA_LOSS)), ALPHA_LOSS, color='purple', alpha=0.5, label='alpha_loss')
    axs[2].plot(np.arange(len(ALPHA_LOSS)), gaussian_filter1d(ALPHA_LOSS, sigma=5), color='red', linewidth=2)
    axs[2].set_ylabel("alpha_loss")
    axs[2].set_title("alpha_loss")
    axs[2].legend()
    axs[2].grid(True)
    # ===== 下下图：ALPHA =====
    axs[3].plot(np.arange(len(ALPHA_BUFFER)), ALPHA_BUFFER, color='purple', alpha=0.5, label='alpha')
    axs[3].plot(np.arange(len(ALPHA_BUFFER)), gaussian_filter1d(ALPHA_BUFFER, sigma=5), color='red', linewidth=2)
    axs[3].set_ylabel("alpha")
    axs[3].set_title("alpha")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(image + f"loss-sac-apf-S10A3-{timestamp}.png", format='png')

    # ================= 奖励绘图 =================
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # ===== 上图：step_reward =====
    axs[0].plot(np.arange(len(STEP_REWARD_BUFFER)), STEP_REWARD_BUFFER, color='purple', alpha=0.5, label='step_reward')
    axs[0].plot(np.arange(len(STEP_REWARD_BUFFER)), gaussian_filter1d(STEP_REWARD_BUFFER, sigma=5), color='red', linewidth=2)
    axs[0].set_ylabel("step_reward")
    axs[0].set_title("step_reward")
    axs[0].legend()
    axs[0].grid(True)

    # ===== 下图：Q_reward =====
    axs[1].plot(np.arange(len(Q_REWARD_BUFFER)), Q_REWARD_BUFFER, color='purple', alpha=0.5, label='Q_reward')
    axs[1].plot(np.arange(len(Q_REWARD_BUFFER)), gaussian_filter1d(Q_REWARD_BUFFER, sigma=5), color='red', linewidth=2)
    axs[1].set_ylabel("Q_reward")
    axs[1].set_title("Q_reward")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(image + f"step-Q-sac-apf-S10A3-{timestamp}.png", format='png')
    plt.show()



