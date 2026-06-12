import torch
import os
import time
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd


from ENV.mpParams import mpParams
import ENV.Tools as tools
import ENV.FK_LH4500 as fk

import config
from SAC_2019_APF_Action3_V3.agent_sac_apf_S10A3 import SACAgent
from env_LH4500_APF_Mill_S10A3 import ENV_APF
from SAC_2019_APF_Action3_V3.replay_buffer import ReplayBuffer


current_path=os.path.dirname(os.path.realpath(__file__))
model = current_path+"/models/"
image = current_path+"/images/"
paths = current_path + "/paths/"
data = current_path+"/data/"
timestamp=time.strftime("%Y%m%d%H%M%S")
params = mpParams() # 通用参数

# 参考关节序列
traj_deg = pd.read_excel(paths+'myMethod_关节变化趋势60带关节56优化-解析.xlsx', header=None).values
cols_to_rad = [1, 2, 4, 5, 6] # 保留第1列和第4列（索引0和3）不变
traj_rad = traj_deg.copy()
traj_rad[:, cols_to_rad] = np.deg2rad(traj_rad[:, cols_to_rad])
traj_norm = tools.normalize_q(traj_rad, params)

device=torch.device(config.DEVICE)  # 训练设备
Env = ENV_APF(params)               # 场景环境

PLOT_REWARD=True                #是否绘图
NUM_EPISODE = 500               #玩多少局
NUM_STEP = params.num_step      #每局最多步数

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = NUM_EPISODE*NUM_STEP * 0.6  # 探索衰减
best_reward = -1e10

STATE_DIM = 10              # 输入状态维度: 7个当前关节驱动量+位置误差+姿态误差+是否成功
ACTION_DIM = 3              # 输出动作维度: 3排斥力权重大小

num_candidates = 10


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

# q4_action_arr = np.array([-10, 0])
q4_action_arr = np.array([-10, 0])
q5_action_arr = np.array([-10, 10])
q6_action_arr = np.array([-10, 10])
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

    for step_i in range(NUM_STEP):

        # 【有目的搜索】-q5
        # q_c_rad = tools.denormalize_q(state[:7], params)        # 当前关节构型_rad
        # T08_current = fk.fkine_LH4500(q_c_rad, "08", True)
        # TCP_p = T08_current[:3, 3]  # 当前位置向量
        # TCP_a = T08_current[:3, 2]  # 当前姿态矩末端执行器的z轴方向
        # q5_explore_dir = np.dot(TCP_a, [0, 0, 1])
        #
        # if q5_explore_dir<0:
        #     q5_action_arr = np.array([0, 10])
        # else:
        #     q5_action_arr = np.array([-10, 0])
        #
        # # 【有目的搜索】-q6
        # # y=y0-(m/l)x0  (l,m,n)是TCP
        # # 1. 提取对应的分量
        # x0, y0 = TCP_p[0], TCP_p[1]
        # dx, dy = TCP_a[0], TCP_a[1]
        #
        # # 3. 计算平面直线方程的参数 y = kx + b
        # # 注意：加入防除零保护，防止机械臂刚好垂直于 X 轴运动 (dx=0)
        # if dx != 0:
        #     y_at_x_zero = y0 - (dy / dx) * x0 # 当 x = 0 时 y的值
        #
        # y_at_x_zero=y_at_x_zero-399.5  # 转换到磨机坐标系下
        # if y_at_x_zero>0:
        #     q6_action_arr = np.array([-10, 0])
        # else:
        #     q6_action_arr = np.array([0, 10])

        epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            best_explor_reward = -1e10  # 探索最大奖励
            candidates = []
            for explorer_i in range(num_candidates):
                # ε-贪心探索策略
                action = np.zeros(3)
                # 随机探索
                # action=np.random.uniform(low=-params.max_action_3, high=params.max_action_3, size=ACTION_DIM)

                action[0] = np.random.uniform(low=q4_action_arr[0],high=q4_action_arr[1])
                action[1] = np.random.uniform(low=q5_action_arr[0],high=q5_action_arr[1])
                action[2] = np.random.uniform(low=q6_action_arr[0],high=q6_action_arr[1])

                # 环境交互
                next_state, reward, done, info = Env.step(action, episode_i, step_i, traj_norm, False)
                # buffer.add(state, action, reward, next_state, float(done))

                # 找最优action;  q_ltr长远收益, 环境返回的是眼前收益reward
                q_ltr = agent.q1(torch.FloatTensor(state).to(device), torch.FloatTensor(action).to(device))
                q_ltr = q_ltr.detach().cpu().numpy()

                r_ = 0.6 * reward + 0.4 * q_ltr
                # r_ = q_ltr

                # r_= 0.7*(reward/20) + 0.3*(q_ltr/500)
                # r_= k_step*reward + (1-k_step)*q_ltr

                STEP_REWARD_BUFFER.append(reward)
                Q_REWARD_BUFFER.append(q_ltr)

                if r_ > best_explor_reward:
                    best_explor_reward=r_
                    best_action = action

                # candidates.append({
                #     "action": action.copy(),
                #     "reward": reward,
                #     "q_ltr": q_ltr,
                #     "r_": r_,
                #     "next_state": next_state,
                #     "done": done
                # })

            # 排序并指数衰减分布抽样
            # candidates.sort(key=lambda x: x["r_"], reverse=True)
            # N = len(candidates)
            # lam = 1  # 衰减系数,越大越偏向前面的元素 1 2
            # prob = np.exp(-lam * np.arange(N))
            # prob /= prob.sum()
            # idx = np.random.choice(np.arange(N), p=prob)
            # selected = candidates[idx]
            # best_action = selected["action"]

        else:
            best_action = agent.select_action(state, evaluate=False)


        # 最后真正更新环境状态
        next_state, reward, done, info = Env.step(best_action, episode_i, step_i, traj_norm, True)
        buffer.add(state, best_action, reward, next_state, float(done))

        # 放入经验池
        # buffer.add(state, action, reward, next_state, float(done))

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
            # ACTOR_LOSS.append(info["actor_loss"])
            # CRTIC_LOSS.append(info["q1 loss"])
            # ALPHA_LOSS.append(info["alpha_loss"])
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
        # f"r_progress:{progress_reward_sum:.2f}, "
        f"DTW奖惩/步:{dtw_reward_sum / (step_i + 1):.2f}, "
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
        torch.save(agent.actor.state_dict(), model + f"sac_apf_actor_A3_{timestamp}.pth")
        print(f"...saving best model reward:{round(best_reward, 2)}")
    print(f"--------Episode{episode_i+1}", 'reward %.2f' % episode_reward,'avg_step_best_reward %.2f' % best_reward, "--------")

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
    plt.savefig(image + f"Reward-ddpg-apf-A3-{timestamp}.png", format='png')

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
    plt.show()








