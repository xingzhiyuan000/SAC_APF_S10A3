import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
sys.path.append("..") # 把上一级目录添加到 Python 的模块搜索路径中

from ENV.mpParams import mpParams
import ENV.Tools as tools
import ENV.Evaluation as eva

import config
from agent_sac_apf_S10A3 import SACAgent
from env_LH4500_APF_Mill_S10A3 import ENV_APF


device=torch.device(config.DEVICE)  # 训练设备
params = mpParams()             # 通用参数
Env = ENV_APF(params)           # 场景环境

STATE_DIM = 10                               # 输入状态维度: 7个当前关节驱动量+位置误差+姿态误差+是否成功
ACTION_DIM = 3                               # 输出动作维度: 吸引力和排斥力权重大小

# 载入训练好的Actor模型
current_path=os.path.dirname(os.path.realpath(__file__))
data_dir = current_path+"/data/"
data_path = os.path.join(data_dir, "Q_deg.xlsx")

model=current_path+"/models/"
actor_path= model + "sac_apf_actor_A3_20260609151626.pth"

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

actor = agent.actor.to(device)

actor.load_state_dict(torch.load(actor_path))       # 载入trained模型

# 测试阶段
NUM_EPISODE=1
NUM_STEP=params.num_step
STATE_BUFFER = np.zeros((NUM_STEP+1, STATE_DIM-3))    # State缓冲区
ACTION_BUFFER = np.zeros((NUM_STEP, ACTION_DIM))    # Action缓冲区

Done_index=0
for episode_i in range(NUM_EPISODE):
    REP_BUFFER_Q4 = []  # q4排斥力权重
    REP_BUFFER_Q5 = []  # q5排斥力权重
    REP_BUFFER_Q6 = []  # q5排斥力权重
    K_STEP_BUFFER = []  # 排斥衰减权重

    collision_count = 0 # 碰撞次数
    success_flag = 0    # 是否成功
    reach_flag = 0      # 是否完成

    state = Env.reset()
    STATE_BUFFER[0] = state[:7]
    episode_reward=0
    Done_index=1
    start_time=time.time() # 记录开始时间
    for step_i in range(NUM_STEP):
        # action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        action = agent.select_action(state, evaluate=False)

        # action的3个动作
        ACTION_BUFFER[step_i, 0] = action[0]
        ACTION_BUFFER[step_i, 1] = action[1]
        ACTION_BUFFER[step_i, 2] = action[2]

        # 环境交互
        next_state, reward, done, info = Env.step(action,1,step_i)
        # 状态更新
        state = next_state
        episode_reward += reward
        REP_BUFFER_Q4.append(info["res_rep_q4"])  # 每局q4步长更新权重
        REP_BUFFER_Q5.append(info["res_rep_q5"])  # 每局q5步长更新权重
        REP_BUFFER_Q6.append(info["res_rep_q6"])  # 每局q6步长更新权重

        K_STEP_BUFFER.append(info["k_step"])      # 每局排斥衰减权重
        collision_count += info["collision_done"]  # 碰撞次数
        success_flag = max(success_flag, int(info["success"]))
        reach_flag = max(reach_flag, int(info["reach"]))

        STATE_BUFFER[step_i+1] = state[:7]
        Done_index += 1
        if done:
            break

    Q_rad_normal = STATE_BUFFER[0:Done_index].copy()
    Q_deg = Q_rad_normal.copy()
    Q_rad = Q_rad_normal.copy()
    cols_rad = [1, 2, 4, 5, 6]  # 指定需要转换的列（第2,3,5,6,7列）
    cols_mm = [0, 3]
    for i in range (Done_index):
        q_rad=tools.denormalize_q(Q_rad_normal[i], params) # 逆归一化
        Q_rad[i]=q_rad
        Q_deg[i, cols_rad] = np.rad2deg(q_rad[cols_rad])  # 弧度 → 角度
        Q_deg[i, cols_mm]=q_rad[cols_mm]

    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time # 计算差值

    print(f"Episode:{episode_i+1},一局完成累计回报:{round(episode_reward, 2)}")
    pathLen = eva.path_len(Q_rad)
    print(f"Episode:{episode_i+1},一局完成路径长度:{round(pathLen, 2)}")
    jointEnergyCost=eva.jont_energy_cost(Q_rad, params)
    print(f"Episode:{episode_i+1},一局完成关节能耗:{round(jointEnergyCost, 2)}")
    print(f"Episode:{episode_i+1},一局完成碰撞次数:{collision_count}")
    print(f"Episode:{episode_i+1},一局完成步数:{step_i + 1}")
    print(f"Episode:{episode_i+1},一局完成是否达到:{reach_flag}")
    print(f"Episode:{episode_i+1},一局完成是否成功:{success_flag}")
    print(f"Episode:{episode_i+1},一局完成规划时间:{execution_time:.2f} 秒")


    # 路径导出为EXCEL
    df = pd.DataFrame(Q_deg)                # 转成 DataFrame
    df.to_excel(data_path, index=False, header=False)  # 导出 Excel

    x = np.arange(Q_deg.shape[0])                       # 横轴（step）
    # 2行1列，上下共享x轴
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ===== 上图：q4q5q6的action变化曲线 =====
    axs[0].plot(x[0:Done_index - 1], ACTION_BUFFER[0:Done_index - 1, 0], label='Weight_q4')
    axs[0].plot(x[0:Done_index - 1], ACTION_BUFFER[0:Done_index - 1, 1], label='Weight_q5')
    axs[0].plot(x[0:Done_index - 1], ACTION_BUFFER[0:Done_index - 1, 2], label='Weight_q6')
    axs[0].set_ylabel("Action Value")
    axs[0].set_title("Actions/Weights of q4q5q6")
    axs[0].legend()
    axs[0].grid(True)

    # ===== 中图：排斥衰减权重变化曲线 =====
    axs[1].plot(x[0:Done_index - 1], K_STEP_BUFFER[0:Done_index - 1], label='k_step')
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("k_step Value")
    axs[1].set_title("k_step curve")
    axs[1].legend()
    axs[1].grid(True)

    # ===== 下图：实际q4q5q6权重变化曲线 =====
    axs[2].plot(x[0:Done_index - 1], REP_BUFFER_Q4[0:Done_index - 1], label='Weight_q4')
    axs[2].plot(x[0:Done_index - 1], REP_BUFFER_Q5[0:Done_index - 1], label='Weight_q5')
    axs[2].plot(x[0:Done_index - 1], REP_BUFFER_Q6[0:Done_index - 1], label='Weight_q6')
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Real Weight Value")
    axs[2].set_title("Real Weights of q4q5q6")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()  # 自动调整子图间距

    # ===== 关节变化曲线图 =====
    plt.figure(figsize=(10, 8))
    # 图1：第1列 + 第2列
    plt.subplot(2, 1, 1)
    plt.plot(x, Q_deg[:, 0], label='Joint 1')
    plt.plot(x, Q_deg[:, 3], label='Joint 4')
    plt.title("Joint 1 & Joint 4")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 图2：第2,3,5,6,7列
    plt.subplot(2, 1, 2)
    for i in cols_rad:
        plt.plot(x, Q_deg[:, i], label=f'Joint {i + 1}')

    plt.title("Joint 2,3,5,6,7")
    plt.xlabel("Step")
    plt.ylabel("Degree")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


