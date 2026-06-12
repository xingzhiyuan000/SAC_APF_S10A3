import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from ENV.mpParams import mpParams
from ENV.Collision_LH4500 import is_Collision
from ENV.Collision_Self_LH4500 import is_Self_Collision
import ENV.FK_LH4500 as fk
import ENV.Rewards as rewards
import ENV.Tools as tools



class ENV_APF:
    def __init__(self, params):
        self.params = params
        self.current_state = None
        self.target_state = None
        self.target_pos = None                # 目标位置
        self.target_ori = None                # 目标姿态
        self.target_ori_pos=None              # 目标姿态和位置
        self.prev_distance = None
        self.T08_target = None
        self.collision_count = None
        self.update_step_rad = None           # 固定更新步长
        self.d_max = None
        self.d_min = None
        self.R_min = None
        self.actual_traj = None           # 实际关节序列

    # ===== reset =====
    def reset(self):
        self.collision_count = 0              # 记录单局碰撞次数
        self.actual_traj = []

        # 起始
        q_start = self.params.q_start
        T08_start = fk.fkine_LH4500(q_start, "08", True)
        ori_start = T08_start[:3, :3]  # 姿态矩阵

        # 目标
        q_target = self.params.q_target
        self.T08_target = fk.fkine_LH4500(q_target, "08", True)
        self.target_pos=self.T08_target[:3, 3]
        self.target_ori=self.T08_target[:3, :3]
        self.prev_distance = np.linalg.norm(self.target_pos-T08_start[:3, 3])

        self.d_max = np.linalg.norm(self.target_pos-T08_start[:3, 3])
        self.d_min = self.d_max

        state_start_norm = tools.normalize_q(q_start, self.params)
        state_target_norm = tools.normalize_q(q_target, self.params)
        # 第8个状态：TCP到目标距离
        state_start_norm = np.append(state_start_norm, 1)
        # 状态9：姿态误差
        self.R_min=np.sum((np.sum(self.target_ori * ori_start, axis=0) - 1) ** 2)
        state_start_norm = np.append(state_start_norm, self.R_min)
        # 状态10：是否达到目标
        state_start_norm = np.append(state_start_norm, 0)

        self.current_state = state_start_norm
        self.target_state = state_target_norm

        # 更新步长
        # self.update_step_rad= self.params.update_step_rad   #固定更新步长
        self.update_step_rad= abs((q_target-q_start)/50)      #根据目标调整 50

        return state_start_norm

    # ===== step =====
    def step(self, a, eposide_i, step_i, ref_traj=None, update_state=True):

        k_rep_q4 = a[0]   # q4更新权重
        k_rep_q5 = a[1]   # q5更新权重
        k_rep_q6 = a[2]   # q6更新权重

        q_c_norm = self.current_state[:7]                           # 当前关节
        q_c_rad = tools.denormalize_q(q_c_norm, self.params)        # 当前关节_rad
        self.actual_traj.append(q_c_norm)                           # 记录实际轨迹
        # 目标关节（归一化）
        q_t_norm = self.target_state

        # 引力计算
        F_att = q_t_norm - q_c_norm                             # 引力方向
        F_att_norm = F_att / (np.linalg.norm(F_att) + 1e-8)     # 引力单位方向

        # 末端与目标距离计算
        T08_current = fk.fkine_LH4500(q_c_rad, "08", True)
        pos_current = T08_current[:3, 3]    # 当前位置向量
        ori_current = T08_current[:3, :3]   # 当前姿态矩阵
        d_c = np.linalg.norm(self.target_pos - pos_current)    # 当前末端距目标距离

        # 排斥力调节权重
        k_step = tools.step_weight(d_c, self.d_max, w_min=0.001, w_max=1) #依据末端位置误差
        # 线性插值控制排斥强度 knew=s⋅k+(1−s)⋅1
        k_rep_q4 = k_step * k_rep_q4 + (1 - k_step)
        k_rep_q5 = k_step * k_rep_q5 + (1 - k_step)
        k_rep_q6 = k_step * k_rep_q6 + (1 - k_step)

        # 合力方向计算
        F_total_norm = F_att_norm    # 合力单位方向

        # 更新关节
        update_step_rad_current=self.update_step_rad.copy()
        update_step_rad_current[3] *= k_rep_q4  # q4更新步长更新
        update_step_rad_current[4] *= k_rep_q5  # q5更新步长更新
        update_step_rad_current[5] *= k_rep_q6  # q6更新步长更新

        # update_step_rad_current[3] = update_step_rad_current[3]+ k_step*update_step_rad_current[3]*k_rep_q4  # q4更新步长更新
        # update_step_rad_current[4] = update_step_rad_current[4]+ k_step*update_step_rad_current[4]*k_rep_q5  # q5更新步长更新
        # update_step_rad_current[5] = update_step_rad_current[5]+ k_step*update_step_rad_current[5]*k_rep_q6  # q6更新步长更新

        q_next_rad = q_c_rad + update_step_rad_current * F_total_norm
        # 关节极限限制
        q_next_rad = np.clip(q_next_rad, self.params.qmin, self.params.qmax)
        q_next_norm = tools.normalize_q(q_next_rad, self.params)

        # 状态空间更新
        next_state = q_next_norm

        # 关节更新后末端距目标距离
        T08_next = fk.fkine_LH4500(q_next_rad, "08", True)
        pos_next = T08_next[:3, 3]   # 下时刻位置向量
        ori_next = T08_next[:3, :3]  # 下时刻姿态矩阵
        d_n = np.linalg.norm(self.target_pos-pos_next)

        # 8. rewards奖励计算
        # 8.1进展情况
        # r_progress = rewards.rewardProgress(self.prev_q2_distance, next_q2_distance, self.params)
        # 8.2末端路径长度
        # r_endDis = -np.linalg.norm(pos_current-pos_next)
        # 8.3关节距离
        # r_jointDis = rewards.rewardJointDis(q_next_norm, self.target_state)
        # 8.4是否碰撞
        # r_collision, collision_done, collision_dist, self.F_rep_norm = rewards.rewardCollision_Mill(q_next_rad, self.params)
        r_collision, collision_done = rewards.rewardCollision_Mill(q_next_rad, self.params)
        # 8.5【是否成功】
        r_success, success_done = rewards.rewardSuccess(self.target_pos, pos_next, self.target_ori, ori_current,
                                                        self.params, 3)  # 位置和姿态
        # 8.6能耗（关节累积变化量）
        r_energy = rewards.rewardEnergyCost(q_c_norm, q_next_norm, self.params)
        # 8.7偷懒惩罚:防止不动
        # r_still, still_done = rewards.rewardStill(q_c_norm, q_next_norm)
        # 8.8方向一致性惩罚:防止来回振荡
        # r_alignment = rewards.rewardAlignment(q_next_norm, q_c_norm, q_t_norm)
        # 8.9步数惩罚
        r_step, step_done = rewards.rewardStep(step_i, self.params)
        # 8.10位置误差奖励
        r_tcp2target, pos_err = rewards.rewardTCP2Target(pos_next, self.target_pos)   # 指数形式
        # 8.11姿态误差奖励
        r_orientation, ori_err = rewards.rewardOrientation(ori_next, self.target_ori)  # 指数形式


        collision_free = (self.collision_count == 0)
        success = success_done and collision_free
        r_success = r_success if success else 0

        # 如果碰撞发生则判断不成功
        if collision_done:
            self.collision_count += 1

        # 状态8更新
        next_state = np.append(next_state, d_n / self.d_max)
        # 状态9更新
        next_state = np.append(next_state, np.sum((np.sum(self.target_ori * ori_next, axis=0) - 1) ** 2))
        # 状态10更新
        if success_done and collision_free:
            next_state = np.append(next_state, 1)
        else:
            next_state = np.append(next_state, 0)

        # 保存状态到全局
        if update_state:
            self.current_state = next_state
            self.prev_distance = d_n

        # 是否单局结束判断
        done = success_done or step_done            # 达到目标位姿或达到最大步数

        # 各奖励权重
        w_r_progress = 1             #
        w_r_tcp2target = 1           #位置奖惩权重
        w_r_orientation = 1          #
        w_r_collision_inner = 1e-4   #碰撞奖惩权重-内部
        w_r_collision_outer = 0.01   #碰撞奖惩权重-外部
        w_r_success = 1              #
        w_r_energy = 1               #
        w_r_step = 1                 #
        w_r_dtw = 10                 #DTW时间动态规划权重

        # DTW回合结束奖励
        # if done:
        # if ref_traj is not None and step_i%2==0:
        #     actual_seq = np.array(self.actual_traj)  # 形状为 (N, 7)
        #
        #     # 使用fastdtw计算整体宏观相似度
        #     dtw_dist, _ = fastdtw(actual_seq[:,3].reshape(-1, 1), ref_traj[:,3].reshape(-1, 1), dist=euclidean)
        #     # dtw_dist, _ = fastdtw(ref_traj, ref_traj, dist=euclidean)
        #     res_r_dtw = w_r_dtw * np.exp(-0.01 * dtw_dist)  # 使用负指数将距离转化为 (0, 100] 的大奖励
        # else:
        #     res_r_dtw = 0

        res_r_dtw = 0

        if collision_done:
            res_r_collision = w_r_collision_outer * r_collision
        else:
            res_r_collision = w_r_collision_inner * r_collision

        # res_r_progress = w_r_progress * r_progress
        res_r_tcp2target= w_r_tcp2target * r_tcp2target
        res_r_orientation=w_r_orientation*r_orientation
        res_r_success=w_r_success*r_success
        res_r_energy=w_r_energy*r_energy
        res_r_step=w_r_step*r_step

        # 单步总奖励
        total_reward = res_r_tcp2target+\
                       res_r_orientation+\
                       res_r_collision + \
                       res_r_success + \
                       res_r_energy + \
                       res_r_step+ \
                       res_r_dtw
        # 附件信息
        info = {
            "k_step": k_step,
            "success": success_done and collision_free,
            "reach": success_done,
            "res_rep_q4": k_rep_q4,
            "res_rep_q5": k_rep_q5,
            "res_rep_q6": k_rep_q6,
            "collision_done": collision_done,
            "reward_tcp2target": res_r_tcp2target,
            "reward_orientation": res_r_orientation,
            "reward_collision": res_r_collision,
            "reward_success": res_r_success,
            "reward_energy": res_r_energy,
            "reward_step": res_r_step,
            "reward_dtw": res_r_dtw,
            # "reward_progress": res_r_progress,
            # "reward_endDis": r_endDis,
            # "reward_jointDis": r_jointDis,
            # "reward_still": r_still,
            # "reward_alignment": r_alignment,
        }

        # print(
        #     f"@第{eposide_i + 1}局,"
        #     f"第{step_i + 1}步,"
        #     f"@更新权重q4:{k_rep_q4:.1f}, "
        #     f"@更新权重q5:{k_rep_q5:.1f}, "
        #     f"@更新权重q6:{k_rep_q6:.1f}, "
        #     f"@位置误差:{pos_err:.2f}, "
        #     f"@位置奖惩:{res_r_tcp2target:.2f}, "
        #     f"@姿态误差:{ori_err:.2f}, "
        #     f"#姿态奖惩:{res_r_orientation:.2f}, "
        #     f"@碰撞奖惩:{res_r_collision:.2f}, "
        #     f"@是否碰撞:{collision_done}, "
        #     f"@成功奖惩:{r_success:.2f}, "
        #     f"@能量损耗:{res_r_energy:.2f}, "
        #     f"@懒惰奖惩:{res_r_step:.2f}, "
        #     f"@是否成功:{success_done}, "
        # )

        return next_state, total_reward, done, info
