import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Optional

class GridWorldEnv(gym.Env):
    """
    高级 GridWorld 环境，专为经典 IRL (LP/MM/MaxEnt) 设计
    暴露完整转移矩阵 P[s,a,s'] 和特征映射 Phi[s]
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        size: int = 6,
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        stochasticity: float = 0.1,
        max_episode_steps: int = 200,
        gamma: float = 0.9,
        feature_mode: str = "coords"  # "coords" 适合可视化平滑奖励, "onehot" 适合纯表格
    ):
        super().__init__()
        self.size = size
        self.goal_pos = goal_pos if goal_pos is not None else (size - 1, size - 1)
        self.obstacles = set(obstacles) if obstacles else set()
        self.stochasticity = stochasticity
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.feature_mode = feature_mode

        self.n_states = size * size
        self.n_actions = 4  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        # 核心：预计算转移矩阵 P[s, a, s']
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self._build_transition_matrix()

        # 核心：特征映射 Phi[s] (IRL 算法依赖)
        self.Phi = self._build_feature_matrix()

        # 内部状态
        self._current_state = None
        self._step_count = 0

    def _pos_to_idx(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def _idx_to_pos(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.size)

    def _build_transition_matrix(self):
        """构建带随机性的状态转移矩阵 P"""
        action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
        for s in range(self.n_states):
            r, c = self._idx_to_pos(s)
            # 终点/障碍物为吸收态
            if (r, c) in self.obstacles or (r, c) == self.goal_pos:
                self.P[s, :, s] = 1.0
                continue

            for a in range(self.n_actions):
                dr, dc = action_deltas[a]
                nr, nc = r + dr, c + dc
                # 边界反弹
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    nr, nc = r, c
                intended_ns = self._pos_to_idx((nr, nc))
                # 撞障碍物则留在原地
                if (nr, nc) in self.obstacles:
                    intended_ns = s

                # 随机性建模: (1-eps) 走预期方向, eps 均匀随机到其他3个动作
                self.P[s, a, intended_ns] = (1.0 - self.stochasticity)
                
                # 噪声：均匀分配到其他动作方向
                for other_a in range(self.n_actions):
                    if other_a == a:
                        continue
                    other_dr, other_dc = action_deltas[other_a]
                    other_nr, other_nc = r + other_dr, c + other_dc
                    # 边界反弹
                    if not (0 <= other_nr < self.size and 0 <= other_nc < self.size):
                        other_nr, other_nc = r, c
                    other_ns = self._pos_to_idx((other_nr, other_nc))
                    # 撞障碍物则留在原地
                    if (other_nr, other_nc) in self.obstacles:
                        other_ns = s
                    
                    self.P[s, a, other_ns] += self.stochasticity / (self.n_actions - 1)

        # 数值安全归一化
        row_sums = self.P.sum(axis=2, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.P /= row_sums

    def _build_feature_matrix(self) -> np.ndarray:
        """构建状态特征矩阵 Phi[s]"""
        if self.feature_mode == "onehot":
            return np.eye(self.n_states)
        elif self.feature_mode == "coords":
            # 归一化坐标特征，便于 LP/MM-IRL 恢复平滑奖励场
            features = np.zeros((self.n_states, 2))
            for s in range(self.n_states):
                r, c = self._idx_to_pos(s)
                features[s] = [r / max(1, self.size - 1), c / max(1, self.size - 1)]
            return features
        elif self.feature_mode == "rbf":
            # 径向基函数特征，能表示稀疏奖励结构
            features = np.zeros((self.n_states, 3))
            goal_r, goal_c = self.goal_pos
            for s in range(self.n_states):
                r, c = self._idx_to_pos(s)
                # 特征1: 到目标的距离高斯 (越近越大)
                dist_to_goal = np.sqrt((r - goal_r)**2 + (c - goal_c)**2)
                features[s, 0] = np.exp(-dist_to_goal**2 / (2 * 3.0**2))
                # 特征2: 到最近障碍物的距离 (越远越大)
                if self.obstacles:
                    dist_to_obs = min([np.sqrt((r - or_)**2 + (c - oc)**2) 
                                      for or_, oc in self.obstacles])
                    features[s, 1] = 1.0 - np.exp(-dist_to_obs**2 / (2 * 2.0**2))
                else:
                    features[s, 1] = 1.0
                # 特征3: 常数偏置
                features[s, 2] = 1.0
            return features
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    @property
    def ground_truth_reward(self) -> np.ndarray:
        """真实奖励向量 (用于生成专家数据 & 评估恢复质量)"""
        R = np.zeros(self.n_states)
        R[self._pos_to_idx(self.goal_pos)] = 1.0
        R[list(self._pos_to_idx(o) for o in self.obstacles)] = -1.0
        return R

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        # 支持 options 中的 fixed_start 参数
        fixed_start = options.get('fixed_start', None) if options else None
        if fixed_start is not None:
            # 固定起点模式：使用指定的起点
            self._current_state = self._pos_to_idx(fixed_start)
        else:
            # 随机起点模式 (避开障碍/终点)
            valid_starts = [s for s in range(self.n_states) 
                            if self._idx_to_pos(s) not in self.obstacles 
                            and self._idx_to_pos(s) != self.goal_pos]
            self._current_state = self.np_random.choice(valid_starts)
        return self._current_state, {}

    def step(self, action: int):
        # 采样下一步状态
        next_state_probs = self.P[self._current_state, action]
        next_state = self.np_random.choice(self.n_states, p=next_state_probs)

        r, c = self._idx_to_pos(next_state)
        terminated = (r, c) == self.goal_pos or (r, c) in self.obstacles
        truncated = self._step_count >= self.max_episode_steps
        reward = self.ground_truth_reward[next_state]

        self._step_count += 1
        self._current_state = next_state
        return next_state, reward, terminated, truncated, {}

    def compute_optimal_policy(self, reward_vec: Optional[np.ndarray] = None):
        """
        值迭代求解最优策略
        返回: policy (int array), values (float array)
        """
        R = reward_vec if reward_vec is not None else self.ground_truth_reward
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        theta = 1e-9

        while True:
            delta = 0.0
            V_old = V.copy()
            for s in range(self.n_states):
                q_vals = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_vals[a] = np.sum(self.P[s, a] * (R + self.gamma * V_old))
                V[s] = np.max(q_vals)
                policy[s] = np.argmax(q_vals)
            delta = max(delta, np.max(np.abs(V - V_old)))
            if delta < theta:
                break
        return policy, V