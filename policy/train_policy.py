import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv

class PolicyTrainer:
    def __init__(self, env, reward_weights):
        self.env = env
        self.reward_weights = reward_weights
        self.R = self.env.Phi @ reward_weights
        self.policy = np.zeros(self.env.n_states, dtype=int)
        self.values = np.zeros(self.env.n_states)

    def train_value_iteration(self, gamma=0.9, theta=1e-9):
        """使用 Value Iteration 根据恢复的奖励训练策略"""
        P = self.env.P
        while True:
            delta = 0.0
            V_old = self.values.copy()
            for s in range(self.env.n_states):
                # 如果 s 是障碍物或终点，跳过
                if s in [self.env._pos_to_idx(o) for o in self.env.obstacles] or \
                   s == self.env._pos_to_idx(self.env.goal_pos):
                    continue
                    
                q_vals = np.zeros(self.env.n_actions)
                for a in range(self.env.n_actions):
                    q_vals[a] = np.sum(P[s, a] * (self.R + gamma * V_old))
                
                self.values[s] = np.max(q_vals)
                self.policy[s] = np.argmax(q_vals)
            
            delta = max(delta, np.max(np.abs(self.values - V_old)))
            if delta < theta:
                break
        return self.policy, self.values

    def evaluate(self, n_trials=100):
        """评估策略成功率"""
        successes = 0
        for _ in range(n_trials):
            s, _ = self.env.reset()
            done = False
            while not done:
                ns, r, term, trunc, _ = self.env.step(self.policy[s])
                done = term or trunc
                s = ns
            if self.env._idx_to_pos(s) == self.env.goal_pos:
                successes += 1
        return successes / n_trials