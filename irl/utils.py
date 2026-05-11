import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv

def compute_feature_expectation(trajs, env):
    """
    计算折扣归一化特征期望 μ = (1-γ) * E[∑ γ^t φ(s_t)]
    乘以 (1-γ) 消除轨迹长度偏差，使 μ 表示“折扣状态分布”，IRL 约束恒成立
    """
    mu = np.zeros(env.Phi.shape[1])
    for traj in trajs:
        for t, (s, a, r, ns, done) in enumerate(traj):
            mu += (env.gamma ** t) * env.Phi[s]
    return mu * (1.0 - env.gamma)  # ✅ 关键修复

def generate_suboptimal_mus(env, seed=42):
    np.random.seed(seed)
    mu_subs = []
    n_rollouts = 300

    # 1. 纯向上
    trajs = []
    for _ in range(n_rollouts):
        traj = []
        s, _ = env.reset(seed=seed)
        for _ in range(env.max_episode_steps):
            ns, r, term, trunc, _ = env.step(0)
            traj.append((s, 0, r, ns, term or trunc))
            s = ns
            if term or trunc: break
        trajs.append(traj)
    mu_subs.append(compute_feature_expectation(trajs, env))

    # 2. 纯向左
    trajs = []
    for _ in range(n_rollouts):
        traj = []
        s, _ = env.reset(seed=seed)
        for _ in range(env.max_episode_steps):
            ns, r, term, trunc, _ = env.step(2)
            traj.append((s, 2, r, ns, term or trunc))
            s = ns
            if term or trunc: break
        trajs.append(traj)
    mu_subs.append(compute_feature_expectation(trajs, env))

    # 3. 随机策略
    trajs = []
    for _ in range(n_rollouts):
        traj = []
        s, _ = env.reset(seed=seed)
        for _ in range(env.max_episode_steps):
            a = np.random.randint(env.n_actions)
            ns, r, term, trunc, _ = env.step(a)
            traj.append((s, a, r, ns, term or trunc))
            s = ns
            if term or trunc: break
        trajs.append(traj)
    mu_subs.append(compute_feature_expectation(trajs, env))

    return mu_subs