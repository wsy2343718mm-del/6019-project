import numpy as np

def maxent_irl(mu_E, env, n_iters=300, lr=0.2):
    """
    Maximum Entropy IRL (Ziebart et al., 2008) - 数值稳定版
    加入梯度裁剪与自适应学习率，防止奖励尺度爆炸
    """
    w = np.zeros(env.Phi.shape[1])
    gamma = env.gamma
    P = env.P
    Phi = env.Phi
    n_states = env.n_states
    n_actions = env.n_actions

    # 初始状态分布 D0
    D0 = np.zeros(n_states)
    valid = np.ones(n_states)
    valid[env._pos_to_idx(env.goal_pos)] = 0.0
    for o in env.obstacles:
        valid[env._pos_to_idx(o)] = 0.0
    D0 = valid / valid.sum()

    for it in range(n_iters):
        R = Phi @ w
        
        # 1. Soft VI
        V = np.zeros(n_states)
        for _ in range(50):
            Q = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                Q[:, a] = R + gamma * P[:, a] @ V
            Q_max = np.max(Q, axis=1, keepdims=True)
            V = np.log(np.sum(np.exp(Q - Q_max), axis=1)) + Q_max.flatten()

        # 2. Soft Policy
        Q_max = np.max(Q, axis=1, keepdims=True)
        pi = np.exp(Q - Q_max) / np.sum(np.exp(Q - Q_max), axis=1, keepdims=True)

        # 3. 策略诱导转移矩阵
        P_pi = np.einsum('ij,ijk->ik', pi, P)

        # 4. 精确求解折扣状态访问频率
        try:
            D = D0 @ np.linalg.inv(np.eye(n_states) - gamma * P_pi)
        except np.linalg.LinAlgError:
            D = D0.copy()

        # 5. 梯度更新 (加入裁剪防爆炸)
        mu_current = D @ Phi
        grad = mu_E - mu_current
        grad = np.clip(grad, -0.05, 0.05)  # ✅ 关键：限制单步更新幅度
        w += lr * grad / (1.0 + 0.05 * it)  # 学习率衰减

    return w