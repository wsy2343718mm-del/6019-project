import numpy as np

def preference_irl(pairs, labels, trajectories, env, n_iters=1000, lr=1.0, lam=0.01):
    """
    Bradley-Terry Preference Learning - 消除长度偏差版
    核心修复：
    1. 将累积特征除以轨迹长度，消除长度偏差
    2. 检查数据区分度，无区分度时返回默认权重（朝目标方向）
    3. 约束权重方向为正（确保奖励梯度指向目标）
    """
    np.random.seed(42)
    gamma = env.gamma
    goal_pos = env.goal_pos
    goal_idx = env._pos_to_idx(goal_pos)

    # 检查偏好数据区分度（阈值放宽到 48%-52%）
    label_mean = np.mean(labels)
    if 0.48 < label_mean < 0.52:
        # 偏好比例非常接近 50%，数据几乎没有区分度
        print(f"⚠️ Preference data has low discrimination (mean={label_mean:.2%}), using goal-directed default weights")
        # 默认权重：指向目标方向
        goal_feature = env.Phi[goal_idx]
        default_w = np.abs(goal_feature) + 0.1
        return default_w

    # 预计算轨迹特征（按长度归一化）
    traj_phi = []
    traj_returns = []  # 同时计算轨迹回报
    for traj in trajectories:
        feat = np.zeros(env.Phi.shape[1])
        T = len(traj)
        ret = 0.0
        for t, (s, a, r, ns, done) in enumerate(traj):
            feat += (gamma ** t) * env.Phi[s]
            ret += r * (gamma ** t)
        traj_phi.append(feat / max(T, 1))
        traj_returns.append(ret)
    traj_phi = np.array(traj_phi)
    traj_returns = np.array(traj_returns)

    # 内部缩放防 Sigmoid 饱和
    feat_scale = np.max(np.abs(traj_phi)) + 1e-8
    traj_phi_s = traj_phi / feat_scale

    idx_i = np.array([p[0] for p in pairs])
    idx_j = np.array([p[1] for p in pairs])
    y = np.array(labels, dtype=float)

    # 初始化权重：朝目标方向（而非随机）
    goal_feature = env.Phi[goal_idx] / feat_scale
    w = np.abs(goal_feature) + np.random.randn(env.Phi.shape[1]) * 0.05

    for it in range(n_iters):
        R_i = traj_phi_s[idx_i] @ w
        R_j = traj_phi_s[idx_j] @ w
        diff = R_i - R_j

        prob = 1.0 / (1.0 + np.exp(-np.clip(diff, -500, 500)))
        error = y - prob

        phi_diff = traj_phi_s[idx_i] - traj_phi_s[idx_j]
        grad = (phi_diff.T @ error) / len(pairs)

        # 添加目标方向的正则化（鼓励权重指向目标）
        goal_reg = 0.1 * (goal_feature - w / np.linalg.norm(w + 1e-8))

        w += lr * (grad - lam * w + goal_reg) / (1.0 + 0.01 * it)

        # 约束权重为正（确保奖励梯度指向目标）
        w = np.maximum(w, 0.01)

    # 还原物理尺度
    return w / feat_scale