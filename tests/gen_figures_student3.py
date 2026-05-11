"""
Student 3 图表生成脚本
生成内容：
1. MaxEnt-IRL 收敛曲线（梯度范数、奖励权重演化）
2. 状态访问频率分布（专家 vs MaxEnt 诱导）
3. Bradley-Terry 偏好学习曲线
4. 软值迭代 vs 硬值迭代对比
5. MaxEnt/Pref 奖励场与真实奖励对比
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation
from irl.maxent_irl import maxent_irl
from irl.preference_irl import preference_irl

# 加载数据
with open(os.path.join("data", "expert_dataset.pkl"), "rb") as f:
    dataset = pickle.load(f)
cfg = dataset["env_config"]
obstacles = cfg.get("obstacles", [])

env = GridWorldEnv(
    size=cfg["size"], goal_pos=cfg["goal"], obstacles=obstacles,
    stochasticity=cfg["stochasticity"], feature_mode="rbf", gamma=cfg["gamma"]
)

os.makedirs("figures", exist_ok=True)

mu_E = compute_feature_expectation(dataset["clean_trajectories"], env)

# ============================================================
# 图 S3-1: MaxEnt-IRL 收敛分析
# ============================================================
def plot_maxent_convergence():
    # 手动运行 MaxEnt-IRL 并记录中间过程
    w = np.zeros(env.Phi.shape[1])
    gamma = env.gamma
    P = env.P
    Phi = env.Phi
    n_states = env.n_states
    n_actions = env.n_actions

    D0 = np.zeros(n_states)
    valid = np.ones(n_states)
    valid[env._pos_to_idx(env.goal_pos)] = 0.0
    for o in env.obstacles:
        valid[env._pos_to_idx(o)] = 0.0
    D0 = valid / valid.sum()

    grad_norms = []
    w_history = []
    reward_at_goal = []

    n_iters = 300
    lr = 0.2

    for it in range(n_iters):
        R = Phi @ w

        # Soft VI
        V = np.zeros(n_states)
        for _ in range(50):
            Q = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                Q[:, a] = R + gamma * P[:, a] @ V
            Q_max = np.max(Q, axis=1, keepdims=True)
            V = np.log(np.sum(np.exp(Q - Q_max), axis=1)) + Q_max.flatten()

        # Soft Policy
        Q_max = np.max(Q, axis=1, keepdims=True)
        pi = np.exp(Q - Q_max) / np.sum(np.exp(Q - Q_max), axis=1, keepdims=True)

        P_pi = np.einsum('ij,ijk->ik', pi, P)

        try:
            D = D0 @ np.linalg.inv(np.eye(n_states) - gamma * P_pi)
        except np.linalg.LinAlgError:
            D = D0.copy()

        mu_current = D @ Phi
        grad = mu_E - mu_current
        grad = np.clip(grad, -0.05, 0.05)

        grad_norms.append(np.linalg.norm(grad))
        w_history.append(w.copy())
        reward_at_goal.append(R[env._pos_to_idx(env.goal_pos)])

        w += lr * grad / (1.0 + 0.05 * it)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 梯度范数
    axes[0].plot(grad_norms, color="#3498db", linewidth=2)
    axes[0].set_xlabel("Iteration", fontsize=11)
    axes[0].set_ylabel("Gradient Norm (clipped)", fontsize=11)
    axes[0].set_title("MaxEnt-IRL: Gradient Norm over Iterations", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.001, color="red", linestyle="--", alpha=0.5, label="Convergence threshold")
    axes[0].legend()

    # (b) 权重演化
    w_history = np.array(w_history)
    for i in range(w_history.shape[1]):
        axes[1].plot(w_history[:, i], linewidth=1.5, label=f"w{i}")
    axes[1].set_xlabel("Iteration", fontsize=11)
    axes[1].set_ylabel("Weight Value", fontsize=11)
    axes[1].set_title("MaxEnt-IRL: Weight Evolution", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # (c) 目标状态奖励
    axes[2].plot(reward_at_goal, color="#e74c3c", linewidth=2)
    axes[2].set_xlabel("Iteration", fontsize=11)
    axes[2].set_ylabel("Reward at Goal State", fontsize=11)
    axes[2].set_title("MaxEnt-IRL: Goal Reward over Iterations", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("MaxEnt-IRL Convergence Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s3_maxent_convergence.png", dpi=300)
    print("✅ figures/s3_maxent_convergence.png")
    plt.close()

# ============================================================
# 图 S3-2: 状态访问频率对比
# ============================================================
def plot_state_visitation():
    # 运行 MaxEnt-IRL 获取最终策略
    w_maxent = maxent_irl(mu_E, env, n_iters=300, lr=0.2)
    R = env.Phi @ w_maxent

    gamma = env.gamma
    P = env.P
    n_states = env.n_states
    n_actions = env.n_actions

    D0 = np.zeros(n_states)
    valid = np.ones(n_states)
    valid[env._pos_to_idx(env.goal_pos)] = 0.0
    for o in env.obstacles:
        valid[env._pos_to_idx(o)] = 0.0
    D0 = valid / valid.sum()

    # Soft VI
    V = np.zeros(n_states)
    for _ in range(50):
        Q = np.zeros((n_states, n_actions))
        for a in range(n_actions):
            Q[:, a] = R + gamma * P[:, a] @ V
        Q_max = np.max(Q, axis=1, keepdims=True)
        V = np.log(np.sum(np.exp(Q - Q_max), axis=1)) + Q_max.flatten()

    Q_max = np.max(Q, axis=1, keepdims=True)
    pi_soft = np.exp(Q - Q_max) / np.sum(np.exp(Q - Q_max), axis=1, keepdims=True)
    P_pi = np.einsum('ij,ijk->ik', pi_soft, P)
    D_maxent = D0 @ np.linalg.inv(np.eye(n_states) - gamma * P_pi)

    # 专家经验访问频率
    D_expert = np.zeros(n_states)
    for traj in dataset["clean_trajectories"]:
        for s, a, r, ns, d in traj:
            D_expert[s] += 1
    D_expert /= D_expert.sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    size = env.size

    for ax, D, title, cmap in [
        (axes[0], D_expert, "Expert Empirical Visitation\n(500 Trajectories)", "Blues"),
        (ax := axes[1], D_maxent, "MaxEnt-Induced Visitation\n(Analytical Solution)", "Oranges"),
        (ax := axes[2], D_maxent - D_expert / D_expert.sum() * D_maxent.sum(), "Difference (MaxEnt - Expert)", "RdBu_r")
    ]:
        D_grid = D.reshape(size, size)
        mask = np.zeros((size, size), dtype=bool)
        for r, c in obstacles:
            mask[r, c] = True

        im = ax.imshow(D_grid, cmap=cmap, origin="upper")
        for r, c in obstacles:
            ax.add_patch(mpatches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                            facecolor="none", edgecolor="black", linewidth=2))
        gr, gc = env.goal_pos
        ax.plot(gc, gr, "r*", markersize=15)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("State Visitation Frequency: Expert vs MaxEnt-IRL", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s3_visitation_comparison.png", dpi=300)
    print("✅ figures/s3_visitation_comparison.png")
    plt.close()

# ============================================================
# 图 S3-3: Preference-BT 学习曲线
# ============================================================
def plot_preference_learning():
    # 手动运行 Preference-IRL 并记录过程
    pairs = dataset["preference_pairs"]
    labels = dataset["preference_labels"]
    all_trajs = dataset["clean_trajectories"] + dataset["noisy_trajectories"]

    np.random.seed(42)
    gamma = env.gamma
    goal_idx = env._pos_to_idx(env.goal_pos)

    # 预计算轨迹特征
    traj_phi = []
    for traj in all_trajs:
        feat = np.zeros(env.Phi.shape[1])
        T = len(traj)
        for t, (s, a, r, ns, done) in enumerate(traj):
            feat += (gamma ** t) * env.Phi[s]
        traj_phi.append(feat / max(T, 1))
    traj_phi = np.array(traj_phi)

    feat_scale = np.max(np.abs(traj_phi)) + 1e-8
    traj_phi_s = traj_phi / feat_scale

    idx_i = np.array([p[0] for p in pairs])
    idx_j = np.array([p[1] for p in pairs])
    y = np.array(labels, dtype=float)

    goal_feature = np.abs(env.Phi[goal_idx]) + 0.1
    w = goal_feature / feat_scale + np.random.randn(env.Phi.shape[1]) * 0.05
    w = np.maximum(w, 0.01)

    n_iters = 1000
    lr = 1.0
    lam = 0.01

    accuracy_history = []
    loss_history = []
    w_history = []

    for it in range(n_iters):
        R_i = traj_phi_s[idx_i] @ w
        R_j = traj_phi_s[idx_j] @ w
        diff = R_i - R_j
        prob = 1.0 / (1.0 + np.exp(-np.clip(diff, -500, 500)))

        # 准确率
        preds = (prob > 0.5).astype(float)
        acc = np.mean(preds == y)
        accuracy_history.append(acc)

        # 交叉熵损失
        eps = 1e-15
        loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
        loss_history.append(loss)

        w_history.append(w.copy())

        error = y - prob
        phi_diff = traj_phi_s[idx_i] - traj_phi_s[idx_j]
        grad = (phi_diff.T @ error) / len(pairs)
        goal_reg = 0.1 * (goal_feature / feat_scale - w / np.linalg.norm(w + 1e-8))
        w += lr * (grad - lam * w + goal_reg) / (1.0 + 0.01 * it)
        w = np.maximum(w, 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 准确率
    axes[0].plot(accuracy_history, color="#2ecc71", linewidth=1.5)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline (50%)")
    axes[0].set_xlabel("Iteration", fontsize=11)
    axes[0].set_ylabel("Preference Prediction Accuracy", fontsize=11)
    axes[0].set_title(f"Bradley-Terry: Learning Curve\nFinal Accuracy = {accuracy_history[-1]:.1%}", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.4, 1.05)

    # (b) 损失
    axes[1].plot(loss_history, color="#e74c3c", linewidth=1.5)
    axes[1].set_xlabel("Iteration", fontsize=11)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[1].set_title(f"Bradley-Terry: Loss over Iterations\nFinal Loss = {loss_history[-1]:.3f}", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # (c) 权重演化
    w_history = np.array(w_history)
    for i in range(w_history.shape[1]):
        axes[2].plot(w_history[:, i], linewidth=1.5, label=f"w{i}")
    axes[2].set_xlabel("Iteration", fontsize=11)
    axes[2].set_ylabel("Weight Value", fontsize=11)
    axes[2].set_title("Preference-BT: Weight Evolution", fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Preference-Based Learning (Bradley-Terry Model) Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s3_preference_learning.png", dpi=300)
    print("✅ figures/s3_preference_learning.png")
    plt.close()

# ============================================================
# 图 S3-4: 软值迭代 vs 硬值迭代
# ============================================================
def plot_soft_vs_hard_vi():
    # 硬值迭代（标准 VI）
    R = env.Phi @ maxent_irl(mu_E, env, n_iters=300, lr=0.2)
    policy_hard, V_hard = env.compute_optimal_policy(R)

    # 软值迭代（MaxEnt）
    gamma = env.gamma
    P = env.P
    n_states = env.n_states
    n_actions = env.n_actions

    V_soft = np.zeros(n_states)
    for _ in range(50):
        Q = np.zeros((n_states, n_actions))
        for a in range(n_actions):
            Q[:, a] = R + gamma * P[:, a] @ V_soft
        Q_max = np.max(Q, axis=1, keepdims=True)
        V_soft = np.log(np.sum(np.exp(Q - Q_max), axis=1)) + Q_max.flatten()

    Q_max = np.max(Q, axis=1, keepdims=True)
    pi_soft = np.exp(Q - Q_max) / np.sum(np.exp(Q - Q_max), axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    size = env.size

    # (a) 硬值函数
    V_hard_grid = V_hard.reshape(size, size)
    mask = np.zeros((size, size), dtype=bool)
    for r, c in obstacles:
        mask[r, c] = True
    im1 = axes[0].imshow(V_hard_grid, cmap="viridis", origin="upper")
    for r, c in obstacles:
        axes[0].add_patch(mpatches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                             facecolor="none", edgecolor="black", linewidth=2))
    gr, gc = env.goal_pos
    axes[0].plot(gc, gr, "r*", markersize=15)
    axes[0].set_title("Hard Value Iteration\n(Deterministic Policy)", fontsize=11)
    axes[0].set_xticks(range(size))
    axes[0].set_yticks(range(size))
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # (b) 软值函数
    V_soft_grid = V_soft.reshape(size, size)
    im2 = axes[1].imshow(V_soft_grid, cmap="viridis", origin="upper")
    for r, c in obstacles:
        axes[1].add_patch(mpatches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                             facecolor="none", edgecolor="black", linewidth=2))
    axes[1].plot(gc, gr, "r*", markersize=15)
    axes[1].set_title("Soft Value Iteration\n(MaxEnt, Stochastic Policy)", fontsize=11)
    axes[1].set_xticks(range(size))
    axes[1].set_yticks(range(size))
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # (c) 策略熵对比
    # 硬策略熵（确定性，熵为 0）
    H_hard = 0.0
    # 软策略熵
    H_soft = -np.sum(pi_soft * np.log(pi_soft + 1e-10), axis=1)
    H_soft_mean = np.mean(H_soft[~mask.flatten()])

    # 绘制软策略熵热力图
    H_grid = H_soft.reshape(size, size)
    im3 = axes[2].imshow(H_grid, cmap="plasma", origin="upper")
    for r, c in obstacles:
        axes[2].add_patch(mpatches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                             facecolor="none", edgecolor="black", linewidth=2))
    axes[2].plot(gc, gr, "r*", markersize=15)
    axes[2].set_title(f"Soft Policy Entropy Map\n(Mean H = {H_soft_mean:.3f})", fontsize=11)
    axes[2].set_xticks(range(size))
    axes[2].set_yticks(range(size))
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Hard VI vs Soft VI (MaxEnt): Value Functions and Policy Entropy", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s3_soft_vs_hard_vi.png", dpi=300)
    print("✅ figures/s3_soft_vs_hard_vi.png")
    plt.close()

# ============================================================
# 图 S3-5: MaxEnt/Pref 奖励场对比
# ============================================================
def plot_maxent_pref_rewards():
    # 加载已有结果或重新计算
    w_maxent = maxent_irl(mu_E, env, n_iters=300, lr=0.2)
    all_trajs = dataset["clean_trajectories"] + dataset["noisy_trajectories"]
    w_pref = preference_irl(
        dataset["preference_pairs"], dataset["preference_labels"],
        all_trajs, env, n_iters=1000, lr=1.0
    )

    R_maxent = env.Phi @ w_maxent
    R_pref = env.Phi @ w_pref
    R_gt = env.ground_truth_reward

    size = env.size
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    rewards = [R_gt, R_maxent, R_pref, R_maxent - R_pref]
    titles = ["Ground Truth", "MaxEnt-IRL Recovered", "Preference-BT Recovered", "MaxEnt - Pref (Difference)"]
    cmaps = ["RdYlGn", "viridis", "viridis", "RdBu_r"]

    for ax, R, title, cmap in zip(axes, rewards, titles, cmaps):
        R_grid = R.reshape(size, size)
        mask = np.zeros((size, size), dtype=bool)
        for r, c in obstacles:
            mask[r, c] = True

        im = ax.imshow(R_grid, cmap=cmap, origin="upper")
        for r, c in obstacles:
            ax.add_patch(mpatches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                            facecolor="none", edgecolor="black", linewidth=2))
        gr, gc = env.goal_pos
        ax.plot(gc, gr, "r*", markersize=15)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("MaxEnt-IRL vs Preference-BT: Recovered Reward Fields", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s3_maxent_pref_rewards.png", dpi=300)
    print("✅ figures/s3_maxent_pref_rewards.png")
    plt.close()

# ============================================================
# 运行全部
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Student 3 图表生成")
    print("=" * 60)
    plot_maxent_convergence()
    plot_state_visitation()
    plot_preference_learning()
    plot_soft_vs_hard_vi()
    plot_maxent_pref_rewards()
    print("\n✅ 所有 Student 3 图表生成完成！")
