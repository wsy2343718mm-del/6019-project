"""
Student 1 图表生成脚本
生成内容：
1. 环境布局可视化（含障碍物和终点标注）
2. 专家轨迹可视化（干净 vs 噪声）
3. 偏好数据分布（回报分布、偏好对散点图）
4. 转移矩阵稀疏性分析
5. RBF 特征可视化
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv

# 加载数据
with open(os.path.join("data", "expert_dataset.pkl"), "rb") as f:
    dataset = pickle.load(f)
cfg = dataset["env_config"]
obstacles = cfg.get("obstacles", [])

env = GridWorldEnv(
    size=cfg["size"], goal_pos=cfg["goal"], obstacles=obstacles,
    stochasticity=cfg["stochasticity"], feature_mode=cfg.get("feature_mode", "rbf"), gamma=cfg["gamma"]
)

os.makedirs("figures", exist_ok=True)

# ============================================================
# 图 S1-1: 环境布局（GridWorld 地图）
# ============================================================
def plot_env_layout():
    fig, ax = plt.subplots(figsize=(8, 8))

    size = env.size
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.invert_yaxis()

    # 障碍物
    for r, c in obstacles:
        rect = mpatches.Rectangle((c, r), 1, 1, facecolor="black", edgecolor="darkgray", linewidth=1)
        ax.add_patch(rect)
        ax.text(c + 0.5, r + 0.5, "X", color="white", fontsize=12, ha="center", va="center", fontweight="bold")

    # 终点
    gr, gc = env.goal_pos
    rect = mpatches.Rectangle((gc, gr), 1, 1, facecolor="gold", edgecolor="red", linewidth=3, linestyle="--")
    ax.add_patch(rect)
    ax.text(gc + 0.5, gr + 0.5, "G", color="red", fontsize=16, ha="center", va="center", fontweight="bold")

    # 起点
    rect = mpatches.Rectangle((0, 0), 1, 1, facecolor="lightblue", edgecolor="blue", linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.5, "S", color="blue", fontsize=14, ha="center", va="center", fontweight="bold")

    ax.set_title(f"GridWorld Environment ({size}×{size})\n{len(obstacles)} Obstacles, Goal at {env.goal_pos}", fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/s1_env_layout.png", dpi=300)
    print("✅ figures/s1_env_layout.png")
    plt.close()

# ============================================================
# 图 S1-2: 专家轨迹示例（干净 vs 噪声）- 热力图风格
# ============================================================
def plot_trajectories_comparison():
    clean_trajs = dataset["clean_trajectories"]
    noisy_trajs = dataset["noisy_trajectories"]
    size = env.size

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, trajs, title, cmap in [
        (axes[0], clean_trajs, "Clean Expert Trajectories (0% Noise)", "YlOrRd"),
        (axes[1], noisy_trajs, "Noisy Trajectories (15% Action Flip)", "YlGnBu"),
    ]:
        # 计算状态访问热力图
        visit_count = np.zeros((size, size))
        success_count = 0
        total_steps_list = []

        for traj in trajs:
            for s, a, r, ns, d in traj:
                row, col = env._idx_to_pos(s)
                visit_count[row, col] += 1
            total_steps_list.append(len(traj))
            # 检查是否到达目标
            last_s = traj[-1][3]  # next_state of last step
            if env._idx_to_pos(last_s) == env.goal_pos:
                success_count += 1

        # 创建掩码（障碍物）
        mask = np.zeros((size, size), dtype=bool)
        for r, c in obstacles:
            mask[r, c] = True

        im = ax.imshow(visit_count, cmap=cmap, origin="upper", aspect="equal")
        for r, c in obstacles:
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor="black", edgecolor="darkgray", linewidth=1.5))
            ax.text(c, r, "X", color="white", fontsize=10, ha="center", va="center", fontweight="bold")

        gr, gc = env.goal_pos
        ax.add_patch(mpatches.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                                        facecolor="none", edgecolor="red", linewidth=3, linestyle="--"))
        ax.text(gc, gr, "G", color="red", fontsize=16, ha="center", va="center", fontweight="bold")

        ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 1, 1,
                                        facecolor="lightblue", edgecolor="blue", linewidth=2))
        ax.text(0, 0, "S", color="blue", fontsize=12, ha="center", va="center", fontweight="bold")

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels(range(size))
        ax.set_yticklabels(range(size))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="State Visits")

        sr = success_count / len(trajs)
        avg_len = np.mean(total_steps_list)
        ax.set_title(f"{title}\n(n={len(trajs)}, Success={sr:.0%}, AvgLen={avg_len:.0f})", fontsize=12)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.suptitle("Expert Trajectory Heatmap: Clean vs Noisy", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s1_trajectories_comparison.png", dpi=300)
    print("✅ figures/s1_trajectories_comparison.png")
    plt.close()

# ============================================================
# 图 S1-2b: 干净专家轨迹 - 线条叠加图（参考 clean_trajectories.png 风格）
# ============================================================
def plot_clean_trajectories_lines():
    clean_trajs = dataset["clean_trajectories"]
    size = env.size

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.invert_yaxis()

    # 障碍物（浅色半透明）
    for r, c in obstacles:
        ax.add_patch(mpatches.Rectangle((c, r), 1, 1, facecolor="dimgray", alpha=0.6))
    # 终点
    gr, gc = env.goal_pos
    ax.add_patch(mpatches.Rectangle((gc, gr), 1, 1, facecolor="gold", edgecolor="red", linewidth=3, linestyle="--"))
    ax.text(gc + 0.5, gr + 0.5, "G", color="red", fontsize=18, ha="center", va="center", fontweight="bold")
    # 起点
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor="lightblue", edgecolor="blue", linewidth=2))
    ax.text(0.5, 0.5, "S", color="blue", fontsize=14, ha="center", va="center", fontweight="bold")

    # 绘制所有干净轨迹（每条不同颜色，半透明）
    n_show = min(len(clean_trajs), 200)
    cmap_lines = plt.cm.tab20
    for idx, traj in enumerate(clean_trajs[:n_show]):
        states = [env._idx_to_pos(s) for s, a, r, ns, d in traj]
        rows, cols = zip(*states)
        color = cmap_lines(idx % 20)
        ax.plot(cols, rows, color=color, alpha=0.25, linewidth=1.2)

    success_count = sum(1 for traj in clean_trajs
                       if env._idx_to_pos(traj[-1][3]) == env.goal_pos)
    avg_len = np.mean([len(t) for t in clean_trajs])

    ax.set_title(f"Clean Expert Trajectories (n={n_show} overlaid)\n"
                 f"Success={success_count/len(clean_trajs):.0%}, Avg Length={avg_len:.0f} steps",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)

    plt.tight_layout()
    plt.savefig("figures/s1_clean_trajectories_lines.png", dpi=300)
    print("✅ figures/s1_clean_trajectories_lines.png")
    plt.close()

# ============================================================
# 图 S1-2c: 干净专家轨迹 - 独立热力图（参考 clean_heatmap.png 风格）
# ============================================================
def plot_clean_trajectory_heatmap():
    clean_trajs = dataset["clean_trajectories"]
    size = env.size

    # 计算状态访问频率
    visit_count = np.zeros((size, size))
    for traj in clean_trajs:
        for s, a, r, ns, d in traj:
            row, col = env._idx_to_pos(s)
            visit_count[row, col] += 1

    fig, ax = plt.subplots(figsize=(9, 8))

    # 创建掩码
    mask = np.zeros((size, size), dtype=bool)
    for r, c in obstacles:
        mask[r, c] = True

    im = ax.imshow(visit_count, cmap="YlOrRd", origin="upper", aspect="equal")

    # 障碍物用黑色方块 + X 标注
    for r, c in obstacles:
        ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor="black", edgecolor="darkgray", linewidth=1.5))
        ax.text(c, r, "X", color="white", fontsize=10, ha="center", va="center", fontweight="bold")

    # 目标用红色虚线框 + G
    gr, gc = env.goal_pos
    ax.add_patch(mpatches.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                                    facecolor="none", edgecolor="red", linewidth=3, linestyle="--"))
    ax.text(gc, gr, "G", color="red", fontsize=18, ha="center", va="center", fontweight="bold")

    # 起点用蓝色方块 + S
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 1, 1,
                                    facecolor="lightblue", edgecolor="blue", linewidth=2))
    ax.text(0, 0, "S", color="blue", fontsize=14, ha="center", va="center", fontweight="bold")

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Visit Count (500 trajectories)", fontsize=11)

    success_count = sum(1 for traj in clean_trajs
                       if env._idx_to_pos(traj[-1][3]) == env.goal_pos)
    avg_len = np.mean([len(t) for t in clean_trajs])
    ax.set_title(f"Clean Expert Trajectory Heatmap (n=500)\n"
                 f"Success={success_count/len(clean_trajs):.0%}, Avg Length={avg_len:.0f} steps",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/s1_clean_trajectory_heatmap.png", dpi=300)
    print("✅ figures/s1_clean_trajectory_heatmap.png")
    plt.close()

# ============================================================
# 图 S1-3: 偏好数据分布
# ============================================================
def plot_preference_data():
    returns = dataset["trajectory_returns"]
    labels = dataset["preference_labels"]
    pairs = dataset["preference_pairs"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 轨迹回报分布
    axes[0].hist(returns, bins=50, color="teal", alpha=0.7, edgecolor="black")
    axes[0].axvline(np.mean(returns), color="red", linestyle="--", label=f"Mean={np.mean(returns):.2f}")
    axes[0].set_xlabel("Discounted Return", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"Trajectory Return Distribution\n(n={len(returns)})", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) 偏好对回报差异分布
    return_diffs = []
    for i, j in pairs:
        return_diffs.append(returns[i] - returns[j])
    return_diffs = np.array(return_diffs)
    axes[1].hist(return_diffs, bins=40, color="orange", alpha=0.7, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Return Difference (τ_i - τ_j)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"Preference Pair Return Differences\n(n={len(pairs)})", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # (c) 偏好标签分布
    label_counts = [sum(1 for l in labels if l == 1), sum(1 for l in labels if l == 0)]
    axes[2].bar(["τ_i preferred", "τ_j preferred"], label_counts,
                color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=1.5)
    axes[2].set_ylabel("Count", fontsize=11)
    axes[2].set_title(f"Preference Label Distribution\n(i preferred: {label_counts[0]/len(labels):.1%})", fontsize=12)
    axes[2].grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.tick_params(labelsize=10)

    plt.suptitle("Preference-Based Learning Data Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s1_preference_data.png", dpi=300)
    print("✅ figures/s1_preference_data.png")
    plt.close()

# ============================================================
# 图 S1-4: RBF 特征可视化
# ============================================================
def plot_rbf_features():
    Phi = env.Phi  # (n_states, 3)
    size = env.size

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["Goal Proximity (Gaussian)", "Obstacle Distance (Inverse)", "Bias (Constant)"]
    for idx, (ax, title) in enumerate(zip(axes, titles)):
        feat_grid = Phi[:, idx].reshape(size, size)
        im = ax.imshow(feat_grid, cmap="viridis", origin="upper")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 障碍物标记
        for r, c in obstacles:
            ax.add_patch(mpatches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                            facecolor="none", edgecolor="red", linewidth=2))
        # 终点标记
        gr, gc = env.goal_pos
        ax.plot(gc, gr, "r*", markersize=15)

    plt.suptitle("RBF Feature Maps (3-Dimensional)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s1_rbf_features.png", dpi=300)
    print("✅ figures/s1_rbf_features.png")
    plt.close()

# ============================================================
# 图 S1-5: 转移矩阵分析（清晰可视化版）
# ============================================================
def plot_transition_analysis():
    size = env.size

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ===== 左上：起点 (0,0) 处不同动作的转移概率 =====
    ax = axes[0, 0]
    s = 0  # 起点状态 (0,0)
    action_names = ["UP (0)", "DOWN (1)", "LEFT (2)", "RIGHT (3)"]

    x_pos = np.arange(4)
    intended_probs = np.zeros(4)
    noise_probs = np.zeros(4)
    for a in range(4):
        probs = env.P[s, a]
        intended_probs[a] = probs.max()
        noise_probs[a] = 1.0 - probs.max()

    bars1 = ax.bar(x_pos - 0.15, intended_probs, 0.3, color="#2ecc71", edgecolor="black", label="Intended", zorder=3)
    bars2 = ax.bar(x_pos + 0.15, noise_probs, 0.3, color="#e74c3c", edgecolor="black", label="Noise (other actions)", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(action_names, fontsize=10)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(f"Transition Breakdown at Start (0,0)\n(Stochasticity ε={env.stochasticity:.0%})", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9)

    # ===== 右上：从 (0,0) 向右走的实际可达状态（网格可视化） =====
    ax = axes[0, 1]
    a = 3  # 向右
    probs = env.P[s, a]
    prob_grid = np.zeros((size, size))
    for ns in range(env.n_states):
        if probs[ns] > 0.001:
            r, c = env._idx_to_pos(ns)
            prob_grid[r, c] = probs[ns]

    im = ax.imshow(prob_grid, cmap="YlOrRd", origin="upper", vmin=0)
    for r, c in obstacles:
        ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor="black", edgecolor="darkgray", linewidth=1))
    gr, gc = env.goal_pos
    ax.plot(gc, gr, "r*", markersize=12)
    ax.plot(0, 0, "bs", markersize=10, label="Start")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.set_title(f"Transition Distribution: RIGHT from (0,0)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    # ===== 左下：不同随机性水平的对比（柱状图） =====
    ax = axes[1, 0]
    stoch_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

    x_pos2 = np.arange(3)
    width = 0.15
    for idx, (stoch, color) in enumerate(zip(stoch_levels, colors)):
        env_tmp = GridWorldEnv(size=env.size, goal_pos=env.goal_pos, obstacles=obstacles,
                              stochasticity=stoch, feature_mode="rbf")
        P_tmp = env_tmp.P[0, 0]  # s=0, a=0 (UP)
        noise_per_action = []
        for a2 in [1, 2, 3]:
            noise_per_action.append(P_tmp[a2].max() if P_tmp[a2].max() > 0.001 else 0)

        ax.bar(x_pos2 - width + idx * width, [noise_per_action[0], noise_per_action[1], noise_per_action[2]],
               width, color=color, edgecolor="black", linewidth=0.5, label=f"ε={stoch:.1f}")

    ax.set_xticks(x_pos2)
    ax.set_xticklabels(["Max P(DOWN)", "Max P(LEFT)", "Max P(RIGHT)"], fontsize=9)
    ax.set_ylabel("Max Transition Prob to Other Direction", fontsize=10)
    ax.set_title("Noise Leakage: UP action → other directions", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ===== 右下：转移矩阵结构说明 =====
    ax = axes[1, 1]
    ax.axis("off")

    info_text = """
Transition Matrix P[s, a, s']

Grid: 10x10 | States: 100 | Actions: 4
Matrix Shape: (100, 4, 100)

Stochasticity Model:
  P(s'|s,a) = (1-ε) x intended action
            + ε/(|A|-1) x other 3 actions

Boundary Handling:
  x Wall bump  -> stay in place
  x Obstacle   -> stay in place

Special States:
  x Goal (9,9)  -> absorbing (P=1.0)
  x Obstacles   -> absorbing (P=1.0)

Default: ε = 0.10
  x Intended: 90%
  x Each noise action: 3.33%
"""
    ax.text(0.05, 0.95, info_text, fontsize=10, ha="left", va="top",
            transform=ax.transAxes, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9))
    ax.set_title("Transition Matrix Structure", fontsize=12, fontweight="bold")

    plt.suptitle("State Transition Dynamics Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s1_transition_analysis.png", dpi=300)
    print("✅ figures/s1_transition_analysis.png")
    plt.close()

# ============================================================
# 运行全部
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Student 1 图表生成")
    print("=" * 60)
    plot_env_layout()
    plot_trajectories_comparison()
    # plot_clean_trajectories_lines()
    # plot_clean_trajectory_heatmap()
    plot_preference_data()
    plot_rbf_features()
    plot_transition_analysis()
    print("\n✅ 所有 Student 1 图表生成完成！")
