"""
Student 2 图表生成脚本（修正版）
修复：
1. s2_suboptimal_trajectories.png 坐标偏移 → 改用 pcolormesh，网格与 Rectangle 完美对齐
2. s2_lp_mm_rewards.png 拆分 → 分成 s2_reward_fields.png（3奖励场）和 s2_reward_differences.png（3差异图）
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
from irl.lp_irl import solve_lp_irl
from irl.mm_irl import solve_mm_irl

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
mu_subs = generate_suboptimal_mus(env)

n_feat = env.Phi.shape[1]
w_lp = solve_lp_irl(mu_E, mu_subs, n_feat)
w_mm, margin = solve_mm_irl(mu_E, mu_subs, n_feat)

R_gt = env.ground_truth_reward
R_lp = env.Phi @ w_lp
R_mm = env.Phi @ w_mm
SZ = env.size


def setup_grid(ax):
    """设置网格坐标（pcolormesh 兼容）"""
    ax.set_xlim(0, SZ)
    ax.set_ylim(SZ, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(SZ))
    ax.set_yticks(range(SZ))
    ax.set_xticklabels(range(SZ))
    ax.set_yticklabels(range(SZ))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(SZ + 1), minor=True)
    ax.set_yticks(np.arange(SZ + 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)


def add_decorations(ax):
    """添加障碍物和目标（与 pcolormesh 格子完美对齐）"""
    for r, c in obstacles:
        ax.add_patch(mpatches.Rectangle((c, r), 1, 1,
                                        facecolor="black", edgecolor="darkgray", linewidth=1))
        ax.text(c + 0.5, r + 0.5, "X", color="white", fontsize=8,
                ha="center", va="center", fontweight="bold")
    gr, gc = env.goal_pos
    ax.add_patch(mpatches.Rectangle((gc, gr), 1, 1,
                                    facecolor="none", edgecolor="red", linewidth=2.5, linestyle="--"))
    ax.text(gc + 0.5, gr + 0.5, "G", color="red", fontsize=14,
            ha="center", va="center", fontweight="bold")


def mesh_heatmap(ax, data, cmap="viridis"):
    """pcolormesh 热力图——格子角点在整数坐标，与 Rectangle((c,r),1,1) 完美对齐"""
    X, Y = np.meshgrid(np.arange(SZ + 1), np.arange(SZ + 1))
    return ax.pcolormesh(X, Y, data, cmap=cmap, shading="flat")


# ============================================================
# S2-1: 特征期望对比
# ============================================================
def plot_feature_expectations():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["Feature 1\n(Goal Proximity)", "Feature 2\n(Obstacle Distance)", "Feature 3\n(Bias)"]
    for i, label in enumerate(labels):
        axes[i].bar(["Expert", "Suboptimal"], [mu_E[i], mu_subs[i][i]],
                    color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=1.2)
        axes[i].set_ylabel(f"μ[{i}]", fontsize=11)
        axes[i].set_title(f"{label}\nExpert={mu_E[i]:.3f}, Sub={mu_subs[i][i]:.3f}", fontsize=11)
        axes[i].grid(True, alpha=0.3, axis="y")
    plt.suptitle("Feature Expectation: Expert vs Suboptimal Policies", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s2_feature_expectations.png", dpi=300)
    print("✅ figures/s2_feature_expectations.png")
    plt.close()


# ============================================================
# S2-2: LP-IRL 约束可视化
# ============================================================
def plot_lp_constraints():
    fig, ax = plt.subplots(figsize=(8, 6))
    w1 = np.linspace(0, 1, 100)
    w2 = np.linspace(0, 1, 100)
    W1, W2 = np.meshgrid(w1, w2)
    feasible = np.ones_like(W1, dtype=bool)
    for mu_sub in mu_subs:
        d = mu_E - mu_sub
        val = (d[0] - d[2]) * W1 + (d[1] - d[2]) * W2 + d[2]
        feasible &= (val >= -0.001)
    im = ax.imshow(feasible.T, extent=[0, 1, 0, 1], origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("w1 (Goal Feature Weight)", fontsize=12)
    ax.set_ylabel("w2 (Obstacle Feature Weight)", fontsize=12)
    ax.set_title("LP-IRL Feasible Region (w1-w2 projection)\nGreen=Feasible, Red=Infeasible", fontsize=12)
    plt.colorbar(im, ax=ax, label="Feasibility")
    ax.plot(w_lp[0], w_lp[1], "r*", markersize=15, label=f"LP Solution ({w_lp[0]:.2f}, {w_lp[1]:.2f})")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/s2_lp_constraints.png", dpi=300)
    print("✅ figures/s2_lp_constraints.png")
    plt.close()


# ============================================================
# S2-3: MM-IRL 边距分析
# ============================================================
def plot_mm_margin():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    margins = [w_mm @ (mu_E - ms) for ms in mu_subs]
    axes[0].bar(["Pure UP", "Pure LEFT", "Random"], margins,
                color=["#3498db", "#9b59b6", "#f39c12"], edgecolor="black", linewidth=1.2)
    axes[0].axhline(y=margin, color="red", linestyle="--", linewidth=2, label=f"Min Margin = {margin:.4f}")
    axes[0].set_ylabel("Margin", fontsize=11)
    axes[0].set_title(f"MM-IRL Margin Analysis (Max-Min = {margin:.4f})", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(["w1 (Goal)", "w2 (Obstacle)", "w3 (Bias)"], w_mm,
                color=["#2ecc71", "#e74c3c", "#3498db"], edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("Weight Value", fontsize=11)
    axes[1].set_title(f"MM-IRL Recovered Weights (||w||2={np.linalg.norm(w_mm):.3f})", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.suptitle("Maximum Margin IRL Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s2_mm_margin.png", dpi=300)
    print("✅ figures/s2_mm_margin.png")
    plt.close()


# ============================================================
# S2-4: 次优策略轨迹热力图（修正坐标偏移）
# ============================================================
def plot_suboptimal_trajectories():
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    sub_policies = [
        (np.zeros(env.n_states, dtype=int), "Suboptimal 1: Pure UP (Action 0)"),
        (np.full(env.n_states, 2, dtype=int), "Suboptimal 2: Pure LEFT (Action 2)"),
        (np.random.randint(0, 4, env.n_states), "Suboptimal 3: Random Policy"),
    ]
    cmaps = ["Blues", "Oranges", "Purples"]
    for ax, (policy, title), cmap in zip(axes, sub_policies, cmaps):
        visit = np.zeros((SZ, SZ))
        for ep in range(200):
            s, _ = env.reset(seed=42 + ep * 100)
            for _ in range(env.max_episode_steps):
                r, c = env._idx_to_pos(s)
                visit[r, c] += 1
                ns, _, term, trunc, _ = env.step(policy[s])
                s = ns
                if term or trunc:
                    break
        setup_grid(ax)
        im = mesh_heatmap(ax, visit, cmap)
        add_decorations(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{title}\n(n=200 rollouts)", fontsize=11)
    plt.suptitle("Suboptimal Policy State Visitation Heatmaps", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s2_suboptimal_trajectories.png", dpi=300)
    print("✅ figures/s2_suboptimal_trajectories.png")
    plt.close()


# ============================================================
# S2-5: 奖励场（独立图：GT + LP + MM）
# ============================================================
def plot_reward_fields():
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    data = [
        ("Ground Truth\n(Sparse Reward)", R_gt, "RdYlGn"),
        ("LP-IRL\n(Feature Matching)", R_lp, "viridis"),
        ("MM-IRL\n(Max-Margin)", R_mm, "viridis"),
    ]
    for ax, (title, R, cmap) in zip(axes, data):
        Rg = R.reshape(SZ, SZ)
        setup_grid(ax)
        im = mesh_heatmap(ax, Rg, cmap)
        add_decorations(ax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Reward Value", fontsize=9)
        ax.set_title(f"{title}\nRange: [{R.min():.2f}, {R.max():.2f}]", fontsize=12)
    plt.suptitle("LP-IRL & MM-IRL Recovered Reward Fields", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s2_reward_fields.png", dpi=300)
    print("✅ figures/s2_reward_fields.png")
    plt.close()


# ============================================================
# S2-6: 奖励场差异分析（独立图）
# ============================================================
def plot_reward_differences():
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    data = [
        ("LP-IRL vs Ground Truth\n(R_lp - R_gt)", R_lp - R_gt),
        ("MM-IRL vs Ground Truth\n(R_mm - R_gt)", R_mm - R_gt),
        ("LP-IRL vs MM-IRL\n(R_lp - R_mm)", R_lp - R_mm),
    ]
    for ax, (title, R) in zip(axes, data):
        Rg = R.reshape(SZ, SZ)
        setup_grid(ax)
        im = mesh_heatmap(ax, Rg, "RdBu_r")
        add_decorations(ax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Reward Difference", fontsize=9)
        ax.set_title(f"{title}\nRange: [{R.min():.2f}, {R.max():.2f}]", fontsize=12)
    plt.suptitle("Reward Field Difference Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s2_reward_differences.png", dpi=300)
    print("✅ figures/s2_reward_differences.png")
    plt.close()


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Student 2 图表生成")
    print("=" * 60)
    plot_feature_expectations()
    plot_lp_constraints()
    plot_mm_margin()
    plot_suboptimal_trajectories()
    plot_reward_fields()
    plot_reward_differences()
    print("\n✅ 所有 Student 2 图表生成完成！")
