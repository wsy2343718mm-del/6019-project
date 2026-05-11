"""
Student 4 图表生成脚本
生成内容：
1. 四种算法综合性能对比（雷达图 + 柱状图）
2. 噪声消融实验详细分析
3. 奖励场 Pearson/Spearman 相关性分析
4. 策略轨迹可视化对比
5. 计算复杂度与运行时间对比
"""
import os, sys, pickle, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
from irl.lp_irl import solve_lp_irl
from irl.mm_irl import solve_mm_irl
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
mu_subs = generate_suboptimal_mus(env)

# 运行所有算法并计时
timings = {}

t0 = time.time()
w_lp = solve_lp_irl(mu_E, mu_subs, env.Phi.shape[1])
timings["LP-IRL"] = time.time() - t0

t0 = time.time()
w_mm, margin = solve_mm_irl(mu_E, mu_subs, env.Phi.shape[1])
timings["MM-IRL"] = time.time() - t0

t0 = time.time()
w_maxent = maxent_irl(mu_E, env, n_iters=300, lr=0.2)
timings["MaxEnt-IRL"] = time.time() - t0

t0 = time.time()
all_trajs = dataset["clean_trajectories"] + dataset["noisy_trajectories"]
w_pref = preference_irl(
    dataset["preference_pairs"], dataset["preference_labels"],
    all_trajs, env, n_iters=1000, lr=1.0
)
timings["Preference-BT"] = time.time() - t0

R_gt = env.ground_truth_reward
R_lp = env.Phi @ w_lp
R_mm = env.Phi @ w_mm
R_maxent = env.Phi @ w_maxent
R_pref = env.Phi @ w_pref

# ============================================================
# 图 S4-1: 综合性能雷达图
# ============================================================
def plot_radar_comparison():
    # 计算各指标
    names = ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]
    weights = [w_lp, w_mm, w_maxent, w_pref]
    rewards = [R_lp, R_mm, R_maxent, R_pref]

    # 评估策略
    def evaluate_policy(policy, n_trials=100):
        successes = 0
        total_steps = 0
        for _ in range(n_trials):
            s, _ = env.reset(options={'fixed_start': (0, 0)})
            done = False
            steps = 0
            while not done and steps < 150:
                ns, _, term, trunc, _ = env.step(policy[s])
                done = term or trunc
                s = ns
                steps += 1
            if env._idx_to_pos(s) == env.goal_pos:
                successes += 1
                total_steps += steps
        return successes / n_trials, total_steps / successes if successes > 0 else 150

    policies = []
    for R in rewards:
        policy, _ = env.compute_optimal_policy(reward_vec=R * 5.0)
        policies.append(policy)

    success_rates = []
    avg_steps_list = []
    pearsons = []
    spearman_vals = []

    for name, R, policy in zip(names, rewards, policies):
        acc, steps = evaluate_policy(policy)
        success_rates.append(acc)
        avg_steps_list.append(150 - steps)  # 反转，步数越少越好

        pearson = np.corrcoef(R.flatten(), R_gt.flatten())[0, 1]
        spearman_val = spearmanr(R.flatten(), R_gt.flatten())[0]
        pearsons.append(abs(pearson))  # 取绝对值
        spearman_vals.append(abs(spearman_val))

    # 归一化到 [0, 1]
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / (arr.max() - arr.min())

    success_norm = normalize(success_rates)
    steps_norm = normalize(avg_steps_list)
    pearson_norm = normalize(pearsons)
    spearman_norm = normalize(spearman_vals)

    # 雷达图
    categories = ["Success Rate", "Path Efficiency", "Pearson Corr.", "Spearman Corr."]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#3498db", "#9b59b6", "#f39c12", "#e74c3c"]
    for i, (name, color) in enumerate(zip(names, colors)):
        values = [success_norm[i], steps_norm[i], pearson_norm[i], spearman_norm[i]]
        values += values[:1]
        ax.fill(angles, values, color=color, alpha=0.15)
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        for angle, val, cat in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, val + 0.05, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Metric Comparison of IRL Algorithms\n(Normalized Scores)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig("figures/s4_radar_comparison.png", dpi=300)
    print("✅ figures/s4_radar_comparison.png")
    plt.close()

# ============================================================
# 图 S4-2: 噪声消融实验（详细版）
# ============================================================
def plot_ablation_detailed():
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    ablation_results = {name: [] for name in ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]}
    ablation_pearson = {name: [] for name in ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]}

    from data.generate_expert import generate_trajectories, generate_preference_pairs

    for noise_idx, noise in enumerate(noise_levels):
        noisy_trajs = generate_trajectories(env, n_episodes=500, action_noise=noise,
                                           seed=42 + noise_idx * 1000)
        mu_noisy = compute_feature_expectation(noisy_trajs, env)

        pairs_test, labels_test, _ = generate_preference_pairs(
            noisy_trajs, n_pairs=300, pref_noise=0.05, seed=42 + noise_idx * 100)

        def evaluate(policy, n_trials=50):
            successes = 0
            for _ in range(n_trials):
                s, _ = env.reset(options={'fixed_start': (0, 0)})
                done = False
                steps = 0
                while not done and steps < 150:
                    ns, _, term, trunc, _ = env.step(policy[s])
                    done = term or trunc
                    s = ns
                    steps += 1
                if env._idx_to_pos(s) == env.goal_pos:
                    successes += 1
            return successes / n_trials

        # LP
        w_lp_n = solve_lp_irl(mu_noisy, mu_subs, env.Phi.shape[1])
        policy_lp_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_lp_n) * 5.0)
        ablation_results["LP-IRL"].append(evaluate(policy_lp_n))
        ablation_pearson["LP-IRL"].append(np.corrcoef(env.Phi @ w_lp_n, R_gt)[0, 1])

        # MM
        w_mm_n, _ = solve_mm_irl(mu_noisy, mu_subs, env.Phi.shape[1])
        policy_mm_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_mm_n) * 5.0)
        ablation_results["MM-IRL"].append(evaluate(policy_mm_n))
        ablation_pearson["MM-IRL"].append(np.corrcoef(env.Phi @ w_mm_n, R_gt)[0, 1])

        # MaxEnt
        w_maxent_n = maxent_irl(mu_noisy, env, n_iters=200, lr=0.2)
        policy_maxent_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_maxent_n) * 5.0)
        ablation_results["MaxEnt-IRL"].append(evaluate(policy_maxent_n))
        ablation_pearson["MaxEnt-IRL"].append(np.corrcoef(env.Phi @ w_maxent_n, R_gt)[0, 1])

        # Pref
        w_pref_n = preference_irl(pairs_test, labels_test, noisy_trajs, env, n_iters=500, lr=1.0)
        policy_pref_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_pref_n) * 5.0)
        ablation_results["Preference-BT"].append(evaluate(policy_pref_n))
        ablation_pearson["Preference-BT"].append(np.corrcoef(env.Phi @ w_pref_n, R_gt)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # (a) 策略成功率
    colors = {"LP-IRL": "#3498db", "MM-IRL": "#9b59b6", "MaxEnt-IRL": "#f39c12", "Preference-BT": "#e74c3c"}
    for name in ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]:
        axes[0].plot(noise_levels, ablation_results[name], 'o-', linewidth=2, markersize=8,
                    color=colors[name], label=name)

    # Ground Truth 和 Random 基线
    policy_gt, _ = env.compute_optimal_policy()
    acc_gt = 0
    for _ in range(50):
        s, _ = env.reset(options={'fixed_start': (0, 0)})
        done = False
        while not done:
            ns, _, term, trunc, _ = env.step(policy_gt[s])
            done = term or trunc
            s = ns
        if env._idx_to_pos(s) == env.goal_pos:
            acc_gt += 1
    acc_gt /= 50

    axes[0].axhline(y=acc_gt, color="green", linestyle="--", linewidth=2, alpha=0.7, label=f"Ground Truth ({acc_gt:.0%})")
    axes[0].axhline(y=0.25, color="gray", linestyle=":", linewidth=2, label="Random (25%)")

    axes[0].set_xlabel("Expert Action Noise Level", fontsize=12)
    axes[0].set_ylabel("Policy Success Rate", fontsize=12)
    axes[0].set_title("Noise Robustness: Strategy Success Rate", fontsize=13)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) 奖励相关性
    for name in ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]:
        axes[1].plot(noise_levels, ablation_pearson[name], 's-', linewidth=2, markersize=8,
                    color=colors[name], label=name)
    axes[1].set_xlabel("Expert Action Noise Level", fontsize=12)
    axes[1].set_ylabel("Pearson Correlation with GT", fontsize=12)
    axes[1].set_title("Noise Robustness: Reward Correlation", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Noise Ablation Study: Algorithm Robustness Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s4_ablation_detailed.png", dpi=300)
    print("✅ figures/s4_ablation_detailed.png")
    plt.close()

# ============================================================
# 图 S4-3: 奖励场相关性矩阵
# ============================================================
def plot_reward_correlation():
    names = ["Ground Truth", "LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]
    rewards = [R_gt, R_lp, R_mm, R_maxent, R_pref]

    n = len(rewards)
    pearson_matrix = np.zeros((n, n))
    spearman_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            pearson_matrix[i, j] = np.corrcoef(rewards[i].flatten(), rewards[j].flatten())[0, 1]
            spearman_matrix[i, j] = spearmanr(rewards[i].flatten(), rewards[j].flatten())[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pearson
    im1 = axes[0].imshow(pearson_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_title("Pearson Correlation Matrix\n(Reward Fields)", fontsize=12)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{pearson_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(pearson_matrix[i, j]) > 0.5 else "black")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Spearman
    im2 = axes[1].imshow(spearman_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].set_title("Spearman Rank Correlation Matrix\n(Reward Fields)", fontsize=12)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{spearman_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(spearman_matrix[i, j]) > 0.5 else "black")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Reward Field Correlation Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s4_reward_correlation.png", dpi=300)
    print("✅ figures/s4_reward_correlation.png")
    plt.close()

# ============================================================
# 图 S4-4: 策略轨迹对比
# ============================================================
def plot_policy_trajectories():
    policies = []
    for R, name in [(R_lp, "LP-IRL"), (R_mm, "MM-IRL"), (R_maxent, "MaxEnt-IRL"), (R_pref, "Preference-BT")]:
        policy, _ = env.compute_optimal_policy(reward_vec=R * 5.0)
        policies.append((policy, name))

    # 也加一个 Ground Truth
    policy_gt, _ = env.compute_optimal_policy()
    policies.append((policy_gt, "Ground Truth"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (policy, name) in zip(axes, policies):
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect("equal")
        ax.set_xticks(range(env.size + 1))
        ax.set_yticks(range(env.size + 1))
        ax.grid(True, color="lightgray", linewidth=0.5)
        ax.invert_yaxis()

        # 障碍物
        for r, c in obstacles:
            ax.add_patch(mpatches.Rectangle((c, r), 1, 1, facecolor="black", alpha=0.7))
        # 终点
        gr, gc = env.goal_pos
        ax.add_patch(mpatches.Rectangle((gc, gr), 1, 1, facecolor="gold", edgecolor="red", linewidth=2))
        ax.text(gc + 0.5, gr + 0.5, "G", color="red", fontsize=14, ha="center", va="center", fontweight="bold")

        # 绘制 20 条从 (0,0) 出发的轨迹
        n_show = 20
        for ep in range(n_show):
            s, _ = env.reset(options={'fixed_start': (0, 0)}, seed=42 + ep * 100)
            states = [env._idx_to_pos(s)]
            for _ in range(150):
                ns, _, term, trunc, _ = env.step(policy[s])
                s = ns
                states.append(env._idx_to_pos(s))
                if term or trunc:
                    break
            rows, cols = zip(*states)
            ax.plot(cols, rows, alpha=0.4, linewidth=1.5)

        ax.set_title(f"{name} Policy Trajectories\n(20 trials from (0,0))", fontsize=11)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    # 关闭多余的子图
    for i in range(len(policies), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Policy Trajectories: IRL Algorithms vs Ground Truth", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s4_policy_trajectories.png", dpi=300)
    print("✅ figures/s4_policy_trajectories.png")
    plt.close()

# ============================================================
# 图 S4-5: 计算复杂度对比
# ============================================================
def plot_complexity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Preference-BT"]
    times = [timings[n] for n in names]

    # (a) 运行时间
    colors = ["#3498db", "#9b59b6", "#f39c12", "#e74c3c"]
    axes[0].bar(names, times, color=colors, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Runtime (seconds)", fontsize=11)
    axes[0].set_title("Algorithm Runtime Comparison\n(Single Run, Same Machine)", fontsize=12)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, (name, t) in enumerate(zip(names, times)):
        axes[0].text(i, t + max(times) * 0.02, f"{t:.2f}s", ha="center", fontsize=10)

    # (b) 算法复杂度分析
    complexities = {
        "LP-IRL": "O(F³ + K·F²)\n(LP solver, F features, K constraints)",
        "MM-IRL": "O(F³ + K·F²)\n(SOCP solver, iterative)",
        "MaxEnt-IRL": "O(T·S³ + T·S²·A)\n(T iters, Soft VI per iter)",
        "Preference-BT": "O(T·N²·F)\n(T iters, N pairs, F features)"
    }

    axes[1].axis("off")
    y_pos = 0.9
    for name, comp in complexities.items():
        color = colors[names.index(name)]
        axes[1].text(0.1, y_pos, f"{name}:", fontsize=12, fontweight="bold", color=color,
                    transform=axes[1].transAxes)
        axes[1].text(0.35, y_pos, comp, fontsize=10, transform=axes[1].transAxes)
        y_pos -= 0.2

    axes[1].set_title("Theoretical Complexity Analysis", fontsize=12)

    plt.suptitle("Algorithm Complexity and Runtime", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/s4_complexity.png", dpi=300)
    print("✅ figures/s4_complexity.png")
    plt.close()

# ============================================================
# 运行全部
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Student 4 图表生成")
    print("=" * 60)
    plot_radar_comparison()
    plot_ablation_detailed()
    plot_reward_correlation()
    plot_policy_trajectories()
    plot_complexity()
    print("\n✅ 所有 Student 4 图表生成完成！")
