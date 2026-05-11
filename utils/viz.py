import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_policy_quick(env, w):
    """快速评估策略成功率（单轮测试，仅用于可视化展示）"""
    R_rec = env.Phi @ w
    policy, _ = env.compute_optimal_policy(reward_vec=R_rec)
    s, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 150:  # 10x10 地图需要更多步数
        ns, r, term, trunc, _ = env.step(policy[s])
        done = term or trunc
        s = ns
        steps += 1
    return 1.0 if (env._idx_to_pos(s) == env.goal_pos) else 0.0

def save_comparison_plot(env, w_lp, w_mm, w_maxent, w_pref):
    """生成包含四种算法的对比图"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    names = ["LP-IRL", "MM-IRL", "MaxEnt-IRL", "Pref-BT"]
    weights = [w_lp, w_mm, w_maxent, w_pref]
    
    # 1. 预先计算所有奖励场并屏蔽障碍物
    reward_grids = []
    for w in weights:
        R = (env.Phi @ w).reshape(env.size, env.size)
        # 障碍物设为 NaN，热力图会自动留白
        for r in range(env.size):
            for c in range(env.size):
                if (r, c) in env.obstacles:
                    R[r, c] = np.nan
        reward_grids.append(R)
        
    # 2. 计算全局最大值和最小值（统一色标范围）
    all_vals = np.concatenate([r[~np.isnan(r)].flatten() for r in reward_grids])
    vmax = np.max(all_vals) if len(all_vals) > 0 else 1.0
    vmin = np.min(all_vals) if len(all_vals) > 0 else 0.0
    
    # 3. 绘图
    for ax, name, w, R in zip(axes, names, weights, reward_grids):
        acc = evaluate_policy_quick(env, w)
        
        sns.heatmap(R, annot=False, fmt=".2f", cmap="viridis", ax=ax, 
                    vmin=vmin, vmax=vmax, 
                    cbar=(ax == axes[-1]), 
                    cbar_kws={'label': 'Reward Value'} if ax == axes[-1] else {})
        
        ax.set_title(f"{name}\nSuccess: {acc:.0%}")
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig("figures/fig_reward_comparison.png", dpi=300)
    print("📸 对比图已保存至 figures/fig_reward_comparison.png")
    plt.close()