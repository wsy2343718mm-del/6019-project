"""
统一评估与对比实验脚本
功能：
1. 运行所有四种 IRL 算法
2. 在固定起点 (0,0) 评估策略成功率
3. 输出清晰的对比报告
4. 噪声消融实验
5. 生成可视化图表
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from env.gridworld import GridWorldEnv
from data.generate_expert import generate_trajectories, generate_preference_pairs
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
from irl.lp_irl import solve_lp_irl
from irl.mm_irl import solve_mm_irl
from irl.maxent_irl import maxent_irl
from irl.preference_irl import preference_irl

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# 评估函数（固定起点 0,0）
# ============================================================
def evaluate_policy(env, policy, n_trials=100):
    """在固定起点 (0,0) 评估策略成功率"""
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
    avg_steps = total_steps / successes if successes > 0 else 150
    return successes / n_trials, avg_steps

def compute_metrics(R_recovered, R_gt, goal_state):
    """计算恢复奖励与真实奖励的指标"""
    pearson = np.corrcoef(R_recovered.flatten(), R_gt.flatten())[0, 1]
    spearman = spearmanr(R_recovered.flatten(), R_gt.flatten())[0]
    goal_rank = np.argsort(R_recovered)[::-1].tolist().index(goal_state) + 1
    return pearson, spearman, goal_rank

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("📊 IRL 算法统一评估与对比实验")
    print("=" * 70)
    print("\n📌 评估说明：")
    print("   • 起点：固定在 (0,0)")
    print("   • 终点：(9,9)")
    print("   • 环境随机性：10%")
    print("   • 评估次数：每个策略 100 次")
    
    # 加载数据
    print("\n" + "-" * 70)
    print("【步骤 1】加载数据")
    print("-" * 70)
    with open('data/expert_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    cfg = dataset['env_config']
    obstacles = cfg.get('obstacles', [])  # 获取障碍物列表
    print(f"   环境: {cfg['size']}x{cfg['size']}, 目标: {cfg['goal']}")
    print(f"   障碍物数量: {len(obstacles)}")
    if obstacles:
        print(f"   障碍物位置: {obstacles}")
    print(f"   干净轨迹: {len(dataset['clean_trajectories'])} 条")
    print(f"   噪声轨迹: {len(dataset['noisy_trajectories'])} 条")
    print(f"   偏好对: {len(dataset['preference_pairs'])} 对")
    
    # 重建环境（包含障碍物）
    env = GridWorldEnv(size=cfg['size'], goal_pos=cfg['goal'], obstacles=obstacles,
                       stochasticity=cfg['stochasticity'], feature_mode="rbf", gamma=cfg['gamma'])
    R_gt = env.ground_truth_reward
    goal_state = env._pos_to_idx(cfg['goal'])
    
    # 缩放真实奖励以匹配RBF特征的范围
    # RBF特征的奖励范围约-0.5到0.7，需要放大到-1到1
    scale_factor = 1.0 / max(abs(R_gt.min()), abs(R_gt.max()))
    R_gt_scaled = R_gt * scale_factor
    
    # 计算特征期望
    mu_E = compute_feature_expectation(dataset['clean_trajectories'], env)
    mu_subs = generate_suboptimal_mus(env)
    
    # ============================================================
    # 运行所有算法
    # ============================================================
    print("\n" + "-" * 70)
    print("【步骤 2】运行 IRL 算法")
    print("-" * 70)
    
    results = {}
    
    # LP-IRL
    print("\n🔹 LP-IRL (Linear Programming IRL)")
    w_lp = solve_lp_irl(mu_E, mu_subs, env.Phi.shape[1])
    R_lp = env.Phi @ w_lp
    # 缩放奖励以增强信号
    R_lp_scaled = R_lp * 5.0
    policy_lp, _ = env.compute_optimal_policy(reward_vec=R_lp_scaled)
    acc_lp, steps_lp = evaluate_policy(env, policy_lp)
    pearson_lp, spearman_lp, rank_lp = compute_metrics(R_lp_scaled, R_gt, goal_state)
    results['LP-IRL'] = {
        'weight': w_lp, 'success_rate': acc_lp, 'avg_steps': steps_lp,
        'pearson': pearson_lp, 'spearman': spearman_lp, 'goal_rank': rank_lp
    }
    print(f"   权重: {w_lp.round(3)}")
    print(f"   策略成功率: {acc_lp:.1%}")

    # MM-IRL
    print("\n🔹 MM-IRL (Maximum Margin IRL)")
    w_mm, margin = solve_mm_irl(mu_E, mu_subs, env.Phi.shape[1])
    R_mm = env.Phi @ w_mm
    R_mm_scaled = R_mm * 5.0
    policy_mm, _ = env.compute_optimal_policy(reward_vec=R_mm_scaled)
    acc_mm, steps_mm = evaluate_policy(env, policy_mm)
    pearson_mm, spearman_mm, rank_mm = compute_metrics(R_mm_scaled, R_gt, goal_state)
    results['MM-IRL'] = {
        'weight': w_mm, 'success_rate': acc_mm, 'avg_steps': steps_mm,
        'pearson': pearson_mm, 'spearman': spearman_mm, 'goal_rank': rank_mm,
        'margin': margin
    }
    print(f"   权重: {w_mm.round(3)}, 边距: {margin:.2f}")
    print(f"   策略成功率: {acc_mm:.1%}")

    # MaxEnt-IRL
    print("\n🔹 MaxEnt-IRL (Maximum Entropy IRL)")
    w_maxent = maxent_irl(mu_E, env, n_iters=300, lr=0.2)
    R_maxent = env.Phi @ w_maxent
    R_maxent_scaled = R_maxent * 5.0
    policy_maxent, _ = env.compute_optimal_policy(reward_vec=R_maxent_scaled)
    acc_maxent, steps_maxent = evaluate_policy(env, policy_maxent)
    pearson_maxent, spearman_maxent, rank_maxent = compute_metrics(R_maxent_scaled, R_gt, goal_state)
    results['MaxEnt-IRL'] = {
        'weight': w_maxent, 'success_rate': acc_maxent, 'avg_steps': steps_maxent,
        'pearson': pearson_maxent, 'spearman': spearman_maxent, 'goal_rank': rank_maxent
    }
    print(f"   权重: {w_maxent.round(3)}")
    print(f"   策略成功率: {acc_maxent:.1%}")

    # Preference-BT
    print("\n🔹 Preference-BT (Bradley-Terry Preference Learning)")
    all_trajs = dataset['clean_trajectories'] + dataset['noisy_trajectories']
    w_pref = preference_irl(dataset['preference_pairs'], dataset['preference_labels'],
                           all_trajs, env, n_iters=1000, lr=1.0)
    R_pref = env.Phi @ w_pref
    R_pref_scaled = R_pref * 5.0
    policy_pref, _ = env.compute_optimal_policy(reward_vec=R_pref_scaled)
    acc_pref, steps_pref = evaluate_policy(env, policy_pref)
    pearson_pref, spearman_pref, rank_pref = compute_metrics(R_pref_scaled, R_gt, goal_state)
    results['Preference-BT'] = {
        'weight': w_pref, 'success_rate': acc_pref, 'avg_steps': steps_pref,
        'pearson': pearson_pref, 'spearman': spearman_pref, 'goal_rank': rank_pref
    }
    print(f"   权重: {w_pref.round(3)}")
    print(f"   策略成功率: {acc_pref:.1%}")
    
    # Baseline: Ground Truth 和 Random
    print("\n🔹 Ground Truth (真实奖励)")
    policy_gt, _ = env.compute_optimal_policy()
    acc_gt, steps_gt = evaluate_policy(env, policy_gt)
    results['Ground Truth'] = {'success_rate': acc_gt, 'avg_steps': steps_gt}
    print(f"   策略成功率: {acc_gt:.1%}")
    
    print("\n🔹 Random Policy (随机策略)")
    policy_random = np.random.randint(0, 4, size=env.n_states)
    acc_random, _ = evaluate_policy(env, policy_random)
    results['Random'] = {'success_rate': acc_random}
    print(f"   策略成功率: {acc_random:.1%}")
    
    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 70)
    print("【步骤 3】策略性能对比表")
    print("=" * 70)
    print("\n📊 所有策略均在固定起点 (0,0) 评估 100 次：\n")
    print("| 方法          | 成功率 | 平均步数 | Pearson | 目标排名 |")
    print("|---------------|--------|----------|---------|----------|")
    print(f"| Ground Truth  | {acc_gt:.1%}  | {steps_gt:.1f}     | 1.000   | 1/100    |")
    print(f"| Random        | {acc_random:.1%}  | -        | -       | -        |")
    print("|---------------|--------|----------|---------|----------|")
    for name in ['LP-IRL', 'MM-IRL', 'MaxEnt-IRL', 'Preference-BT']:
        r = results[name]
        print(f"| {name:13} | {r['success_rate']:.1%}  | {r['avg_steps']:.1f}     | {r['pearson']:.3f}   | {r['goal_rank']}/100    |")
    
    # ============================================================
    # 噪声消融实验
    # ============================================================
    print("\n" + "=" * 70)
    print("【步骤 4】噪声消融实验")
    print("=" * 70)
    print("\n📊 测试不同噪声水平对算法性能的影响：\n")
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    ablation_results = {name: [] for name in ['LP-IRL', 'MM-IRL', 'MaxEnt-IRL', 'Preference-BT']}
    
    for noise_idx, noise in enumerate(noise_levels):
        print(f"Noise={noise:.1f}: ", end="")
        
        # 生成含噪轨迹
        noisy_trajs = generate_trajectories(env, n_episodes=500, action_noise=noise, 
                                           seed=42 + noise_idx * 1000)
        mu_noisy = compute_feature_expectation(noisy_trajs, env)
        
        # 生成偏好对
        pairs_test, labels_test, _ = generate_preference_pairs(
            noisy_trajs, n_pairs=300, pref_noise=0.05, seed=42 + noise_idx * 100)
        
        # LP-IRL
        w_lp_n = solve_lp_irl(mu_noisy, mu_subs, env.Phi.shape[1])
        policy_lp_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_lp_n) * 5.0)
        acc_lp_n, _ = evaluate_policy(env, policy_lp_n)
        ablation_results['LP-IRL'].append(acc_lp_n)

        # MM-IRL
        w_mm_n, _ = solve_mm_irl(mu_noisy, mu_subs, env.Phi.shape[1])
        policy_mm_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_mm_n) * 5.0)
        acc_mm_n, _ = evaluate_policy(env, policy_mm_n)
        ablation_results['MM-IRL'].append(acc_mm_n)

        # MaxEnt-IRL
        w_maxent_n = maxent_irl(mu_noisy, env, n_iters=200, lr=0.2)
        policy_maxent_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_maxent_n) * 5.0)
        acc_maxent_n, _ = evaluate_policy(env, policy_maxent_n)
        ablation_results['MaxEnt-IRL'].append(acc_maxent_n)

        # Preference-BT
        w_pref_n = preference_irl(pairs_test, labels_test, noisy_trajs, env, n_iters=500, lr=1.0)
        policy_pref_n, _ = env.compute_optimal_policy(reward_vec=(env.Phi @ w_pref_n) * 5.0)
        acc_pref_n, _ = evaluate_policy(env, policy_pref_n)
        ablation_results['Preference-BT'].append(acc_pref_n)
        
        print(f"LP={acc_lp_n:.0%}, MM={acc_mm_n:.0%}, MaxEnt={acc_maxent_n:.0%}, Pref={acc_pref_n:.0%}")
    
    # 消融实验汇总表
    print("\n📊 消融实验汇总表：\n")
    print("| Noise | LP-IRL | MM-IRL | MaxEnt | Preference-BT |")
    print("|-------|--------|--------|--------|---------------|")
    for i, noise in enumerate(noise_levels):
        print(f"| {noise:.1f}   | {ablation_results['LP-IRL'][i]:.0%}   | {ablation_results['MM-IRL'][i]:.0%}   | {ablation_results['MaxEnt-IRL'][i]:.0%}   | {ablation_results['Preference-BT'][i]:.0%}           |")
    
    # ============================================================
    # 生成可视化
    # ============================================================
    print("\n" + "=" * 70)
    print("【步骤 5】生成可视化图表")
    print("=" * 70)
    
    os.makedirs('figures', exist_ok=True)
    
    # 图1: 成功率对比
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ['Ground\nTruth', 'Random', 'LP-IRL', 'MM-IRL', 'MaxEnt', 'Preference-BT']
    accs = [acc_gt, acc_random, acc_lp, acc_mm, acc_maxent, acc_pref]
    colors = ['#2ecc71', '#95a5a6', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Policy Success Rate Comparison (Fixed Start: 0,0)', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=acc_gt, color='green', linestyle='--', linewidth=2, alpha=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.0%}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/fig_success_comparison.png', dpi=300)
    print("   ✅ figures/fig_success_comparison.png")
    plt.close()
    
    # 图2: 噪声消融曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in [('LP-IRL', '#3498db'), ('MM-IRL', '#9b59b6'), 
                        ('MaxEnt-IRL', '#f39c12'), ('Preference-BT', '#e74c3c')]:
        ax.plot(noise_levels, ablation_results[name], 'o-', linewidth=2, 
                markersize=8, color=color, label=name)
    ax.axhline(y=acc_gt, color='green', linestyle='--', linewidth=2, label=f'Ground Truth ({acc_gt:.0%})')
    ax.axhline(y=acc_random, color='gray', linestyle=':', linewidth=2, label=f'Random ({acc_random:.0%})')
    ax.set_xlabel('Expert Action Noise Level', fontsize=12)
    ax.set_ylabel('Policy Success Rate', fontsize=12)
    ax.set_title('Ablation Study: Algorithm Robustness to Noise', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig_ablation_all.png', dpi=300)
    print("   ✅ figures/fig_ablation_all.png")
    plt.close()
    
    # 图3: 奖励场热力图（配色参考 fig_reward_comparison.png）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    reward_grids = [R_gt, R_lp, R_mm, R_maxent, R_pref]
    titles = ['Ground Truth',
              'LP-IRL',
              'MM-IRL',
              'MaxEnt-IRL',
              'Preference-BT']
    
    # 配色方案：Ground Truth 用 YlOrRd，其他用 YlGnBu
    cmaps = ['YlOrRd', 'YlGnBu', 'YlGnBu', 'YlGnBu', 'YlGnBu']

    for idx, (ax, R, title, cmap) in enumerate(zip(axes.flatten()[:5], reward_grids, titles, cmaps)):
        # 创建掩码：障碍物位置为True
        mask = np.zeros((env.size, env.size), dtype=bool)
        for obs_r, obs_c in obstacles:
            mask[obs_r, obs_c] = True

        # 每个子图使用独立色标
        data_display = R.reshape(env.size, env.size).copy()
        
        sns.heatmap(data_display, annot=True, fmt=".2f",
                    cmap=cmap, ax=ax,
                    cbar=True, annot_kws={"size": 7},
                    mask=mask, linewidth=0.5, linecolor='gray')
        
        # 在障碍物位置画X
        for obs_r, obs_c in obstacles:
            ax.add_patch(plt.Rectangle((obs_c, obs_r), 1, 1, fill=True,
                                       facecolor='black', edgecolor='darkgray', 
                                       linewidth=1.5, alpha=0.8))
            ax.text(obs_c + 0.5, obs_r + 0.5, 'X', color='white', 
                   fontsize=10, fontweight='bold', ha='center', va='center')
        
        # 标记目标位置
        ax.add_patch(plt.Rectangle((9, 9), 1, 1, fill=False, edgecolor='blue', 
                                  linewidth=3, linestyle='--'))
        ax.text(9.5, 9.5, 'G', color='blue', fontsize=14, fontweight='bold',
               ha='center', va='center')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Column', fontsize=10)
        ax.set_ylabel('Row', fontsize=10)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.set_xticklabels(range(env.size))
        ax.set_yticklabels(range(env.size))
    
    # 第六个子图：数据说明
    ax = axes[1, 2]
    ax.text(0.5, 0.97, 'Reward Field Analysis', fontsize=14, ha='center',
            transform=ax.transAxes, weight='bold')

    obs_info = f"Obstacles: {len(obstacles)}" if obstacles else "No obstacles"
    info_text = f"""
Environment:
  • Size: 10x10
  • Goal: (9,9) - blue 'G'
  • {obs_info} - black 'X'

Ground Truth:
  • Goal = 1.0, Obstacles = -1.0
  • Other positions = 0

IRL Methods:
  • LP/MM: 2D coord features
  • MaxEnt/Pref: learned weights

Note:
  • Pearson correlation is LOW
    due to Reward Ambiguity
  • Different rewards → Same policy
  • This is EXPECTED in IRL
"""
    ax.text(0.05, 0.90, info_text, fontsize=9, ha='left', transform=ax.transAxes,
            va='top', family='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/fig_reward_heatmap.png', dpi=300)
    print("   ✅ figures/fig_reward_heatmap.png")
    plt.close()
    
    # ============================================================
    # 保存结果
    # ============================================================
    final_results = {
        'main_results': results,
        'ablation_results': ablation_results,
        'noise_levels': noise_levels
    }
    with open('data/comparison_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 70)
    print("【实验完成】")
    print("=" * 70)
    print(f"\n✅ 关键结论：")
    print(f"   • Ground Truth 策略成功率: {acc_gt:.0%}")
    print(f"   • IRL 算法成功率范围: {min(acc_lp, acc_mm, acc_maxent, acc_pref):.0%} - {max(acc_lp, acc_mm, acc_maxent, acc_pref):.0%}")
    print(f"   • Random 策略成功率: {acc_random:.0%}")
    
    irl_accs = [acc_lp, acc_mm, acc_maxent, acc_pref]
    if max(irl_accs) >= acc_gt * 0.9:
        print(f"\n✅ IRL 恢复的奖励能有效训练策略！")
    else:
        print(f"\n⚠️ IRL 算法表现需要进一步优化")

if __name__ == "__main__":
    main()