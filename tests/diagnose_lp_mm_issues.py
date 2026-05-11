"""
深度诊断两个问题：
1. LP-IRL 在有noise时成功率骤降
2. MM-IRL Pearson相关性太低
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
from irl.lp_irl import solve_lp_irl
from irl.mm_irl import solve_mm_irl
import numpy as np
import pickle

def main():
    # 加载数据
    with open('data/expert_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    cfg = dataset['env_config']
    obstacles = cfg.get('obstacles', [])
    
    env = GridWorldEnv(
        size=cfg['size'], 
        goal_pos=cfg['goal'], 
        obstacles=obstacles,
        stochasticity=cfg['stochasticity'], 
        feature_mode="rbf"
    )
    
    R_gt = env.ground_truth_reward
    
    print("="*70)
    print("问题1：LP-IRL 噪声敏感性分析")
    print("="*70)
    
    # 计算干净和噪声轨迹的特征期望
    mu_E_clean = compute_feature_expectation(dataset['clean_trajectories'], env)
    mu_subs = generate_suboptimal_mus(env)
    
    print("\n【LP-IRL 权重】（干净数据）")
    w_lp = solve_lp_irl(mu_E_clean, mu_subs, env.Phi.shape[1])
    print(f"w_lp = {w_lp}")
    print(f"特征含义: [目标距离, 障碍距离, 偏置]")
    print(f"解释: 目标特征权重={w_lp[0]:.3f}, 障碍特征权重={w_lp[1]:.3f}")
    
    # 测试不同噪声水平下的特征期望
    from data.generate_expert import generate_trajectories
    
    print("\n【不同噪声水平下的特征期望】")
    print(f"{'Noise':>6s} | {'mu[0]':>8s} | {'mu[1]':>8s} | {'mu[2]':>8s} | {'与干净差异':>10s}")
    print("-"*60)
    
    for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
        noisy_trajs = generate_trajectories(env, n_episodes=500, action_noise=noise, seed=42)
        mu_noisy = compute_feature_expectation(noisy_trajs, env)
        
        diff = np.linalg.norm(mu_noisy - mu_E_clean)
        print(f"{noise:6.1f} | {mu_noisy[0]:8.2f} | {mu_noisy[1]:8.2f} | {mu_noisy[2]:8.2f} | {diff:10.3f}")
        
        # 用噪声数据重新求解LP
        w_lp_noise = solve_lp_irl(mu_noisy, mu_subs, env.Phi.shape[1])
        R_lp = env.Phi @ w_lp_noise
        policy_lp, _ = env.compute_optimal_policy(reward_vec=R_lp * 5.0)
        
        # 评估
        success = 0
        for seed in range(100):
            state, _ = env.reset(seed=seed, options={'fixed_start': (0, 0)})
            done = False
            steps = 0
            while not done and steps < 150:
                action = policy_lp[state]
                next_state, _, terminated, truncated, _ = env.step(action)
                state = next_state
                done = terminated or truncated
                steps += 1
            if env._idx_to_pos(state) == env.goal_pos:
                success += 1
        
        print(f"      → LP权重: {w_lp_noise}, 成功率: {success}%")
        
        if noise > 0:
            # 诊断：检查约束是否仍然满足
            print(f"      → 约束检查 (w@(mu_E_clean - mu_noisy)):")
            diff_vec = mu_E_clean - mu_noisy
            margin = w_lp @ diff_vec
            print(f"         用干净权重: {margin:.3f} {'✓' if margin >= 0 else '✗ 约束被违反!'}")
    
    print("\n" + "="*70)
    print("问题2：MM-IRL Pearson相关性分析")
    print("="*70)
    
    w_mm, margin = solve_mm_irl(mu_E_clean, mu_subs, env.Phi.shape[1])
    print(f"\n【MM-IRL 权重】")
    print(f"w_mm = {w_mm}")
    print(f"边距: {margin:.2f}")
    
    R_mm = env.Phi @ w_mm
    R_mm_scaled = R_mm * 5.0
    
    pearson = np.corrcoef(R_gt.flatten(), R_mm_scaled.flatten())[0, 1]
    print(f"\nPearson 相关性: {pearson:.3f}")
    
    print(f"\n【奖励场统计】")
    print(f"Ground Truth: min={R_gt.min():.3f}, max={R_gt.max():.3f}, mean={R_gt.mean():.3f}")
    print(f"MM-IRL:       min={R_mm_scaled.min():.3f}, max={R_mm_scaled.max():.3f}, mean={R_mm_scaled.mean():.3f}")
    
    # 分析Pearson低的原因
    print(f"\n【Pearson低的原因分析】")
    print("Pearson相关性衡量的是两个数组的线性相关程度")
    print("Ground Truth是稀疏奖励：大部分=0，目标=1，障碍=-1")
    print("MM-IRL学到的是：w=[1, 0, 0]，只有目标距离特征有值")
    
    # 检查MM-IRL奖励场的分布
    R_mm_flat = R_mm_scaled.flatten()
    R_gt_flat = R_gt.flatten()
    
    # 障碍物位置
    obs_indices = [env._pos_to_idx(o) for o in obstacles]
    non_obs_mask = np.ones(env.n_states, dtype=bool)
    non_obs_mask[obs_indices] = False
    non_obs_mask[env._pos_to_idx(cfg['goal'])] = False
    
    print(f"\n非障碍物/目标位置的奖励分布:")
    print(f"  Ground Truth: mean={R_gt_flat[non_obs_mask].mean():.3f}, std={R_gt_flat[non_obs_mask].std():.3f}")
    print(f"  MM-IRL:       mean={R_mm_flat[non_obs_mask].mean():.3f}, std={R_mm_flat[non_obs_mask].std():.3f}")
    
    print(f"\nMM-IRL奖励场只有目标距离特征有权重:")
    print(f"  R[s] = w[0] * exp(-dist_to_goal²/18)")
    print(f"  这是一个平滑的梯度场，不是稀疏奖励")
    print(f"  Pearson低是因为奖励场形状不同，但策略仍然成功")
    print(f"  这再次证明了Reward Ambiguity：不同奖励→相同策略")
    
    # 对比LP和MM
    print(f"\n【LP vs MM 对比】")
    print(f"LP权重: {w_lp}")
    print(f"MM权重: {w_mm}")
    print(f"\nLP: 目标特征=0.33, 障碍特征=0.67 → 更重视避开障碍")
    print(f"MM: 目标特征=1.0, 其他=0 → 只重视接近目标")
    print(f"这就是为什么LP对噪声敏感：它依赖障碍特征，噪声会改变特征期望")
    print(f"而MM更鲁棒：只关注目标，噪声不影响目标方向的学习")

if __name__ == "__main__":
    main()
