"""
测试RBF特征能否提高Pearson相关性和算法成功率
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation
import numpy as np
import pickle

def create_rbf_features(env, goal_pos, obstacles, sigma_goal=3.0, sigma_obs=2.0):
    """
    创建RBF特征矩阵
    
    特征1: 到目标的距离高斯 (越近越大)
    特征2: 到最近障碍物的距离 (越远越大)  
    特征3: 常数偏置
    """
    Phi_rbf = np.zeros((env.n_states, 3))
    
    for s in range(env.n_states):
        r, c = env._idx_to_pos(s)
        
        # 特征1: 距离目标越近，值越大
        dist_to_goal = np.sqrt((r - goal_pos[0])**2 + (c - goal_pos[1])**2)
        Phi_rbf[s, 0] = np.exp(-dist_to_goal**2 / (2 * sigma_goal**2))
        
        # 特征2: 距离障碍物越远，值越大
        if obstacles:
            dist_to_nearest_obs = min([
                np.sqrt((r - or_)**2 + (c - oc)**2) 
                for or_, oc in obstacles
            ])
            Phi_rbf[s, 1] = 1.0 - np.exp(-dist_to_nearest_obs**2 / (2 * sigma_obs**2))
        else:
            Phi_rbf[s, 1] = 1.0
        
        # 特征3: 常数偏置
        Phi_rbf[s, 2] = 1.0
    
    return Phi_rbf

def main():
    # 加载数据
    with open('data/expert_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    cfg = dataset['env_config']
    obstacles = cfg.get('obstacles', [])
    goal_pos = cfg['goal']
    
    env = GridWorldEnv(
        size=cfg['size'], 
        goal_pos=goal_pos, 
        obstacles=obstacles,
        stochasticity=cfg['stochasticity'], 
        feature_mode="coords"
    )
    
    print("="*70)
    print("RBF特征测试")
    print("="*70)
    
    # 创建RBF特征
    Phi_rbf = create_rbf_features(env, goal_pos, obstacles)
    
    print("\n【1】RBF特征矩阵")
    print("-"*70)
    print(f"Phi_rbf shape: {Phi_rbf.shape}")
    print(f"特征维度: 3 (goal_dist, obs_dist, bias)")
    print(f"\nPhi_rbf[:5]:\n{Phi_rbf[:5]}")
    
    goal_idx = env._pos_to_idx(goal_pos)
    print(f"\n目标(9,9)的RBF: {Phi_rbf[goal_idx]}")
    print(f"起点(0,0)的RBF: {Phi_rbf[0]}")
    print(f"障碍物(3,3)的RBF: {Phi_rbf[env._pos_to_idx((3,3))]}")
    
    # 用最小二乘拟合最优权重
    print("\n【2】最小二乘拟合")
    print("-"*70)
    R_gt = env.ground_truth_reward
    
    # w = argmin ||Phi @ w - R_gt||^2
    w_optimal = np.linalg.lstsq(Phi_rbf, R_gt, rcond=None)[0]
    R_fitted = Phi_rbf @ w_optimal
    
    pearson = np.corrcoef(R_gt.flatten(), R_fitted.flatten())[0, 1]
    
    print(f"最优权重: {w_optimal}")
    print(f"R_gt: min={R_gt.min():.3f}, max={R_gt.max():.3f}")
    print(f"R_fitted: min={R_fitted.min():.3f}, max={R_fitted.max():.3f}")
    print(f"\n✅ Pearson相关性: {pearson:.3f}")
    print(f"   (对比之前2D特征的0.006-0.027)")
    
    print(f"\n验证关键位置:")
    print(f"  目标(9,9): R_gt={R_gt[goal_idx]:.3f}, R_fitted={R_fitted[goal_idx]:.3f}")
    print(f"  起点(0,0): R_gt={R_gt[0]:.3f}, R_fitted={R_fitted[0]:.3f}")
    print(f"  障碍(3,3): R_gt={R_gt[env._pos_to_idx((3,3))]:.3f}, R_fitted={R_fitted[env._pos_to_idx((3,3))]:.3f}")
    
    # 测试用RBF特征训练策略
    print("\n【3】用RBF特征训练策略")
    print("-"*70)
    
    # 临时替换Phi
    Phi_backup = env.Phi.copy()
    env.Phi = Phi_rbf
    
    # 计算专家特征期望
    mu_E = compute_feature_expectation(dataset['clean_trajectories'], env)
    print(f"专家特征期望 mu_E: {mu_E}")
    
    # 用最优权重计算奖励
    R_rbf = Phi_rbf @ w_optimal
    policy_rbf, V_rbf = env.compute_optimal_policy(reward_vec=R_rbf)
    
    # 评估策略
    success_count = 0
    for seed in range(100):
        state, _ = env.reset(seed=seed, options={'fixed_start': (0, 0)})
        done = False
        steps = 0
        while not done and steps < 150:
            action = policy_rbf[state]
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            steps += 1
        
        if env._idx_to_pos(state) == goal_pos:
            success_count += 1
    
    print(f"策略成功率: {success_count}/100 = {success_count}%")
    
    # 恢复原Phi
    env.Phi = Phi_backup
    
    print("\n" + "="*70)
    print("【结论】")
    print("="*70)
    print(f"""
RBF特征 vs 2D坐标特征:
  Pearson: {pearson:.3f} vs 0.006-0.027 (提升{pearson/0.02:.0f}倍!)
  策略成功: {success_count}% vs 0-100% (取决于算法)
  
RBF特征的优势:
  ✓ 能表示"目标附近有高奖励"
  ✓ 能表示"障碍物附近有低奖励"
  ✓ 只有3维，不会过拟合
  ✓ Pearson显著提高
  
建议:
  修改 env/gridworld.py 添加 rbf 特征模式
  在 generate_expert.py 中使用 feature_mode="rbf"
""")

if __name__ == "__main__":
    main()
