"""
诊断为什么IRL学不到真实奖励 - 检查特征期望
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation
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
        feature_mode="coords"
    )
    
    print("="*70)
    print("诊断：特征期望分析")
    print("="*70)
    
    # 计算专家特征期望
    mu_E = compute_feature_expectation(dataset['clean_trajectories'], env)
    
    print(f"\n专家特征期望 mu_E: {mu_E}")
    print(f"Phi shape: {env.Phi.shape}")
    print(f"Phi[:5]:\n{env.Phi[:5]}")
    
    # 真实奖励对应的特征权重
    # Ground Truth: R(goal)=1, R(obstacle)=-1, others=0
    # 但这是非线性奖励，无法用线性特征表示！
    
    print("\n" + "="*70)
    print("问题1：线性特征无法表示稀疏奖励")
    print("="*70)
    print(f"""
真实奖励函数 R_gt:
  - (9,9) 目标: R=1.0
  - 障碍物: R=-1.0
  - 其他: R=0.0

线性近似 R = Phi @ w:
  - Phi[s] = [row/9, col/9] (只有2维!)
  - w = [w1, w2]
  - R[s] = w1*(row/9) + w2*(col/9)

这只能表示"梯度场"，无法表示"只在目标有值"!
""")
    
    # 测试不同的特征模式
    print("="*70)
    print("测试：用onehot特征能否学到真实奖励？")
    print("="*70)
    
    env_onehot = GridWorldEnv(
        size=cfg['size'], 
        goal_pos=cfg['goal'], 
        obstacles=obstacles,
        stochasticity=cfg['stochasticity'], 
        feature_mode="onehot"
    )
    
    print(f"\nOnehot Phi shape: {env_onehot.Phi.shape}")
    print(f"这是 100x100 的矩阵（每个状态一个维度）")
    
    mu_E_onehot = compute_feature_expectation(dataset['clean_trajectories'], env_onehot)
    print(f"专家特征期望 mu_E_onehot[:10]: {mu_E_onehot[:10]}")
    
    # 用最小二乘拟合最优权重
    # mu_E = Phi^T @ d @ R, 其中 d 是折扣状态分布
    # 简化：假设均匀分布，w = (Phi^T Phi)^{-1} Phi^T mu_E
    
    print("\n" + "="*70)
    print("解决方案分析")
    print("="*70)
    print("""
【问题根源】
当前使用 2D 坐标特征 [row/9, col/9]：
  ✓ 优点：平滑、可泛化
  ✗ 缺点：只能表示线性梯度，无法表示稀疏奖励

【提高相关性的方法】

方法1：使用 Onehot 特征（100维）
  ✓ 可以完美表示任何奖励函数
  ✗ 无法泛化到新状态，参数量大
  ✓ Pearson 相关会很高

方法2：增加多项式特征
  添加 row², col², row×col 等交叉项
  ✓ 能表示更复杂的奖励形状
  ✓ 仍保持一定泛化能力

方法3：使用径向基函数（RBF）
  以目标为中心的高斯核
  ✓ 能表示"只在目标有值"的奖励
  ✓ 这是最接近真实奖励的特征

【推荐】
对于这个项目，建议：
1. 改用 onehot 特征（最简单）
2. 或者添加 RBF 特征（更学术）
3. 在报告中讨论特征选择的影响
""")
    
    # 演示 RBF 特征
    print("\n" + "="*70)
    print("演示：RBF 特征")
    print("="*70)
    
    goal_row, goal_col = cfg['goal']
    
    # 创建 RBF 特征：以目标为中心的高斯
    Phi_rbf = np.zeros((env.n_states, 3))
    for s in range(env.n_states):
        r, c = env._idx_to_pos(s)
        dist_to_goal = np.sqrt((r - goal_row)**2 + (c - goal_col)**2)
        dist_to_nearest_obs = min([
            np.sqrt((r - or_)**2 + (c - oc)**2) 
            for or_, oc in obstacles
        ]) if obstacles else 999
        
        # 特征1：距离目标越近越大
        Phi_rbf[s, 0] = np.exp(-dist_to_goal**2 / (2 * 3.0**2))
        # 特征2：距离障碍越远越大
        Phi_rbf[s, 1] = 1.0 - np.exp(-dist_to_nearest_obs**2 / (2 * 2.0**2))
        # 特征3：常数偏置
        Phi_rbf[s, 2] = 1.0
    
    print(f"RBF Phi shape: {Phi_rbf.shape}")
    print(f"Phi[:5]:\n{Phi_rbf[:5]}")
    print(f"\n目标(9,9)的RBF: {Phi_rbf[env._pos_to_idx((9,9))]}")
    print(f"障碍物(3,3)的RBF: {Phi_rbf[env._pos_to_idx((3,3))]}")
    print(f"起点(0,0)的RBF: {Phi_rbf[env._pos_to_idx((0,0))]}")
    
    # 用最小二乘拟合
    print("\n" + "="*70)
    print("用最小二乘拟合最优 RBF 权重")
    print("="*70)
    
    R_gt = env.ground_truth_reward
    
    # w = (Phi^T Phi)^{-1} Phi^T R
    w_optimal = np.linalg.lstsq(Phi_rbf, R_gt, rcond=None)[0]
    R_fitted = Phi_rbf @ w_optimal
    
    pearson = np.corrcoef(R_gt.flatten(), R_fitted.flatten())[0, 1]
    print(f"\n最优权重: {w_optimal}")
    print(f"拟合奖励 Pearson 相关: {pearson:.3f}")
    print(f"R_gt: min={R_gt.min():.3f}, max={R_gt.max():.3f}")
    print(f"R_fitted: min={R_fitted.min():.3f}, max={R_fitted.max():.3f}")
    print(f"\n目标(9,9): R_gt={R_gt[env._pos_to_idx((9,9))]:.3f}, R_fitted={R_fitted[env._pos_to_idx((9,9))]:.3f}")
    print(f"障碍物(3,3): R_gt={R_gt[env._pos_to_idx((3,3))]:.3f}, R_fitted={R_fitted[env._pos_to_idx((3,3))]:.3f}")

if __name__ == "__main__":
    main()
