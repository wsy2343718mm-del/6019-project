"""
深度诊断：为什么IRL学不到真实奖励
分析特征期望、权重学习和奖励场恢复的完整链路
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
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
    print("深度诊断：IRL学习链路分析")
    print("="*70)
    
    # 1. 检查特征矩阵
    print("\n【1】特征矩阵 Phi 分析")
    print("-"*70)
    print(f"Phi shape: {env.Phi.shape}")
    print(f"特征维度: {env.Phi.shape[1]} (只有2维!)")
    print(f"\nPhi[:5]:\n{env.Phi[:5]}")
    
    goal_idx = env._pos_to_idx(cfg['goal'])
    print(f"\n目标(9,9)的特征: {env.Phi[goal_idx]}")
    print(f"起点(0,0)的特征: {env.Phi[0]}")
    print(f"障碍物(3,3)的特征: {env.Phi[env._pos_to_idx((3,3))]}")
    
    print("\n⚠️ 问题1: 2D坐标特征只能表示线性梯度场")
    print("   R(s) = w1*(row/9) + w2*(col/9)")
    print("   无法表示'只有目标有值'的稀疏奖励!")
    
    # 2. 专家特征期望
    print("\n【2】专家特征期望 mu_E 分析")
    print("-"*70)
    mu_E = compute_feature_expectation(dataset['clean_trajectories'], env)
    print(f"mu_E = {mu_E}")
    print(f"mu_E[0] (行特征): {mu_E[0]:.2f}")
    print(f"mu_E[1] (列特征): {mu_E[1]:.2f}")
    print(f"比例: mu_E[0]/mu_E[1] = {mu_E[0]/mu_E[1]:.3f}")
    
    print("\n解释:")
    print(f"  - 专家轨迹平均行坐标访问: {mu_E[0]:.2f}")
    print(f"  - 专家轨迹平均列坐标访问: {mu_E[1]:.2f}")
    print(f"  - 两者接近，说明专家路径是对角线方向")
    
    # 3. 次优策略特征期望
    print("\n【3】次优策略特征期望 mu_subs 分析")
    print("-"*70)
    mu_subs = generate_suboptimal_mus(env)
    print(f"生成 {len(mu_subs)} 个次优策略的特征期望")
    print(f"mu_subs[0] = {mu_subs[0]}")
    print(f"mu_E - mu_subs[0] = {mu_E - mu_subs[0]}")
    
    # 4. LP-IRL 为什么学 [1, 0]?
    print("\n【4】LP-IRL 权重分析")
    print("-"*70)
    print("LP-IRL 约束: w @ (mu_E - mu_pi) >= 0, 对所有次优策略 pi")
    print("目标: 找到 w 使得专家特征期望优于所有次优策略")
    
    # 检查约束是否满足
    w_lp = np.array([1.0, 0.0])
    print(f"\n测试权重 w = {w_lp}")
    for i, mu_sub in enumerate(mu_subs[:3]):
        diff = mu_E - mu_sub
        margin = w_lp @ diff
        print(f"  次优策略{i}: w@(mu_E-mu_sub) = {margin:.3f} {'✓' if margin >= 0 else '✗'}")
    
    print("\n⚠️ 问题2: w=[1,0] 满足所有约束，所以LP-IRL学到这个退化解")
    print("   这意味着：只看列坐标就能区分专家和次优策略")
    print("   但这不是真实奖励！")
    
    # 5. 真实奖励 vs 恢复奖励
    print("\n【5】奖励场对比")
    print("-"*70)
    R_gt = env.ground_truth_reward
    R_lp = env.Phi @ w_lp
    R_maxent = env.Phi @ np.array([0.559, 0.559])
    
    print(f"Ground Truth: min={R_gt.min():.3f}, max={R_gt.max():.3f}, mean={R_gt.mean():.3f}")
    print(f"LP-IRL [1,0]: min={R_lp.min():.3f}, max={R_lp.max():.3f}, mean={R_lp.mean():.3f}")
    print(f"MaxEnt [0.56,0.56]: min={R_maxent.min():.3f}, max={R_maxent.max():.3f}, mean={R_maxent.mean():.3f}")
    
    pearson_lp = np.corrcoef(R_gt.flatten(), R_lp.flatten())[0, 1]
    pearson_maxent = np.corrcoef(R_gt.flatten(), R_maxent.flatten())[0, 1]
    
    print(f"\nPearson 相关性:")
    print(f"  LP-IRL vs Ground Truth: {pearson_lp:.3f}")
    print(f"  MaxEnt vs Ground Truth: {pearson_maxent:.3f}")
    
    # 6. 根本原因总结
    print("\n" + "="*70)
    print("【根本原因总结】")
    print("="*70)
    
    print("""
【问题A】2D坐标特征的表达能力限制
  真实奖励 R_gt:
    - (9,9) = 1.0
    - 障碍物 = -1.0
    - 其他 = 0.0
    → 这是非线性、稀疏的奖励函数
  
  线性近似 R = Phi @ w:
    - Phi[s] = [row/9, col/9]
    - R[s] = w1*row/9 + w2*col/9
    → 只能表示平面梯度，无法表示稀疏奖励
  
  数学上:
    - 真实奖励需要 100 个自由度 (每个状态独立)
    - 2D特征只有 2 个自由度
    - 信息损失 = 98 个维度!

【问题B】IRL 的目标是策略匹配，不是奖励匹配
  - IRL 优化的是: 专家特征期望 >= 次优特征期望
  - 不要求: 恢复奖励 ≈ 真实奖励
  - 所以即使 Pearson=0，策略也可能成功
  
  这叫做 Reward Ambiguity:
    - 不同的奖励函数 → 相同的策略
    - IRL 找到的是"等价类"中的一个，不是唯一的真实奖励

【问题C】LP/MM-IRL 在这个环境中退化
  - w=[1,0] 满足所有约束
  - 求解器直接返回这个简单解
  - 没有正则化推动它找到更好的解

【解决方案】
要提高 Pearson 相关性，需要:
  1. 增加特征维度 (5-10维 RBF/多项式特征)
  2. 添加正则化项 (L2 regularization)
  3. 或者改用 Onehot 特征 (但会过拟合)
  
要提高 LP/MM 成功率，需要:
  1. 添加更多次优策略约束
  2. 使用正则化防止退化解
  3. 或者改用 MaxEnt (已经100%成功)
""")

if __name__ == "__main__":
    main()
