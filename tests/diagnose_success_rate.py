"""
诊断为什么成功率是100%，以及检查评估是否有问题
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv
import numpy as np

def main():
    obstacles = [
        (3, 3), (3, 4), (3, 5), (3, 6),
        (5, 3), (5, 4), (5, 5), (5, 6),
        (7, 2), (7, 3), (7, 4),
        (2, 7), (4, 7), (6, 7),
    ]
    
    env = GridWorldEnv(
        size=10, 
        goal_pos=(9,9), 
        obstacles=obstacles, 
        stochasticity=0.1,
        feature_mode='coords'
    )
    
    print("="*70)
    print("诊断问题1：检查特征矩阵 Phi")
    print("="*70)
    print(f"\nPhi shape: {env.Phi.shape}")
    print(f"Phi 前10行:\n{env.Phi[:10]}")
    print(f"\n问题：Phi只有2维（行、列的归一化坐标)")
    print(f"障碍物格子的Phi: {env.Phi[env._pos_to_idx((3,3))]}")
    print(f"目标格子的Phi: {env.Phi[env._pos_to_idx((9,9))]}")
    
    # 测试不同的权重
    print("\n" + "="*70)
    print("诊断问题2：检查不同权重下的策略")
    print("="*70)
    
    # LP权重: [1, 0] - 只看重列
    w_lp = np.array([1.0, 0.0])
    R_lp = env.Phi @ w_lp
    policy_lp, V_lp = env.compute_optimal_policy(reward_vec=R_lp)
    
    print(f"\nLP权重: {w_lp}")
    print(f"R_lp 最小值: {R_lp.min():.3f}, 最大值: {R_lp.max():.3f}")
    print(f"障碍物(3,3)的奖励: {R_lp[env._pos_to_idx((3,3))]:.3f}")
    print(f"目标(9,9)的奖励: {R_lp[env._pos_to_idx((9,9))]:.3f}")
    print(f"起点(0,0)的奖励: {R_lp[env._pos_to_idx((0,0))]:.3f}")
    
    # 检查从(0,0)的策略
    print(f"\n策略检查 (从(0,0)开始):")
    state, _ = env.reset(options={'fixed_start': (0,0)})
    for step in range(25):
        action = policy_lp[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_names = ['↑UP', '↓DOWN', '←LEFT', '→RIGHT']
        print(f"  步{step+1:2d}: {env._idx_to_pos(state)} --{action_names[action]:6s}--> {env._idx_to_pos(next_state)}")
        state = next_state
        if terminated or truncated:
            is_goal = env._idx_to_pos(state) == env.goal_pos
            print(f"  >>> 终止于 {env._idx_to_pos(state)}, 目标={is_goal}")
            break
    
    # 评估100次
    print(f"\n评估LP-IRL策略 100 次:")
    success_count = 0
    for seed in range(100):
        state, _ = env.reset(seed=seed, options={'fixed_start': (0,0)})
        done = False
        steps = 0
        while not done and steps < 150:
            action = policy_lp[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            steps += 1
        
        if env._idx_to_pos(state) == env.goal_pos:
            success_count += 1
    
    print(f"成功率: {success_count}/100 = {success_count}%")
    
    print("\n" + "="*70)
    print("诊断问题3：检查 Ground Truth 奖励")
    print("="*70)
    R_gt = env.ground_truth_reward
    print(f"\nR_gt 最小值: {R_gt.min():.3f}")
    print(f"R_gt 最大值: {R_gt.max():.3f}")
    print(f"障碍物(3,3)的奖励: {R_gt[env._pos_to_idx((3,3))]:.3f}")
    print(f"目标(9,9)的奖励: {R_gt[env._pos_to_idx((9,9))]:.3f}")
    print(f"普通格子(0,0)的奖励: {R_gt[env._pos_to_idx((0,0))]:.3f}")
    
    # 检查值迭代
    policy_gt, V_gt = env.compute_optimal_policy()
    print(f"\nGround Truth 策略检查 (从(0,0)开始):")
    state, _ = env.reset(options={'fixed_start': (0,0)})
    for step in range(25):
        action = policy_gt[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_names = ['↑UP', '↓DOWN', '←LEFT', '→RIGHT']
        print(f"  步{step+2}: {env._idx_to_pos(state)} --{action_names[action]:6s}--> {env._idx_to_pos(next_state)} (V={V_gt[state]:.3f})")
        state = next_state
        if terminated or truncated:
            is_goal = env._idx_to_pos(state) == env.goal_pos
            print(f"  >>> 终止于 {env._idx_to_pos(state)}, 目标={is_goal}, V={V_gt[state]:.3f}")
            break

if __name__ == "__main__":
    main()
