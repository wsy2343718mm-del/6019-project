"""
验证奖励场如何影响策略选择
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
        stochasticity=0.0,  # 确定性环境，便于观察
        feature_mode='coords'
    )
    
    print("="*70)
    print("验证奖励场如何影响策略")
    print("="*70)
    
    # LP权重 [1, 0]
    w_lp = np.array([1.0, 0.0])
    R_lp = env.Phi @ w_lp
    policy_lp, V_lp = env.compute_optimal_policy(reward_vec=R_lp)
    
    print("\n【LP-IRL 奖励场 (只看列坐标)】")
    print("权重: [1.0, 0.0]")
    print("\n从 (0,0) 开始的策略分析:")
    
    state, _ = env.reset(options={'fixed_start': (0,0)})
    action_names = ['↑上', '↓下', '←左', '→右']
    
    print(f"\n{'步':>3s} | {'位置':>7s} | {'动作':>5s} | {'即时R':>6s} | {'状态V':>7s} | 下一步位置 | 下一步R")
    print("-"*90)
    
    for step in range(25):
        pos = env._idx_to_pos(state)
        action = policy_lp[state]
        reward = R_lp[state]
        value = V_lp[state]
        
        # 查看每个动作的Q值
        next_states = []
        for a in range(4):
            ns_probs = env.P[state, a]
            ns = np.argmax(ns_probs)
            next_states.append((a, ns))
        
        next_state = next_states[action][1]
        next_pos = env._idx_to_pos(next_state)
        next_R = R_lp[next_state]
        
        print(f"{step+1:3d} | {str(pos):>7s} | {action_names[action]:>5s} | {reward:6.3f} | {value:7.3f} | {str(next_pos):>11s} | {next_R:.3f}")
        
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        
        if terminated or truncated:
            is_goal = env._idx_to_pos(state) == env.goal_pos
            print(f">>> 终止于 {env._idx_to_pos(state)}, 目标={is_goal}")
            break
    
    print("\n" + "="*70)
    print("【关键观察】")
    print("="*70)
    print("""
1. 即时奖励 R 从 0.0 逐渐增加到 1.0
2. 策略不是选"即时R最大"的格子，而是选"能最快到目标"的路径
3. 状态价值 V(s) 才是真正指导策略的指标
   - V(s) = 从s出发能获得的总折扣回报
   - V(s) 越靠近目标越大
4. 热力图显示的是 R(s)，但策略实际看的是 V(s)
""")
    
    # 对比 Ground Truth
    print("="*70)
    print("【对比：Ground Truth 奖励场】")
    print("="*70)
    
    R_gt = env.ground_truth_reward
    policy_gt, V_gt = env.compute_optimal_policy()
    
    state, _ = env.reset(options={'fixed_start': (0,0)})
    print(f"\n{'步':>3s} | {'位置':>7s} | {'动作':>5s} | {'即时R':>6s} | {'状态V':>7s}")
    print("-"*50)
    
    for step in range(25):
        pos = env._idx_to_pos(state)
        action = policy_gt[state]
        reward = R_gt[state]
        value = V_gt[state]
        
        print(f"{step+1:3d} | {str(pos):>7s} | {action_names[action]:>5s} | {reward:6.3f} | {value:7.3f}")
        
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        
        if terminated or truncated:
            is_goal = env._idx_to_pos(state) == env.goal_pos
            print(f">>> 终止于 {env._idx_to_pos(state)}, 目标={is_goal}, V={V_gt[state]:.3f}")
            break
    
    print("\n【Ground Truth 的特点】")
    print("- 只有终点R=1，其他都是R=0")
    print("- 但V(s)仍然从起点到终点递增")
    print("- 策略看的是V(s)，不是R(s)")

if __name__ == "__main__":
    main()
