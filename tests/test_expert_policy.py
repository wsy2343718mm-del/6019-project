"""
测试专家策略质量
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
        stochasticity=0.0,  # 先测试确定性环境
        feature_mode='coords'
    )
    
    policy, values = env.compute_optimal_policy()
    
    print("="*70)
    print("测试专家策略（确定性环境，stochasticity=0.0）")
    print("="*70)
    
    # 测试从不同起点开始
    test_starts = [(0,0), (0,1), (1,0), (1,1), (9,0), (0,9), (8,8)]
    
    for start_pos in test_starts:
        print(f"\n--- 从 {start_pos} 开始 ---")
        state, _ = env.reset(options={'fixed_start': start_pos})
        
        traj_states = [start_pos]
        for step in range(30):
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_pos = env._idx_to_pos(next_state)
            traj_states.append(next_pos)
            
            state = next_state
            if terminated or truncated:
                is_goal = next_pos == env.goal_pos
                print(f"  步{step+1}: 到达 {next_pos}, 终止={terminated}, 目标={is_goal}")
                print(f"  轨迹长度: {len(traj_states)-1}")
                print(f"  路径: {traj_states[:10]}{'...' if len(traj_states) > 10 else ''}")
                break
    
    # 重点分析 (0,0)
    print("\n" + "="*70)
    print("详细分析从 (0,0) 开始的轨迹")
    print("="*70)
    
    env_stoch = GridWorldEnv(
        size=10, 
        goal_pos=(9,9), 
        obstacles=obstacles, 
        stochasticity=0.1,  # 有噪声
        feature_mode='coords'
    )
    
    np.random.seed(42)
    for trial in range(5):
        print(f"\n试验 {trial+1}:")
        state, _ = env_stoch.reset(seed=42+trial, options={'fixed_start': (0,0)})
        
        traj_states = [(0,0)]
        for step in range(30):
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env_stoch.step(action)
            next_pos = env_stoch._idx_to_pos(next_state)
            traj_states.append(next_pos)
            
            state = next_state
            if terminated or truncated:
                is_goal = next_pos == env_stoch.goal_pos
                print(f"  步{step+1}: {traj_states[-2]} -> {next_pos}, 终止, 目标={is_goal}")
                print(f"  完整路径: {traj_states}")
                break

if __name__ == "__main__":
    main()
