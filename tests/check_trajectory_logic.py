"""
检查轨迹生成逻辑
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
    
    policy, values = env.compute_optimal_policy()
    
    print("="*70)
    print("测试从不同起点开始的轨迹")
    print("="*70)
    
    # 测试100条轨迹，统计成功率
    success_count = 0
    total_count = 0
    
    for seed in range(100):
        state, _ = env.reset(seed=seed)
        start_pos = env._idx_to_pos(state)
        
        traj_len = 0
        done = False
        while not done:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            traj_len += 1
            done = terminated or truncated
        
        final_pos = env._idx_to_pos(state)
        total_count += 1
        
        if final_pos == env.goal_pos:
            success_count += 1
        else:
            if success_count < 5:  # 只打印前几个失败的
                print(f"Seed {seed:2d}: {start_pos} -> {final_pos} (失败)")
    
    print(f"\n成功率: {success_count}/{total_count} = {success_count/total_count*100:.1f}%")
    
    # 检查为什么可视化中成功率是0
    print("\n" + "="*70)
    print("检查 generate_expert.py 中的轨迹生成逻辑")
    print("="*70)
    
    print("\n查看 generate_trajectories 函数:")
    print("问题可能在于: traj.append((state, action, reward, next_state, done))")
    print("然后检查 final_state = traj[-1][0] 还是 traj[-1][3]")

if __name__ == "__main__":
    main()
