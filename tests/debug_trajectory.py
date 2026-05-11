"""
详细测试单条轨迹
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
    
    # 测试有噪声的环境
    env = GridWorldEnv(
        size=10, 
        goal_pos=(9,9), 
        obstacles=obstacles, 
        stochasticity=0.1,
        feature_mode='coords'
    )
    
    policy, values = env.compute_optimal_policy()
    
    print("="*70)
    print("测试有噪声环境下的轨迹 (stochasticity=0.1)")
    print("="*70)
    
    # 固定seed=42，从(0,0)开始
    np.random.seed(42)
    state, info = env.reset(seed=42, options={'fixed_start': (0,0)})
    
    print(f"\n起点: {env._idx_to_pos(state)}, state_id={state}")
    print(f"检查 _current_state: {env._current_state}")
    
    print("\n详细轨迹:")
    for step in range(20):
        action = policy[state]
        
        # 查看转移概率
        probs = env.P[state, action]
        top_probs = np.argsort(probs)[-5:][::-1]
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_pos = env._idx_to_pos(next_state)
        
        print(f"\n步{step+1:2d}:")
        print(f"  当前: {env._idx_to_pos(state)} (id={state})")
        print(f"  动作: {action}")
        print(f"  实际到达: {next_pos} (id={next_state})")
        print(f"  奖励: {reward}, 终止: {terminated}")
        print(f"  Top 5 转移概率:")
        for s in top_probs:
            if probs[s] > 0.01:
                print(f"    {env._idx_to_pos(s)}: {probs[s]:.3f}")
        
        state = next_state
        if terminated or truncated:
            is_goal = next_pos == env.goal_pos
            print(f"\n  >>> 终止于 {next_pos}, 目标={is_goal}")
            break

if __name__ == "__main__":
    main()
