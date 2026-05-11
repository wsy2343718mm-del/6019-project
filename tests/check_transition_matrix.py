"""
检查转移矩阵是否正确
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
    print("检查转移矩阵 P[s, a, s']")
    print("="*70)
    
    # 检查状态0 (0,0) 的动作1 (DOWN)
    s = 0
    a = 1  # DOWN
    
    print(f"\n状态 {s} = (0,0), 动作 {a} = DOWN")
    print(f"转移概率 P[{s}, {a}, :] 的非零项:")
    
    probs = env.P[s, a]
    nonzero = np.where(probs > 0.001)[0]
    
    for ns in nonzero:
        pos = env._idx_to_pos(ns)
        print(f"  {ns:3d} -> {pos}: {probs[ns]:.4f}")
    
    print(f"\n概率总和: {probs.sum():.4f}")
    
    # 手动采样测试
    print("\n" + "="*70)
    print("手动采样测试")
    print("="*70)
    
    np.random.seed(42)
    for i in range(10):
        sampled = np.random.choice(env.n_states, p=probs)
        print(f"采样{i+1}: 状态{sampled} = {env._idx_to_pos(sampled)}")
    
    # 检查 env.step() 使用的随机数生成器
    print("\n" + "="*70)
    print("检查 env.step() 使用的 np_random")
    print("="*70)
    
    state, _ = env.reset(seed=42, options={'fixed_start': (0,0)})
    print(f"重置后状态: {state}")
    print(f"env.np_random: {env.np_random}")
    
    # 直接调用 step
    action = 1
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"执行动作{action}后: {next_state} = {env._idx_to_pos(next_state)}")
    
    # 用同样的随机数种子手动采样
    env2, _ = env.reset(seed=42, options={'fixed_start': (0,0)})
    print(f"\n用 env.np_random 采样:")
    for i in range(5):
        sampled = env.np_random.choice(env.n_states, p=env.P[state, action])
        print(f"  采样{i+1}: {sampled} = {env._idx_to_pos(sampled)}")

if __name__ == "__main__":
    main()
