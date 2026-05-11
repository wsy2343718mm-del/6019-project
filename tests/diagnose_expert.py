"""
诊断专家策略和障碍物布局问题
检查障碍物是否阻断了通往目标的路径
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv

def main():
    obstacles = [
        (3, 3), (3, 4), (3, 5), (3, 6),  # 横向墙壁
        (5, 3), (5, 4), (5, 5), (5, 6),  # 横向墙壁
        (7, 2), (7, 3), (7, 4),          # 小块障碍
        (2, 7), (4, 7), (6, 7),          # 纵向障碍
    ]
    
    env = GridWorldEnv(
        size=10, 
        goal_pos=(9,9), 
        obstacles=obstacles, 
        stochasticity=0.1
    )
    
    print("="*70)
    print("🔍 障碍物布局分析")
    print("="*70)
    
    # 打印网格可视化
    print("\n📍 网格布局 (10x10):")
    print("   ", end="")
    for c in range(10):
        print(f"{c:3d}", end="")
    print()
    
    for r in range(10):
        print(f"{r:2d} ", end="")
        for c in range(10):
            if (r, c) in env.obstacles:
                print("  █", end="")
            elif (r, c) == env.goal_pos:
                print("  ★", end="")
            else:
                print("  ·", end="")
        print()
    
    # 值迭代分析
    policy, values = env.compute_optimal_policy()
    
    print("\n" + "="*70)
    print("📊 状态价值函数分析")
    print("="*70)
    
    print("\n状态价值矩阵:")
    print("   ", end="")
    for c in range(10):
        print(f"{c:6d}", end="")
    print()
    
    for r in range(10):
        print(f"{r:2d} ", end="")
        for c in range(10):
            state_idx = env._pos_to_idx((r, c))
            if (r, c) in env.obstacles:
                print(f"  XXXX", end="")
            elif (r, c) == env.goal_pos:
                print(f" {values[state_idx]:6.2f}", end="")
            else:
                print(f" {values[state_idx]:6.2f}", end="")
        print()
    
    # 策略分析
    action_names = ['↑上', '↓下', '←左', '→右']
    print("\n" + "="*70)
    print("🎯 专家策略分析")
    print("="*70)
    
    print("\n策略矩阵 (箭头表示):")
    print("   ", end="")
    for c in range(10):
        print(f"{c:3c}", end="")
    print()
    
    arrows = ['↑', '↓', '←', '→']
    for r in range(10):
        print(f"{r:2d} ", end="")
        for c in range(10):
            state_idx = env._pos_to_idx((r, c))
            if (r, c) in env.obstacles:
                print("  █", end="")
            elif (r, c) == env.goal_pos:
                print("  ★", end="")
            else:
                action = policy[state_idx]
                print(f"  {arrows[action]}", end="")
        print()
    
    # 关键问题诊断
    print("\n" + "="*70)
    print("❌ 问题诊断")
    print("="*70)
    
    # 检查障碍物是否形成墙壁阻断
    print("\n🚧 检查障碍物布局:")
    print(f"   第3行障碍: (3,3)-(3,6) → 阻断行3的列3-6")
    print(f"   第5行障碍: (5,3)-(5,6) → 阻断行5的列3-6")
    print(f"   第7行障碍: (7,2)-(7,4) → 阻断行7的列2-4")
    print(f"   第7列障碍: (2,7),(4,7),(6,7) → 分散阻断")
    
    # 检查从(0,0)到(9,9)的路径
    print("\n🛤️  路径可达性分析:")
    print("   从(0,0)到(9,9)需要:")
    print("   - 向下移动9步")
    print("   - 向右移动9步")
    print("   - 必须经过第3、5、7行")
    print("   - 必须经过第3、5、7列")
    
    # 模拟从(0,0)开始的专家轨迹
    print("\n🎮 模拟专家从(0,0)开始的轨迹:")
    state, _ = env.reset(seed=0, options={'fixed_start': (0, 0)})
    print(f"   起点: (0,0), 状态值: {values[state]:.2f}")
    
    for step in range(20):
        action = policy[state]
        action_name = action_names[action]
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_pos = env._idx_to_pos(next_state)
        
        print(f"   步{step+1:2d}: {env._idx_to_pos(state)} --{action_name}--> {next_pos} "
              f"(奖励: {reward:.2f}, 终止: {terminated})")
        
        state = next_state
        if terminated or truncated:
            final_pos = env._idx_to_pos(state)
            if final_pos == env.goal_pos:
                print(f"   ✅ 到达目标!")
            else:
                print(f"   ❌ 未到达目标，终止于: {final_pos}")
            break
    
    # 统计所有起点的可达性
    print("\n" + "="*70)
    print("📈 全面可达性测试")
    print("="*70)
    
    success_count = 0
    total_tests = 0
    
    for seed in range(100):
        state, _ = env.reset(seed=seed)
        start_pos = env._idx_to_pos(state)
        
        done = False
        while not done:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
        
        final_pos = env._idx_to_pos(state)
        if final_pos == env.goal_pos:
            success_count += 1
        total_tests += 1
    
    print(f"\n🎯 随机起点成功率: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    # 分析为什么专家轨迹这么短
    print("\n" + "="*70)
    print("🔬 轨迹长度分析")
    print("="*70)
    print("\n可能的问题:")
    print("1. 障碍物布局可能阻断了所有通往目标的路径")
    print("2. 专家策略被障碍物困住，提前终止")
    print("3. 障碍物被设置为吸收态，碰到就结束")
    print("\n检查环境代码中的终止条件...")

if __name__ == "__main__":
    main()
