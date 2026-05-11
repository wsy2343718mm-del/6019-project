"""
检查新障碍物布局
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv

def main():
    obstacles = [
        (0, 5), (1, 5),                    # 上方纵向阻挡
        (2, 0), (2, 1),                    # 左侧阻挡
        (3, 3), (3, 5), (3, 6), (3, 8),   # 中间散点
        (4, 3), (4, 8),                    # 中间散点
        (5, 2), (5, 3), (5, 8), (5, 9),   # 中间阻挡
        (7, 1), (7, 7),                    # 下方散点
        (8, 1), (8, 4), (8, 5), (8, 6),   # 下方阻挡
        (9, 1), (9, 6),                    # 底部阻挡
    ]
    
    print(f"障碍物列表（共{len(obstacles)}个）:")
    for obs in obstacles:
        print(f"  {obs}")
    
    # 检查是否有重复
    unique_obs = set(obstacles)
    print(f"\n去重后: {len(unique_obs)}个")
    if len(obstacles) != len(unique_obs):
        print("⚠️ 有重复！")
        from collections import Counter
        counts = Counter(obstacles)
        for obs, count in counts.items():
            if count > 1:
                print(f"  {obs}: {count}次")
    
    env = GridWorldEnv(
        size=10, 
        goal_pos=(9,9), 
        obstacles=obstacles, 
        stochasticity=0.1,
        feature_mode='coords'
    )
    
    print(f"\n环境中实际障碍物数量: {len(env.obstacles)}")
    print(f"环境中障碍物位置: {sorted(env.obstacles)}")
    
    # 打印网格
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
            elif (r, c) == (0, 0):
                print("  S", end="")
            else:
                print("  ·", end="")
        print()

if __name__ == "__main__":
    main()
