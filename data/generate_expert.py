import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

# 设置标准输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 确保能导入项目根目录的 env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv

def generate_trajectories(env, n_episodes, policy=None, action_noise=0.0, seed=42):
    """生成专家轨迹（支持动作噪声注入）"""
    np.random.seed(seed)
    if policy is None:
        policy, _ = env.compute_optimal_policy()

    trajectories = []
    for ep_idx in tqdm(range(n_episodes), desc="🔄 Rolling out expert trajectories"):
        traj = []
        # ✅ 关键修复：每条轨迹使用不同的 seed，确保起点多样化
        state, _ = env.reset(seed=seed + ep_idx)
        done = False
        while not done:
            # 专家动作
            action = policy[state]
            # 注入动作噪声（模拟专家不完美/演示噪声）
            if np.random.rand() < action_noise:
                action = np.random.randint(env.n_actions)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            traj.append((state, action, reward, next_state, done))
            state = next_state
        trajectories.append(traj)
    return trajectories

def compute_discounted_return(traj, gamma=0.9):
    """计算单条轨迹的折扣累积回报"""
    return sum(r * (gamma ** t) for t, (_, _, r, _, _) in enumerate(traj))

def generate_preference_pairs(trajectories, n_pairs, pref_noise=0.0, seed=42):
    """
    基于 Bradley-Terry 模型生成偏好对
    P(τ_i ≻ τ_j) = σ(R(τ_i) - R(τ_j))，支持偏好标签噪声
    
    改进：优先选择有回报差异的轨迹对，确保偏好数据有区分度
    """
    np.random.seed(seed)
    returns = np.array([compute_discounted_return(traj) for traj in trajectories])
    
    # 计算轨迹是否到达目标（用于筛选有意义的偏好对）
    goal_reached = np.array([1.0 if r > 0.5 else 0.0 for r in returns])
    
    pairs = []
    labels = []  # 1 表示 traj_i 优于 traj_j，0 表示相反
    
    attempts = 0
    max_attempts = n_pairs * 10  # 防止无限循环
    
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = np.random.choice(len(trajectories), 2, replace=False)
        
        # 计算回报差异
        return_diff = returns[i] - returns[j]
        
        # 如果回报差异太小（绝对值 < 0.01），跳过
        # 这确保偏好对有区分度
        if abs(return_diff) < 0.01:
            continue
        
        # Bradley-Terry 概率（放大差异以提高区分度）
        prob_i_better = 1.0 / (1.0 + np.exp(-return_diff * 5))  # 放大 5 倍
        
        # 采样偏好标签
        is_i_better = np.random.rand() < prob_i_better
        # 注入偏好噪声（人类标注错误/不一致）
        if np.random.rand() < pref_noise:
            is_i_better = not is_i_better
            
        pairs.append((i, j))
        labels.append(1 if is_i_better else 0)
    
    # 如果还是不够，用随机对补充
    while len(pairs) < n_pairs:
        i, j = np.random.choice(len(trajectories), 2, replace=False)
        return_diff = returns[i] - returns[j]
        prob_i_better = 1.0 / (1.0 + np.exp(-return_diff * 5))
        is_i_better = np.random.rand() < prob_i_better
        if np.random.rand() < pref_noise:
            is_i_better = not is_i_better
        pairs.append((i, j))
        labels.append(1 if is_i_better else 0)
        
    return pairs, labels, returns

def main():
    # 新的障碍物布局（根据 Excel 设计）
    obstacles = [
        (0, 5),                              # 行1: F列
        (1, 5),                              # 行2: F列
        (2, 0), (2, 1),                      # 行3: A,B列
        (3, 3), (3, 5), (3, 6), (3, 8),     # 行4: D,F,G,I列
        (4, 3), (4, 8),                      # 行5: D,I列
        (5, 2), (5, 3), (5, 8), (5, 9),     # 行6: C,D,I,J列
        # 行7: 无障碍物
        (7, 1), (7, 7),                      # 行8: B,H列
        (8, 1), (8, 4), (8, 5), (8, 6),     # 行9: B,E,F,G列
        (9, 1), (9, 6),                      # 行10: B,G列
    ]
    
    print("🌍 初始化 GridWorld 环境 (10x10, 含障碍物)...")
    print(f"   障碍物数量: {len(obstacles)}")
    print(f"   障碍物位置: {obstacles}")
    env = GridWorldEnv(
        size=10,
        goal_pos=(9,9),
        obstacles=obstacles,
        stochasticity=0.1,
        feature_mode="rbf"  # 使用RBF特征，能更好表示稀疏奖励
    )
    
    # 1. 生成干净专家轨迹 (500 条)
    print("📦 生成专家轨迹...")
    clean_trajs = generate_trajectories(env, n_episodes=500, action_noise=0.0)
    
    # 2. 生成含动作噪声的轨迹 (用于消融实验)
    print("🔊 生成含动作噪声的轨迹 (flip_prob=0.15)...")
    noisy_trajs = generate_trajectories(env, n_episodes=500, action_noise=0.15)
    
    # 3. 生成 Bradley-Terry 偏好对（混合干净+噪声轨迹，确保有区分度）
    print("⚖️ 生成偏好对 (混合 clean + noisy 轨迹)...")
    all_trajs = clean_trajs + noisy_trajs  # 混合两种轨迹
    pairs, labels, returns = generate_preference_pairs(all_trajs, n_pairs=300, pref_noise=0.05)
    
    # 4. 保存数据 (下游 IRL 算法直接读取)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(data_dir, "expert_dataset.pkl")
    dataset = {
        "clean_trajectories": clean_trajs,
        "noisy_trajectories": noisy_trajs,
        "preference_pairs": pairs,
        "preference_labels": labels,
        "trajectory_returns": returns,
        "env_config": {
            "size": env.size,
            "goal": env.goal_pos,
            "obstacles": list(env.obstacles),
            "stochasticity": env.stochasticity,
            "gamma": env.gamma,
            "feature_dim": env.Phi.shape[1],
            "feature_mode": env.feature_mode
        }
    }
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f"\n✅ 数据已保存至: {save_path}")
    print(f"📊 统计信息:")
    print(f"   - 干净轨迹数: {len(clean_trajs)}")
    print(f"   - 噪声轨迹数: {len(noisy_trajs)}")
    print(f"   - 偏好对数: {len(pairs)}")
    print(f"   - 平均轨迹回报: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"   - 偏好正例比例: {np.mean(labels):.2%}")

if __name__ == "__main__":
    main()