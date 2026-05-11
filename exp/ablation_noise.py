import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from data.generate_expert import generate_trajectories
from irl.utils import compute_feature_expectation
from irl.maxent_irl import maxent_irl  # 仅测试 MaxEnt 作为代表，LP/MM 同理

def main():
    print("🌪️ 开始噪声消融实验 (Action Noise Ablation Study)...")
    env = GridWorldEnv(size=10, goal_pos=(9,9), obstacles=[], stochasticity=0.1)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = []

    for noise in noise_levels:
        print(f"\n📊 测试噪声水平: {noise}")
        # 1. 生成含噪专家数据
        noisy_trajs = generate_trajectories(env, n_episodes=300, action_noise=noise)
        
        # 2. 计算特征期望
        mu_noisy = compute_feature_expectation(noisy_trajs, env)
        
        # 3. 运行 MaxEnt-IRL
        try:
            w_rec = maxent_irl(mu_noisy, env, n_iters=200, lr=0.2)
            
            # 4. 评估策略 - 固定起点 (0,0)
            R_rec = env.Phi @ w_rec
            policy, _ = env.compute_optimal_policy(reward_vec=R_rec)
            successes = 0
            for _ in range(50):
                # ✅ 固定起点 (0,0)
                s, _ = env.reset(options={'fixed_start': (0, 0)})
                for _ in range(50):
                    ns, r, term, trunc, _ = env.step(policy[s])
                    if term or trunc: 
                        if env._idx_to_pos(ns) == env.goal_pos: successes += 1
                        break
                    s = ns
            acc = successes / 50
            results.append((noise, acc))
            print(f"✅ 噪声={noise} | 恢复权重: {w_rec.round(2)} | 成功率: {acc:.1%}")
            
        except Exception as e:
            print(f"❌ 噪声={noise} 求解失败: {e}")
            results.append((noise, 0.0))

    # 打印最终表格 (可复制到报告)
    print("\n📈 ===== 消融实验汇总表 =====")
    print("| Noise Level | Success Rate |")
    print("|-------------|--------------|")
    for noise, acc in results:
        print(f"| {noise}          | {acc:.2%}         |")

if __name__ == "__main__":
    main()