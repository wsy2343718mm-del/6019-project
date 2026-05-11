"""
LP-IRL 和 MM-IRL 算法执行脚本
功能：运行算法并保存结果，不输出详细评估（统一由 main_comparison.py 输出）
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
from irl.lp_irl import solve_lp_irl
from irl.mm_irl import solve_mm_irl

def main():
    print("=" * 60)
    print("【LP-IRL / MM-IRL】算法执行")
    print("=" * 60)
    
    # 加载数据
    data_path = os.path.join("data", "expert_dataset.pkl")
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
        
    cfg = dataset["env_config"]
    env = GridWorldEnv(
        size=cfg["size"], goal_pos=cfg["goal"], obstacles=[], 
        stochasticity=cfg["stochasticity"], feature_mode="coords", gamma=cfg["gamma"]
    )
    
    # 计算专家特征期望
    mu_E = compute_feature_expectation(dataset["clean_trajectories"], env)
    mu_subs = generate_suboptimal_mus(env)
    
    # 求解 IRL
    n_feat = env.Phi.shape[1]
    print("🔹 运行 LP-IRL...")
    w_lp = solve_lp_irl(mu_E, mu_subs, n_feat)
    print(f"   权重: {w_lp.round(3)}")
    
    print("🔹 运行 MM-IRL...")
    w_mm, margin = solve_mm_irl(mu_E, mu_subs, n_feat)
    print(f"   权重: {w_mm.round(3)}, 边距: {margin:.4f}")
    
    # 保存结果
    R_lp = env.Phi @ w_lp
    R_mm = env.Phi @ w_mm
    results = {
        "w_lp": w_lp, "w_mm": w_mm, "margin": margin,
        "R_lp": R_lp, "R_mm": R_mm, "mu_E": mu_E
    }
    
    res_path = os.path.join("data", "irl_results.pkl")
    with open(res_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n✅ 结果已保存至: {res_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()