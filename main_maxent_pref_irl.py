"""
MaxEnt-IRL 和 Preference-BT 算法执行脚本
功能：运行算法并保存结果，不输出详细评估（统一由 main_comparison.py 输出）
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation
from irl.maxent_irl import maxent_irl
from irl.preference_irl import preference_irl

def main():
    print("=" * 60)
    print("【MaxEnt-IRL / Preference-BT】算法执行")
    print("=" * 60)
    
    # 加载数据
    with open(os.path.join("data", "expert_dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)
    cfg = dataset["env_config"]
    
    env = GridWorldEnv(
        size=cfg["size"], goal_pos=cfg["goal"], obstacles=[], 
        stochasticity=cfg["stochasticity"], feature_mode="coords", gamma=cfg["gamma"]
    )
    
    # 计算特征期望
    mu_E = compute_feature_expectation(dataset["clean_trajectories"], env)
    
    # 运行 MaxEnt-IRL
    print("🔹 运行 MaxEnt-IRL...")
    w_maxent = maxent_irl(mu_E, env, n_iters=300, lr=0.2)
    print(f"   权重: {w_maxent.round(3)}")
    
    # 运行 Preference-BT
    print("🔹 运行 Preference-BT...")
    all_trajs = dataset["clean_trajectories"] + dataset["noisy_trajectories"]
    w_pref = preference_irl(
        dataset["preference_pairs"], dataset["preference_labels"],
        all_trajs, env, n_iters=1000, lr=1.0
    )
    print(f"   权重: {w_pref.round(3)}")
    
    # 保存结果
    R_maxent = env.Phi @ w_maxent
    R_pref = env.Phi @ w_pref
    results = {
        "w_maxent": w_maxent, "w_pref": w_pref,
        "R_maxent": R_maxent, "R_pref": R_pref
    }
    
    res_path = os.path.join("data", "maxent_pref_results.pkl")
    with open(res_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n✅ 结果已保存至: {res_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()