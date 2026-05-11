import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from env.gridworld import GridWorldEnv
from utils.viz import save_comparison_plot

from policy.train_policy import PolicyTrainer

def main():
    print("🎨 生成奖励场对比热力图...")
    
    # 1. 重建环境（必须与生成数据时的配置一致）
    env = GridWorldEnv(size=10, goal_pos=(9,9), obstacles=[], 
                       stochasticity=0.1, feature_mode="coords")
    
    # 2. 加载 LP-IRL / MM-IRL 结果
    with open(os.path.join("data", "irl_results.pkl"), "rb") as f:
        results_s2 = pickle.load(f)
    w_lp = results_s2["w_lp"]
    w_mm = results_s2["w_mm"]
    
    # 3. 加载 MaxEnt-IRL / Preference-BT 结果
    with open(os.path.join("data", "maxent_pref_results.pkl"), "rb") as f:
        results_s3 = pickle.load(f)
    w_maxent = results_s3["MaxEnt-IRL"]["w"]
    w_pref = results_s3["Preference-BT"]["w"]
    
    # 4. 生成对比图
    print(f"📊 权重汇总:")
    print(f"   LP-IRL:       {w_lp.round(3)}")
    print(f"   MM-IRL:       {w_mm.round(3)}")
    print(f"   MaxEnt-IRL:   {w_maxent.round(3)}")
    print(f"   Preference-BT:{w_pref.round(3)}")
    
    save_comparison_plot(env, w_lp, w_mm, w_maxent, w_pref)
    print("✅ 可视化完成！")

if __name__ == "__main__":
    main()