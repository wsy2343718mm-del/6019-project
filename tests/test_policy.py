import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from policy.train_policy import PolicyTrainer
from env.gridworld import GridWorldEnv

def main():
    # 1. 重建环境（必须与生成数据时一致）
    env = GridWorldEnv(size=10, goal_pos=(9,9), obstacles=[], 
                       stochasticity=0.1, feature_mode="coords")
    
    # 2. 读取之前 LP-IRL 恢复的权重作为测试
    with open("../data/irl_results.pkl", "rb") as f:
        data = pickle.load(f)
    w_test = data["w_lp"]
    
    print(f"🔧 加载权重: {w_test}")
    
    # 3. 初始化训练器并训练
    trainer = PolicyTrainer(env, w_test)
    trainer.train_value_iteration(gamma=env.gamma)
    
    # 4. 评估并打印结果
    acc = trainer.evaluate(n_trials=100)
    print(f"✅ 策略训练完成 | 测试成功率: {acc:.1%}")
    
    # 5. 简单展示策略在起点的动作 (0:上, 1:下, 2:左, 3:右)
    start_s = env.reset()[0]
    print(f"📍 起点状态 {start_s} 的最优动作: {trainer.policy[start_s]}")

if __name__ == "__main__":
    main()