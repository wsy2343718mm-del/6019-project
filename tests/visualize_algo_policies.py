"""
可视化各个IRL算法学到的策略路径
对比: Ground Truth, LP-IRL, MM-IRL, MaxEnt-IRL, Preference-BT
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from env.gridworld import GridWorldEnv

def main():
    # 加载数据
    with open('data/expert_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    cfg = dataset['env_config']
    obstacles = cfg.get('obstacles', [])
    
    env = GridWorldEnv(
        size=cfg['size'], 
        goal_pos=cfg['goal'], 
        obstacles=obstacles,
        stochasticity=cfg['stochasticity'], 
        feature_mode="rbf"  # 使用RBF特征
    )
    
    # 加载对比结果
    with open('data/comparison_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    main_results = results['main_results']
    
    print("="*70)
    print("可视化各IRL算法策略路径")
    print("="*70)
    
    # 重建各算法的奖励和策略（添加缩放）
    R_gt = env.ground_truth_reward
    policy_gt, _ = env.compute_optimal_policy(reward_vec=R_gt)
    
    w_lp = main_results['LP-IRL']['weight']
    R_lp = env.Phi @ w_lp
    policy_lp, _ = env.compute_optimal_policy(reward_vec=R_lp * 5.0)
    
    w_mm = main_results['MM-IRL']['weight']
    R_mm = env.Phi @ w_mm
    policy_mm, _ = env.compute_optimal_policy(reward_vec=R_mm * 5.0)
    
    w_maxent = main_results['MaxEnt-IRL']['weight']
    R_maxent = env.Phi @ w_maxent
    policy_maxent, _ = env.compute_optimal_policy(reward_vec=R_maxent * 5.0)
    
    w_pref = main_results['Preference-BT']['weight']
    R_pref = env.Phi @ w_pref
    policy_pref, _ = env.compute_optimal_policy(reward_vec=R_pref * 5.0)
    
    # 从固定起点(0,0)运行各策略
    policies = {
        'Ground Truth': policy_gt,
        'LP-IRL': policy_lp,
        'MM-IRL': policy_mm,
        'MaxEnt-IRL': policy_maxent,
        'Preference-BT': policy_pref
    }
    
    trajectories = {}
    for name, policy in policies.items():
        traj = []
        state, _ = env.reset(options={'fixed_start': (0, 0)})
        done = False
        steps = 0
        while not done and steps < 50:
            traj.append(state)
            action = policy[state]
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            steps += 1
        traj.append(state)  # 添加终点
        trajectories[name] = traj
        
        final_pos = env._idx_to_pos(state)
        is_goal = final_pos == env.goal_pos
        print(f"\n{name}:")
        print(f"  路径: {[env._idx_to_pos(s) for s in traj[:15]]}{'...' if len(traj) > 15 else ''}")
        print(f"  长度: {len(traj)-1} 步, 终点: {final_pos}, 成功: {is_goal}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {
        'Ground Truth': '#2ecc71',
        'LP-IRL': '#3498db',
        'MM-IRL': '#9b59b6',
        'MaxEnt-IRL': '#f39c12',
        'Preference-BT': '#e74c3c'
    }
    
    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = axes[idx]
        positions = [env._idx_to_pos(s) for s in traj]
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        
        # 绘制网格
        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(-0.5, env.size - 0.5)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 绘制障碍物
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                facecolor='black', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
        
        # 绘制目标点
        ax.plot(env.goal_pos[1], env.goal_pos[0], 'g*', markersize=30, 
               label='Goal', zorder=10)
        
        # 绘制路径
        color = colors[name]
        ax.plot(cols, rows, color=color, linewidth=3, alpha=0.8, 
               marker='o', markersize=6, zorder=5)
        
        # 添加方向箭头
        for i in range(len(cols)-1):
            dx = cols[i+1] - cols[i]
            dy = rows[i+1] - rows[i]
            if dx != 0 or dy != 0:
                ax.annotate('', xy=(cols[i+1], rows[i+1]), 
                           xytext=(cols[i], rows[i]),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                         lw=3, alpha=0.7))
        
        # 标记起点和终点
        ax.plot(cols[0], rows[0], 's', color='lime', markersize=12, 
               markeredgecolor='darkgreen', markeredgewidth=2, zorder=8)
        ax.text(cols[0], rows[0], ' S', fontsize=10, fontweight='bold',
               color='white', ha='right', va='center', zorder=9)
        
        final_pos = positions[-1]
        if final_pos == env.goal_pos:
            ax.plot(cols[-1], rows[-1], 'X', color='gold', markersize=16,
                   markeredgecolor='orange', markeredgewidth=2, zorder=8)
        else:
            ax.plot(cols[-1], rows[-1], 'X', color='red', markersize=16,
                   markeredgecolor='darkred', markeredgewidth=2, zorder=8)
        
        success = final_pos == env.goal_pos
        ax.set_title(f'{name}\n({"SUCCESS" if success else "FAILED"}, {len(traj)-1} steps)', 
                    fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.invert_yaxis()
    
    # 第六个子图：说明
    ax = axes[5]
    ax.text(0.5, 0.95, 'Algorithm Comparison', fontsize=14, ha='center',
            transform=ax.transAxes, weight='bold')
    
    info_text = f"""
All methods start from (0,0)
Goal: (9,9) marked with ★
Obstacles: black squares
Start: green square (S)
End: gold X = success
     red X = failed

Arrows show direction of movement
Each line = one algorithm's path
"""
    ax.text(0.05, 0.85, info_text, fontsize=10, ha='left', 
            transform=ax.transAxes, va='top', family='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/alg_policy_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 策略对比图已保存: {save_path}")

if __name__ == "__main__":
    main()
