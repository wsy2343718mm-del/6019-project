"""
专家轨迹可视化分析工具
生成专家路径图和状态访问热力图，帮助诊断数据质量问题
"""
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 设置项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def load_dataset_and_env():
    """加载数据集和环境配置"""
    data_path = os.path.join(DATA_DIR, 'expert_dataset.pkl')
    
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 重建环境
    env_config = dataset['env_config']
    env = GridWorldEnv(
        size=env_config['size'],
        goal_pos=env_config['goal'],
        obstacles=env_config['obstacles'],
        stochasticity=env_config['stochasticity'],
        feature_mode="coords"
    )
    
    return env, dataset

def visualize_trajectories(env, trajectories, title="Expert Trajectories", 
                          save_path="figures/expert_trajectories.png", 
                          max_display=10):
    """可视化多条专家轨迹 - 改进版，更清晰"""
    
    # 策略1：显示少量轨迹，但每条都清晰
    # 策略2：创建子图，每张显示几条轨迹
    num_per_fig = 10  # 每张图显示10条轨迹
    display_trajectories = trajectories[:min(len(trajectories), max_display)]
    num_figs = (len(display_trajectories) + num_per_fig - 1) // num_per_fig
    
    for fig_idx in range(num_figs):
        start_idx = fig_idx * num_per_fig
        end_idx = min(start_idx + num_per_fig, len(display_trajectories))
        current_trajs = display_trajectories[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
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
        ax.plot(env.goal_pos[1], env.goal_pos[0], 'g*', markersize=35, 
               label='Goal (9,9)', zorder=10, markeredgecolor='darkgreen', 
               markeredgewidth=2)
        
        # 绘制轨迹
        colors = plt.cm.tab10(np.linspace(0, 1, len(current_trajs)))
        
        for local_idx, traj in enumerate(current_trajs):
            global_idx = start_idx + local_idx
            states = [step[0] for step in traj]
            # 添加最后一步的next_state
            states.append(traj[-1][3])
            positions = [env._idx_to_pos(s) for s in states]
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]
            
            # 绘制轨迹线（带箭头）
            ax.plot(cols, rows, color=colors[local_idx], alpha=0.8, linewidth=2.5, 
                   marker='o', markersize=6, zorder=5,
                   label=f'Traj #{global_idx+1}')
            
            # 添加方向箭头
            for i in range(len(cols)-1):
                dx = cols[i+1] - cols[i]
                dy = rows[i+1] - rows[i]
                if dx != 0 or dy != 0:  # 只有移动了才画箭头
                    ax.annotate('', xy=(cols[i+1], rows[i+1]), 
                               xytext=(cols[i], rows[i]),
                               arrowprops=dict(arrowstyle='->', color=colors[local_idx], 
                                             lw=2.5, alpha=0.6))
            
            # 标记起点（绿色方块）
            ax.plot(cols[0], rows[0], 's', color='lime', markersize=12, 
                   markeredgecolor='darkgreen', markeredgewidth=2, zorder=8)
            ax.text(cols[0], rows[0], f' S', fontsize=9, fontweight='bold',
                   color='white', ha='right', va='center', zorder=9)
            
            # 标记终点（红色X或绿色勾）
            final_pos = (rows[-1], cols[-1])
            if final_pos == env.goal_pos:
                # 成功到达目标
                ax.plot(cols[-1], rows[-1], 'X', color='gold', markersize=14,
                       markeredgecolor='orange', markeredgewidth=2, zorder=8)
            else:
                # 未到达目标
                ax.plot(cols[-1], rows[-1], 'X', color='red', markersize=14,
                       markeredgecolor='darkred', markeredgewidth=2, zorder=8)
        
        ax.set_xlabel('Column', fontsize=12, fontweight='bold')
        ax.set_ylabel('Row', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n(Trajectories #{start_idx+1}-#{end_idx})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.invert_yaxis()  # 矩阵坐标
        
        # 保存
        base_name = os.path.splitext(save_path)[0]
        if num_figs > 1:
            fig_save_path = f"{base_name}_part{fig_idx+1}.png"
        else:
            fig_save_path = save_path
        
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 轨迹图已保存: {fig_save_path}")
        plt.close()

def plot_heatmap(env, trajectories, title="State Visitation Heatmap",
                save_path="figures/visitation_heatmap.png"):
    """绘制状态访问热力图"""
    # 统计每个状态的访问次数
    visit_counts = np.zeros(env.n_states)
    
    for traj in trajectories:
        for step in traj:
            state = step[0]
            visit_counts[state] += 1
    
    # 转换为二维热力图
    heatmap = np.zeros((env.size, env.size))
    for s in range(env.n_states):
        row, col = env._idx_to_pos(s)
        heatmap[row, col] = visit_counts[s]
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                             ['white', 'yellow', 'orange', 'red'])
    
    im = ax.imshow(heatmap, cmap=cmap, aspect='equal', interpolation='nearest')
    
    # 绘制障碍物
    for obs in env.obstacles:
        rect = plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                            facecolor='black', edgecolor='darkgray', 
                            alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(obs[1], obs[0], '✕', color='white', fontsize=12,
               ha='center', va='center', fontweight='bold')
    
    # 绘制目标点
    ax.plot(env.goal_pos[1], env.goal_pos[0], 'g*', markersize=35,
           label='Goal', zorder=10)
    
    # 添加数值标签
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) not in env.obstacles and (i, j) != env.goal_pos:
                count = int(heatmap[i, j])
                if count > 0:
                    ax.text(j, i, str(count), ha='center', va='center',
                           fontsize=8, fontweight='bold', alpha=0.7)
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xticklabels(range(env.size))
    ax.set_yticklabels(range(env.size))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Visit Count', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 热力图已保存: {save_path}")
    plt.close()
    
    return heatmap

def analyze_trajectory_quality(env, trajectories):
    """分析轨迹质量统计信息"""
    print("\n" + "="*60)
    print("📊 专家轨迹质量分析")
    print("="*60)
    
    total = len(trajectories)
    success_count = 0
    goal_steps = []
    traj_lengths = []
    obstacle_hits = 0
    
    for traj in trajectories:
        traj_lengths.append(len(traj))
        
        # 检查是否到达目标
        # traj 格式: [(state, action, reward, next_state, done), ...]
        # 最后一步的 next_state 才是终点
        final_step = traj[-1]
        final_state = final_step[3]  # next_state
        final_pos = env._idx_to_pos(final_state)
        
        if final_pos == env.goal_pos:
            success_count += 1
            goal_steps.append(len(traj))
        elif final_pos in env.obstacles:
            obstacle_hits += 1
    
    success_rate = success_count / total * 100
    avg_length = np.mean(traj_lengths)
    std_length = np.std(traj_lengths)
    avg_goal_steps = np.mean(goal_steps) if goal_steps else 0
    
    print(f"轨迹总数: {total}")
    print(f"✅ 成功率: {success_rate:.2f}% ({success_count}/{total})")
    print(f"💥 碰撞障碍物: {obstacle_hits} ({obstacle_hits/total*100:.2f}%)")
    print(f"📏 平均轨迹长度: {avg_length:.2f} ± {std_length:.2f} 步")
    if goal_steps:
        print(f"🎯 到达目标平均步数: {avg_goal_steps:.2f} 步")
    
    # 起点分布分析
    start_positions = np.zeros((env.size, env.size))
    for traj in trajectories:
        start_state = traj[0][0]
        start_pos = env._idx_to_pos(start_state)
        start_positions[start_pos[0], start_pos[1]] += 1
    
    print(f"\n📍 起点分布 (前10个最常见起点):")
    start_flat = [(start_positions[i//env.size, i%env.size], i//env.size, i%env.size) 
                  for i in range(env.n_states)]
    start_flat.sort(reverse=True)
    for count, row, col in start_flat[:10]:
        if count > 0:
            print(f"   ({row},{col}): {int(count)} 次")
    
    print("="*60)
    
    return {
        'success_rate': success_rate,
        'avg_length': avg_length,
        'goal_steps': avg_goal_steps,
        'obstacle_hits': obstacle_hits
    }

def main():
    print("🔍 加载专家数据集和环境...")
    env, dataset = load_dataset_and_env()
    
    clean_trajs = dataset['clean_trajectories']
    noisy_trajs = dataset['noisy_trajectories']
    
    print(f"\n📦 数据集信息:")
    print(f"   干净轨迹数: {len(clean_trajs)}")
    print(f"   噪声轨迹数: {len(noisy_trajs)}")
    print(f"   障碍物数量: {len(env.obstacles)}")
    print(f"   目标位置: {env.goal_pos}")
    
    # 1. 分析干净轨迹
    print("\n" + "="*60)
    print("🔬 分析干净专家轨迹 (action_noise=0.0)")
    print("="*60)
    clean_stats = analyze_trajectory_quality(env, clean_trajs)
    
    # 2. 分析噪声轨迹
    print("\n" + "="*60)
    print("🔊 分析噪声轨迹 (action_noise=0.15)")
    print("="*60)
    noisy_stats = analyze_trajectory_quality(env, noisy_trajs)
    
    # 3. 可视化
    print("\n🎨 生成可视化图表...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 干净轨迹可视化
    visualize_trajectories(env, clean_trajs, 
                          title="Clean Expert Trajectories (noise=0.0)",
                          save_path=os.path.join(FIGURES_DIR, "clean_trajectories.png"),
                          max_display=10)
    
    plot_heatmap(env, clean_trajs,
                title="Clean Trajectory Visitation Heatmap",
                save_path=os.path.join(FIGURES_DIR, "clean_heatmap.png"))
    
    # 噪声轨迹可视化
    visualize_trajectories(env, noisy_trajs,
                          title="Noisy Expert Trajectories (noise=0.15)",
                          save_path=os.path.join(FIGURES_DIR, "noisy_trajectories.png"),
                          max_display=10)
    
    plot_heatmap(env, noisy_trajs,
                title="Noisy Trajectory Visitation Heatmap",
                save_path=os.path.join(FIGURES_DIR, "noisy_heatmap.png"))
    
    print("\n✅ 所有可视化完成！")
    print("📁 查看生成的图表:")
    print("   - figures/clean_trajectories.png")
    print("   - figures/clean_heatmap.png")
    print("   - figures/noisy_trajectories.png")
    print("   - figures/noisy_heatmap.png")

if __name__ == "__main__":
    main()
