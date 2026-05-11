"""全面验证 LP-IRL 和 MM-IRL 实现的正确性"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np

# 加载结果
with open('../data/expert_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
with open('../data/irl_results.pkl', 'rb') as f:
    results = pickle.load(f)

print('='*60)
print('📊 全面验证报告')
print('='*60)

# 1. 检查特征期望向量
mu_E = results['mu_E']
print(f'\n【1】专家特征期望 μ_E: {mu_E}')
print(f'    数量级: ~{np.linalg.norm(mu_E):.1f} (应该与轨迹数和折扣因子相关)')

# 2. 检查约束满足的具体 gap
w_lp = results['w_lp']
w_mm = results['w_mm']
print(f'\n【2】LP-IRL 权重: {w_lp}')
print(f'    权重和: {w_lp.sum():.6f} (应该=1)')
print(f'    权重非负: {np.all(w_lp >= -1e-6)}')

print(f'\n【3】MM-IRL 权重: {w_mm}')
print(f'    权重和: {w_mm.sum():.6f}')
print(f'    权重非负: {np.all(w_mm >= -1e-6)}')
print(f'    L2范数: {np.linalg.norm(w_mm):.4f} (应该<=1)')

# 4. 检查恢复的奖励分布
R_gt = results['R_gt']
R_lp = results['R_lp']
R_mm = results['R_mm']
print(f'\n【4】真实奖励分布:')
print(f'    有奖励的状态数: {np.sum(R_gt > 0)} (稀疏奖励应有少数状态有奖励)')
print(f'    最大奖励: {R_gt.max():.2f} at state {np.argmax(R_gt)}')

print(f'\n【5】LP-IRL 恢复的奖励:')
print(f'    范围: [{R_lp.min():.3f}, {R_lp.max():.3f}]')
print(f'    标准差: {np.std(R_lp):.3f} (太小可能表示没学到有意义的信息)')

print(f'\n【6】MM-IRL 恢复的奖励:')
print(f'    范围: [{R_mm.min():.3f}, {R_mm.max():.3f}]')
print(f'    标准差: {np.std(R_mm):.3f}')

# 5. 关键：检查目标状态(5,5)的奖励是否最高
goal_state = 99  # 9*10+9=99
print(f'\n【7】目标状态 (5,5) 的奖励对比:')
print(f'    Ground Truth: {R_gt[goal_state]:.3f}')
print(f'    LP-IRL:       {R_lp[goal_state]:.3f} (排名: {np.argsort(R_lp)[::-1].tolist().index(goal_state)+1}/{len(R_lp)})')
print(f'    MM-IRL:       {R_mm[goal_state]:.3f} (排名: {np.argsort(R_mm)[::-1].tolist().index(goal_state)+1}/{len(R_mm)})')

# 6. 检查约束 gap（应该都>=0）
from irl.utils import generate_suboptimal_mus
from env.gridworld import GridWorldEnv
cfg = data['env_config']
env = GridWorldEnv(size=cfg['size'], goal_pos=cfg['goal'], obstacles=[], 
                   stochasticity=cfg['stochasticity'], feature_mode='coords', gamma=cfg['gamma'])
mu_subs = generate_suboptimal_mus(env)

print(f'\n【8】约束满足 Gap (w·(μ_E - μ_sub)):')
for i, mu_sub in enumerate(mu_subs):
    gap_lp = w_lp @ (mu_E - mu_sub)
    gap_mm = w_mm @ (mu_E - mu_sub)
    print(f'    次优策略{i+1}: LP-IRL gap={gap_lp:.4f}, MM-IRL gap={gap_mm:.4f}')

print(f'\n【9】成功率指标:')
success_lp = results['success_lp']
success_mm = results['success_mm']
print(f'    LP-IRL: {success_lp:.1%}')
print(f'    MM-IRL: {success_mm:.1%}')

# 10. 最终判断
print('\n' + '='*60)
print('📋 验证清单')
print('='*60)
checks = []
checks.append(('权重归一化 (Σw=1)', abs(w_lp.sum() - 1) < 1e-6 and abs(w_mm.sum() - 1) < 1e-6))
checks.append(('权重非负 (w>=0)', np.all(w_lp >= -1e-6) and np.all(w_mm >= -1e-6)))
checks.append(('约束满足 (gap>=0)', all(w_lp @ (mu_E - mu_sub) >= -1e-5 for mu_sub in mu_subs)))
checks.append(('目标状态奖励排名靠前', np.argsort(R_lp)[::-1].tolist().index(goal_state) < 5))
checks.append(('策略成功率>90%', success_lp > 0.9 and success_mm > 0.9))

for name, passed in checks:
    status = '✅' if passed else '❌'
    print(f'{status} {name}')

all_passed = all(c[1] for c in checks)
print('\n' + ('🎉 所有检查通过！算法实现正确' if all_passed else '⚠️ 存在问题，需检查'))