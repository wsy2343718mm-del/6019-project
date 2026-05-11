"""全面验证 MaxEnt-IRL + Preference-BT 实现的正确性"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation

# 加载结果
with open('../data/expert_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
with open('../data/maxent_pref_results.pkl', 'rb') as f:
    results = pickle.load(f)
with open('../data/irl_results.pkl', 'rb') as f:
    results_s2 = pickle.load(f)

cfg = data['env_config']
env = GridWorldEnv(size=cfg['size'], goal_pos=cfg['goal'], obstacles=[], 
                   stochasticity=cfg['stochasticity'], feature_mode='coords', gamma=cfg['gamma'])

print('='*60)
print('📊 MaxEnt-IRL / Preference-BT 全面验证报告')
print('='*60)

# 提取结果
w_maxent = results['MaxEnt-IRL']['w']
w_pref = results['Preference-BT']['w']

# 1. 检查权重
print(f'\n【1】MaxEnt-IRL 权重: {w_maxent}')
print(f'    L2范数: {np.linalg.norm(w_maxent):.4f}')
print(f'    权重有区分度: {np.std(w_maxent) > 1e-3}')

print(f'\n【2】Preference-BT 权重: {w_pref}')
print(f'    L2范数: {np.linalg.norm(w_pref):.4f}')
print(f'    权重有区分度: {np.std(w_pref) > 1e-3}')
if np.linalg.norm(w_pref) < 1e-2:
    print('    ⚠️ 权重几乎为零，需要检查！')

# 2. 目标状态奖励排名
goal_state = 99  # 9*10+9=99
R_gt = env.ground_truth_reward
R_maxent = env.Phi @ w_maxent
R_pref = env.Phi @ w_pref

print(f'\n【3】目标状态 (5,5) 的奖励对比:')
print(f'    Ground Truth: {R_gt[goal_state]:.3f}')
print(f'    MaxEnt-IRL:   {R_maxent[goal_state]:.3f} (排名: {np.argsort(R_maxent)[::-1].tolist().index(goal_state)+1}/{len(R_maxent)})')
print(f'    Preference-BT: {R_pref[goal_state]:.3f} (排名: {np.argsort(R_pref)[::-1].tolist().index(goal_state)+1}/{len(R_pref)})')

# 3. 奖励场分布
print(f'\n【4】恢复奖励场统计:')
print(f'    MaxEnt-IRL: 范围 [{R_maxent.min():.3f}, {R_maxent.max():.3f}], 标准差 {np.std(R_maxent):.3f}')
print(f'    Preference-BT: 范围 [{R_pref.min():.3f}, {R_pref.max():.3f}], 标准差 {np.std(R_pref):.3f}')

# 4. 偏好数据分析
pairs = data['preference_pairs']
labels = data['preference_labels']
returns = data['trajectory_returns']
print(f'\n【5】偏好数据分析:')
print(f'    偏好对数: {len(pairs)}')
print(f'    正例比例: {np.mean(labels):.1%}')
print(f'    平均回报差: {np.mean([abs(returns[i] - returns[j]) for i, j in pairs]):.4f}')

# 检查偏好数据是否有区分度
same_return_count = sum(1 for i, j in pairs if abs(returns[i] - returns[j]) < 1e-6)
print(f'    回归相同的偏好对数: {same_return_count}/{len(pairs)}')
if same_return_count > len(pairs) * 0.5:
    print('    ⚠️ 大多数偏好对的回报相同，这会导致 Bradley-Terry 无法学习有效权重！')

# 5. 与 LP-IRL / MM-IRL 算法对比
print(f'\n【6】与 LP-IRL / MM-IRL 算法对比:')
w_lp = results_s2['w_lp']
w_mm = results_s2['w_mm']
print(f'    LP-IRL  成功率: {results_s2["success_lp"]:.1%}, 相关系数: {results_s2["corr_lp"]:.3f}')
print(f'    MM-IRL  成功率: {results_s2["success_mm"]:.1%}, 相关系数: {results_s2["corr_mm"]:.3f}')
print(f'    MaxEnt  成功率: {results["MaxEnt-IRL"]["acc"]:.1%}, 相关系数: {results["MaxEnt-IRL"]["corr"]:.3f}')
print(f'    Pref-BT 成功率: {results["Preference-BT"]["acc"]:.1%}, 相关系数: {results["Preference-BT"]["corr"]:.3f}')

# 6. 验证清单
print('\n' + '='*60)
print('📋 验证清单')
print('='*60)

checks = []

# MaxEnt-IRL 检查
checks.append(('MaxEnt 权重非零', np.linalg.norm(w_maxent) > 1e-3))
checks.append(('MaxEnt 目标状态排名靠前', np.argsort(R_maxent)[::-1].tolist().index(goal_state) < 5))
checks.append(('MaxEnt 奖励场有区分度', np.std(R_maxent) > 1e-3))
checks.append(('MaxEnt 策略成功率>90%', results['MaxEnt-IRL']['acc'] > 0.9))

# Preference-BT 检查
pref_ok = np.linalg.norm(w_pref) > 1e-3
checks.append(('Pref-BT 权重非零', pref_ok))
checks.append(('Pref-BT 目标状态排名靠前', np.argsort(R_pref)[::-1].tolist().index(goal_state) < 10))
checks.append(('Pref-BT 策略成功率>90%', results['Preference-BT']['acc'] > 0.9))

for name, passed in checks:
    status = '✅' if passed else '❌'
    print(f'{status} {name}')

print('\n' + '='*60)
if all(c[1] for c in checks):
    print('🎉 所有检查通过！MaxEnt-IRL / Preference-BT 实现正确')
else:
    print('⚠️ 存在问题，需进一步分析')
print('='*60)