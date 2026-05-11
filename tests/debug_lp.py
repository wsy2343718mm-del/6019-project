import os, sys, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from env.gridworld import GridWorldEnv
from irl.utils import compute_feature_expectation, generate_suboptimal_mus
from irl.lp_irl import solve_lp_irl
from data.generate_expert import generate_trajectories

with open('data/expert_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
cfg = dataset['env_config']
obstacles = cfg.get('obstacles', [])
env = GridWorldEnv(size=cfg['size'], goal_pos=cfg['goal'], obstacles=obstacles,
                   stochasticity=cfg['stochasticity'], feature_mode='rbf', gamma=cfg['gamma'])

# 测试 0 噪声
noisy_0 = generate_trajectories(env, n_episodes=500, action_noise=0.0, seed=42)
mu_0 = compute_feature_expectation(noisy_0, env)
mu_subs = generate_suboptimal_mus(env)

w_lp = solve_lp_irl(mu_0, mu_subs, 3)
R_lp = env.Phi @ w_lp
policy_lp, _ = env.compute_optimal_policy(reward_vec=R_lp * 5.0)

success = 0
for _ in range(50):
    s, _ = env.reset(options={'fixed_start': (0,0)})
    done = False
    while not done:
        ns, _, term, trunc, _ = env.step(policy_lp[s])
        done = term or trunc
        s = ns
    if env._idx_to_pos(s) == env.goal_pos:
        success += 1

lines = []
lines.append(f'max_episode_steps: {env.max_episode_steps}')
lines.append(f'0噪声 LP权重: {w_lp.round(4)}')
lines.append(f'0噪声成功率: {success/50:.0%}')
lines.append(f'mu_E: {mu_0.round(4)}')
for i, ms in enumerate(mu_subs):
    lines.append(f'mu_sub[{i}]: {ms.round(4)}, diff: {(mu_0 - ms).round(4)}')

# 测试 0.1 噪声
noisy_1 = generate_trajectories(env, n_episodes=500, action_noise=0.1, seed=42+1000)
mu_1 = compute_feature_expectation(noisy_1, env)
w_lp_1 = solve_lp_irl(mu_1, mu_subs, 3)
R_lp_1 = env.Phi @ w_lp_1
policy_lp_1, _ = env.compute_optimal_policy(reward_vec=R_lp_1 * 5.0)
success1 = 0
for _ in range(50):
    s, _ = env.reset(options={'fixed_start': (0,0)})
    done = False
    while not done:
        ns, _, term, trunc, _ = env.step(policy_lp_1[s])
        done = term or trunc
        s = ns
    if env._idx_to_pos(s) == env.goal_pos:
        success1 += 1

lines.append(f'\n0.1噪声 LP权重: {w_lp_1.round(4)}')
lines.append(f'0.1噪声成功率: {success1/50:.0%}')
lines.append(f'mu_E(noisy): {mu_1.round(4)}')
for i, ms in enumerate(mu_subs):
    lines.append(f'mu_sub[{i}]: {ms.round(4)}, diff: {(mu_1 - ms).round(4)}')

with open('tests/lp_debug.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
