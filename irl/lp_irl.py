import numpy as np
from scipy.optimize import linprog

def solve_lp_irl(mu_E, mu_subs, n_features):
    """LP-IRL: min -Σw (encourage non-degenerate solutions) s.t. w@(μ_E-μ_π)>=0, Σw=1, w>=0"""
    # 使用小的负权重和作为目标，避免退化解[0,0,...,1]
    c = -np.ones(n_features) * 0.01
    A_ub = np.array([mu_sub - mu_E for mu_sub in mu_subs])
    b_ub = np.full(len(mu_subs), 5e-4)  # 微小容差防浮点误差
    A_eq = np.ones((1, n_features))
    b_eq = np.array([1.0])              # 固定尺度，避免零解
    bounds = [(0, 1.0) for _ in range(n_features)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        print("⚠️ LP-IRL 不可行，返回均匀权重")
        return np.ones(n_features) / n_features
    return res.x