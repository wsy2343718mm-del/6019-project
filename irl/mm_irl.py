import cvxpy as cp
import numpy as np

def solve_mm_irl(mu_E, mu_subs, n_features):
    """MM-IRL: max margin s.t. w@(μ_E-μ_π)>=margin, Σw=1, w>=0, ||w||_2<=1"""
    w = cp.Variable(n_features)
    margin = cp.Variable(nonneg=True)
    
    constraints = [w @ (mu_E - mu_sub) >= margin for mu_sub in mu_subs]
    constraints.append(cp.sum(w) == 1.0)
    constraints.append(w >= 0)
    constraints.append(cp.norm(w, 2) <= 1.0)
    
    prob = cp.Problem(cp.Maximize(margin), constraints)
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=2000)
    
    if w.value is None or margin.value < 1e-4:
        print("⚠️ MM-IRL 边距不足，返回均匀权重")
        return np.ones(n_features) / n_features, 0.0
    return w.value, margin.value