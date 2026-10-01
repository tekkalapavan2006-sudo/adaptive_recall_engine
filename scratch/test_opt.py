import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from recovery_model import RecallModel, build_default_R
from harness import run_one_seed, pack_params
from metrics import _symmetrised_spread
from adapter import Adapter

def calc_spread(pi, H, pi_min=0.1, pi_max=10.0):
    # clip and normalise
    pi = np.clip(pi, pi_min, pi_max)
    pi = pi / pi.mean()
    pi = np.clip(pi, pi_min, pi_max)
    s = _symmetrised_spread(pi, H)
    return s if s is not None else 1e9

# Test optimization on Seed 42
X = np.load if False else None
seed = 42
from data import make_patterns
X = make_patterns(16, 64, seed=seed)
R = build_default_R(64, seed=seed)
model = RecallModel(X, R)

a_star = model.find_equilibrium(X[0])
H = model.hessian(a_star)
base_s = _symmetrised_spread(np.ones(64), H)
print(f"Base spread for pattern 0: {base_s:.2f}")

# Optimize pi using scipy minimize
def obj(log_pi):
    pi = np.exp(log_pi)
    return calc_spread(pi, H)

res = minimize(obj, np.zeros(64), method='L-BFGS-B', options={'maxiter': 200})
best_pi = np.exp(res.x)
best_pi = np.clip(best_pi, 0.1, 10.0)
best_pi = best_pi / best_pi.mean()
opt_s = _symmetrised_spread(best_pi, H)
print(f"Optimized spread for pattern 0: {opt_s:.2f}, Reduction: {base_s / opt_s:.2f}x")
