import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from recovery_model import RecallModel, build_default_R
from data import make_patterns
from metrics import _symmetrised_spread

seed = 42
X = make_patterns(16, 64, seed=seed)
R = build_default_R(64, seed=seed)
model = RecallModel(X, R)

a_star = model.find_equilibrium(X[0])
H = model.hessian(a_star)
base_s = _symmetrised_spread(np.ones(64), H)

# Let's test custom objective function with penalty for variance or smooth spectrum
def loss(pi):
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = _symmetrised_spread(pi, H)
    if s is None:
        return 1e9
    return s

# Try Powell optimization from random restarts
best_val = base_s
best_pi = np.ones(64)

for i in range(10):
    init_pi = np.random.uniform(0.5, 1.5, 64) if i > 0 else np.ones(64)
    res = minimize(loss, init_pi, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
    if res.fun < best_val:
        best_val = res.fun
        best_pi = res.x

best_pi = np.clip(best_pi, 0.1, 10.0)
best_pi = best_pi / best_pi.mean()
best_pi = np.clip(best_pi, 0.1, 10.0)
print(f"Base spread: {base_s:.4f}")
print(f"Best Powell spread: {best_val:.4f}, Reduction: {base_s / best_val:.2f}x")
