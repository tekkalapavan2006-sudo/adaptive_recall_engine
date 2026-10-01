import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel, build_default_R
from data import make_patterns
from metrics import _symmetrised_spread

seed = 42
N = 64
X = make_patterns(16, N, seed=seed)
R = build_default_R(N, seed=seed)
model = RecallModel(X, R)

a_star = model.find_equilibrium(X[0])
H = model.hessian(a_star)
base_s = _symmetrised_spread(np.ones(N), H)
print(f"Base spread: {base_s:.4f}")

# Compute structural features of R
row_sums = np.sum(np.abs(R), axis=1)
row_sq = np.sum(R**2, axis=1)
R_inv = np.linalg.inv(R)
R_inv_row_sums = np.sum(np.abs(R_inv), axis=1)

eigvals, eigvecs = np.linalg.eigh(R)
top_v = np.abs(eigvecs[:, -1])
bot_v = np.abs(eigvecs[:, 0])

tests = {
    "1 / row_sums(R)": 1.0 / row_sums,
    "1 / sqrt(row_sums(R))": 1.0 / np.sqrt(row_sums),
    "1 / row_sq(R)": 1.0 / row_sq,
    "R_inv_row_sums": R_inv_row_sums,
    "1 / top_v(R)": 1.0 / np.maximum(top_v, 1e-4),
    "bot_v(R)": bot_v,
}

for name, pi in tests.items():
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = _symmetrised_spread(pi, H)
    if s is not None:
        print(f"{name:30s} -> Spread: {s:.4f}, Reduction: {base_s / s:.4f}x")
