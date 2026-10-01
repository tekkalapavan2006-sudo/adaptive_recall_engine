import sys
import numpy as np

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
print(f"Base spread: {base_s:.4f}")

# Try different formulas for pi
candidates = {}

# 1. Inverse diagonal of H
d = np.diag(H)
candidates["1 / diag(H)"] = 1.0 / np.maximum(d, 1e-5)
candidates["1 / sqrt(diag(H))"] = 1.0 / np.sqrt(np.maximum(d, 1e-5))

# 2. Row norms of H
rn = np.linalg.norm(H, axis=1)
candidates["1 / row_norm(H)"] = 1.0 / np.maximum(rn, 1e-5)
candidates["1 / sqrt(row_norm(H))"] = 1.0 / np.sqrt(np.maximum(rn, 1e-5))

# 3. Inverse of H diagonal
H_inv = np.linalg.inv(H)
d_inv = np.diag(H_inv)
candidates["diag(H_inv)"] = d_inv
candidates["1 / diag(H_inv)"] = 1.0 / np.maximum(d_inv, 1e-5)

# 4. Top eigenvector of H
eigvals, eigvecs = np.linalg.eigh(H)
v_max = eigvecs[:, -1]
v_min = eigvecs[:, 0]
candidates["1 / (v_max^2 + eps)"] = 1.0 / (v_max**2 + 0.01)
candidates["v_min^2 + eps"] = v_min**2 + 0.01

# 5. Inverse quadratic form with X[0]
candidates["1 / (abs(X[0]) + eps)"] = 1.0 / (np.abs(X[0]) + 0.05)
candidates["abs(X[0]) + eps"] = np.abs(X[0]) + 0.05

for name, pi in candidates.items():
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = _symmetrised_spread(pi, H)
    if s is not None:
        print(f"{name:30s} -> Spread: {s:.4f} (Reduction: {base_s / s:.2f}x)")
    else:
        print(f"{name:30s} -> Invalid")
