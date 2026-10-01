import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel
from harness import run_one_seed
from adapter import Adapter

def test_func(func, name):
    class TempAgent(Adapter):
        def __init__(self, stored_patterns, model_params):
            self.X = stored_patterns
            self.N = stored_patterns.shape[1]
            self.R = model_params["R"]
            self.beta = float(model_params.get("beta", 8.0))
            self.eta = float(model_params.get("eta", 0.5))

        def predict_precision(self, corrupted_query):
            return func(self, corrupted_query)

    rep = run_one_seed(lambda X, p: TempAgent(X, p), seed=42, K=16, N=64, noise_levels=[0.75, 0.85], n_per_level=30, n_aniso=16)
    print(f"{name:45s} -> Acc: {rep.agent_acc:.4f} (Base: {rep.baseline_acc:.4f}, Delta: {rep.delta:+.4f}), Reduc: {rep.spread_reduction:.2f}x")
    return rep.delta, rep.spread_reduction

# Test 1: Baseline R preconditioning
def f1(self, q):
    # R diagonal preconditioning
    d = np.diag(self.R)
    return 1.0 / np.sqrt(d)

test_func(f1, "1/sqrt(diag(R))")

# Test 2: R inverse diagonal
def f2(self, q):
    R_inv = np.linalg.inv(self.R)
    return np.diag(R_inv)

test_func(f2, "diag(R_inv)")

# Test 3: Magnitude-based small shift
def f3(self, q):
    return 1.0 + 0.05 * np.abs(q)

test_func(f3, "1 + 0.05*|q|")

# Test 4: Pattern-wise variance preconditioning
def f4(self, q):
    # Stored pattern per-dimension variance across K patterns
    var_X = np.var(self.X, axis=0)
    return 1.0 + 0.5 * var_X

test_func(f4, "1 + 0.5*var(X)")

# Test 5: inverse pattern variance
def f5(self, q):
    var_X = np.var(self.X, axis=0) + 1e-4
    return 1.0 / var_X

test_func(f5, "1 / var(X)")

# Test 6: Pattern energy gradient preconditioning
def f6(self, q):
    z = self.beta * (self.X @ q)
    s = np.exp(z - z.max())
    s = s / s.sum()
    recon = self.X.T @ s
    # precision proportional to expected magnitude of predicted pattern
    return np.abs(recon) + 0.01

test_func(f6, "abs(recon) + 0.01")
