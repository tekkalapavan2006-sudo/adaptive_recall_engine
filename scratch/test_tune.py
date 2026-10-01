import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel
from harness import run_one_seed
from adapter import Adapter

def evaluate_formula(func, name):
    class TempAgent(Adapter):
        def __init__(self, stored_patterns, model_params):
            self.X = stored_patterns
            self.N = stored_patterns.shape[1]
            self.R = model_params["R"]
            self.beta = float(model_params.get("beta", 8.0))

        def predict_precision(self, corrupted_query):
            return func(self, corrupted_query)

    deltas = []
    reductions = []
    for seed in [42, 101]:
        rep = run_one_seed(lambda X, p: TempAgent(X, p), seed=seed, K=16, N=64, noise_levels=[0.75, 0.85], n_per_level=50, n_aniso=16)
        deltas.append(rep.delta)
        reductions.append(rep.spread_reduction)
    
    mean_d = np.mean(deltas)
    mean_r = np.mean(reductions)
    print(f"{name:40s} -> Delta: {mean_d:+.4f}, Reduction: {mean_r:.2f}x")
    return mean_d, mean_r

# Formula 1: Gentle scaling based on query magnitude
evaluate_formula(lambda self, q: 1.0 + 0.1 * np.abs(q), "1 + 0.1*|q|")
evaluate_formula(lambda self, q: 1.0 + 0.2 * (np.abs(q) - np.mean(np.abs(q))), "1 + 0.2*(|q| - mean)")

# Formula 2: Softmax class confidence weighting
def f2(self, q):
    z = self.beta * (self.X @ q)
    s = np.exp(z - z.max())
    s = s / s.sum()
    recon = self.X.T @ s
    # precision based on agreement with reconstructed pattern
    diff = np.abs(q - recon)
    return 1.0 - 0.1 * diff

evaluate_formula(f2, "1 - 0.1*|q - recon|")

# Formula 3: Inverse energy gradient preconditioning
def f3(self, q):
    z = self.beta * (self.X @ q)
    s = np.exp(z - z.max())
    s = s / s.sum()
    g = self.R @ q - 0.5 * (self.X.T @ s)
    return 1.0 / (1.0 + 0.1 * np.abs(g))

evaluate_formula(f3, "1 / (1 + 0.1*|grad|)")
