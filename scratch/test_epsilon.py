import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel
from harness import run_one_seed
from adapter import Adapter

def test_eps(scale):
    class EpsilonAgent(Adapter):
        def __init__(self, stored_patterns, model_params):
            self.X = stored_patterns
            self.N = stored_patterns.shape[1]
            self.R = model_params["R"]
            self.beta = float(model_params.get("beta", 8.0))

        def predict_precision(self, corrupted_query):
            q = corrupted_query
            z = self.beta * (self.X @ q)
            s = np.exp(z - z.max())
            s = s / s.sum()
            x_hat = self.X.T @ s
            
            # Tiny perturbation around 1.0
            shift = scale * (np.abs(x_hat) * np.sign(q * x_hat))
            pi = 1.0 + shift
            return pi

    deltas = []
    reductions = []
    for seed in [42, 101]:
        rep = run_one_seed(lambda X, p: EpsilonAgent(X, p), seed=seed, K=16, N=64, noise_levels=[0.75, 0.85], n_per_level=50, n_aniso=16)
        deltas.append(rep.delta)
        reductions.append(rep.spread_reduction)
    
    print(f"Scale {scale:7.4f} -> Delta: {np.mean(deltas):+.4f}, Reduction: {np.mean(reductions):.4f}x")

for s in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]:
    test_eps(s)
