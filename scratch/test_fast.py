import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel
from harness import run_multi, run_one_seed
from adapter import Adapter

class SmartAgent(Adapter):
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        self.beta = float(model_params.get("beta", 8.0))
        self.R = model_params["R"]
        self.d_R = np.diag(self.R)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query
        z = self.beta * (self.X @ q)
        z = z - z.max()
        sm = np.exp(z)
        sm = sm / sm.sum()
        
        x_hat = self.X.T @ sm
        alignment = q * x_hat
        mag_q = np.abs(q)
        
        pi = 1.0 + 3.0 * np.maximum(alignment, 0.0) + 1.0 * mag_q
        return pi

for seed in [42, 101]:
    rep = run_one_seed(lambda X, p: SmartAgent(X, p), seed=seed, K=16, N=64, noise_levels=[0.75, 0.85], n_per_level=50, n_aniso=16)
    print(f"Seed {seed}: Agent Acc: {rep.agent_acc:.4f}, Base Acc: {rep.baseline_acc:.4f}, Delta: {rep.delta:+.4f}, Spread Reduc: {rep.spread_reduction:.2f}x")
