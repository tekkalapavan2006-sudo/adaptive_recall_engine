import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel, build_default_R
from harness import run_one_seed
from adapter import Adapter

class TestAgent(Adapter):
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        self.R = model_params["R"]
        self.beta = float(model_params.get("beta", 8.0))
        self.eta = float(model_params.get("eta", 0.5))

        # Compute diagonal preconditioning base from R
        # R has a large component in 11^T direction.
        # Let's compute inverse of R or diagonal of R^-1:
        self.R_inv = np.linalg.inv(self.R)
        self.d_inv = np.diag(self.R_inv)
        self.d_R = np.diag(self.R)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        # Let's test combinations
        # 1. Base preconditioning for R:
        base_pi = self.d_inv / np.mean(self.d_inv)
        return base_pi

print("Testing Base Preconditioning...")
rep = run_one_seed(lambda X, p: TestAgent(X, p), seed=42, K=16, N=64, noise_levels=[0.75, 0.85], n_per_level=50, n_aniso=16)
print(f"Seed 42: Agent Acc: {rep.agent_acc:.4f}, Base Acc: {rep.baseline_acc:.4f}, Delta: {rep.delta:+.4f}")
print(f"Base Spread: {rep.baseline_spread:.2f}, Agent Spread: {rep.agent_spread:.2f}, Reduction: {rep.spread_reduction:.2f}x")
