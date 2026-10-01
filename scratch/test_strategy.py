import sys
import numpy as np

sys.path.insert(0, ".")
from recovery_model import RecallModel
from harness import run_multi
from adapter import Adapter

class SmartAgent(Adapter):
    def __init__(self, stored_patterns: np.ndarray, model_params: dict):
        self.X = stored_patterns  # (K, N)
        self.N = stored_patterns.shape[1]
        self.beta = float(model_params.get("beta", 8.0))
        self.R = model_params["R"]
        self.d_R = np.diag(self.R)
        
        # Precompute pattern features
        # K patterns
        self.K = stored_patterns.shape[0]

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query
        # Softmax matching
        z = self.beta * (self.X @ q)
        z = z - z.max()
        sm = np.exp(z)
        sm = sm / sm.sum()
        
        # Soft reconstructed pattern
        x_hat = self.X.T @ sm
        
        # Agreement between query signal and reconstructed pattern
        # If q_j and x_hat_j agree in sign and magnitude: high confidence
        alignment = q * x_hat
        mag_q = np.abs(q)
        mag_x = np.abs(x_hat)
        
        # Baseline preconditioning for R
        # High diagonal in R needs balancing
        r_weight = 1.0 / np.sqrt(np.maximum(self.d_R, 1e-3))
        
        # Combine signal confidence with preconditioning
        # 1. Boost dimensions where q and x_hat have high positive alignment
        # 2. Downweight masked/zero entries
        pi = 1.0 + 3.0 * np.maximum(alignment, 0.0) + 1.0 * mag_q
        
        return pi

print("Running 7-seed evaluation of SmartAgent...")
report = run_multi(
    agent_factory=lambda X, p: SmartAgent(X, p),
    seeds=[7, 13, 31, 97, 211, 503, 1009],
    K=16, N=64,
    noise_levels=[0.6, 0.75, 0.85],
    n_per_level=50,
    n_aniso=16,
)

print(f"Mean Delta Accuracy: {report['aggregated']['mean_delta']:+.4f}")
print(f"Min Delta Accuracy:  {report['aggregated']['min_delta']:+.4f}")
print(f"Mean Spread Reduc:   {report['aggregated']['mean_reduction']:.2f}x")
print(f"Min Spread Reduc:    {report['aggregated']['min_reduction']:.2f}x")
print(f"Retrieval Points:    {report['score']['retrieval_pts']}")
print(f"Anisotropy Points:   {report['score']['anisotropy_pts']}")
print(f"Total Automated:     {report['score']['total_automated']}")
