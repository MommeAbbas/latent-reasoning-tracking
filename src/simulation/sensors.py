import numpy as np
from dataclasses import dataclass

def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

def _log_gaussian_diag(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    return float(-0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(((y - mu) ** 2) / var)))

@dataclass
class SensorConfig:
    obs_dim: int = 3
    noise_std: tuple = (0.008, 0.008, 0.008) 
    outlier_prob: float = 0.05
    outlier_scale: float = 50.0 
    clip_observation: bool = False

class ReasoningSensors:
    def __init__(self, config: SensorConfig = SensorConfig()):
        self.cfg = config

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Hidden Accumulator model
        x = [p, c, u, f, m]
        
        'p' (x[0]) is not observed directly
        """
        p, c, u, f, m = x[0], x[1], x[2], x[3], x[4]

        # Pure Coherence vs Uncertainty (no progress)
        y1 = c - 0.5 * u
        
        # Interaction term with sin(c) to make it nonlinear
        # Helps RBPF lock onto the mode via rotation, but doesn't reveal 'p'
        y2 = 0.8 * c + 0.3 * u + 0.1 * np.sin(4.0 * np.pi * c)
        
        # Fatigue/momentum monitor
        y3 = f - 0.2 * m + 0.1 * (u * c)
        
        return np.array([y1, y2, y3], dtype=float)

    def observe(self, x: np.ndarray) -> np.ndarray:
        mu = self.h(x)
        std_in = np.array(self.cfg.noise_std, dtype=float)
        
        if np.random.rand() < self.cfg.outlier_prob:
            std = std_in * np.sqrt(self.cfg.outlier_scale)
        else:
            std = std_in

        y = mu + np.random.randn(self.cfg.obs_dim) * std
        if self.cfg.clip_observation:
            y = np.clip(y, 0.0, 1.0)
        return y

    def log_likelihood(self, y: np.ndarray, x: np.ndarray) -> float:
        mu = self.h(x)
        var_in = (np.array(self.cfg.noise_std, dtype=float) ** 2)
        var_out = var_in * self.cfg.outlier_scale
        eps = self.cfg.outlier_prob
        
        log_in = np.log(1.0 - eps) + _log_gaussian_diag(y, mu, var_in)
        log_out = np.log(eps) + _log_gaussian_diag(y, mu, var_out)
        return _logsumexp(np.array([log_in, log_out], dtype=float))