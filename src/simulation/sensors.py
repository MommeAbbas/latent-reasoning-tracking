import numpy as np
from dataclasses import dataclass


def _logsumexp(a: np.ndarray) -> float:
    """Stable logsumexp for a 1D array."""
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


def _log_gaussian_diag(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """
    Log N(y; mu, diag(var)) for diagonal covariance.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)

    if y.shape != mu.shape or y.shape != var.shape:
        raise ValueError("y, mu, var must have the same shape")

    # log N = -0.5 * [sum(log(2pi*var)) + sum((y-mu)^2/var)]
    return float(
        -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(((y - mu) ** 2) / var))
    )


@dataclass
class SensorConfig:
    obs_dim: int = 3

    # Base (inlier) Gaussian noise std per sensor dimension
    noise_std: tuple = (0.05, 0.05, 0.05)

    # Robust mixture likelihood parameters
    outlier_prob: float = 0.05          # epsilon
    outlier_scale: float = 10.0         # variance multiplier for outliers (>= 1)

    clip_observation: bool = False      # optional, usually False


class ReasoningSensors:
    """
    Observation model for latent reasoning state x (dumension d).

    Deterministic observation h(x):
        y1 = p
        y2 = c - u
        y3 = u^2

    Only the first three latent components are observed.
    Additional latent dimensions (if any) are hidden.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        self.cfg = config
        
        if self.cfg.obs_dim != 3:
            raise ValueError("This sensor model assumes obs_dim = 3.")
        
        if len(self.cfg.noise_std) != self.cfg.obs_dim:
            raise ValueError(
                f"noise_std must have length {self.cfg.obs_dim}, "
                f"got {len(self.cfg.noise_std)}"
            )

        if not (0.0 <= self.cfg.outlier_prob < 1.0):
            raise ValueError("outlier_prob must be in [0, 1).")
        
        if self.cfg.outlier_scale < 1.0:
            raise ValueError("outlier_scale must be >= 1.0.")

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Deterministic sensor mapping h(x).
        
        Uses only the first three latent dimensions:
            x[0] = p
            x[1] = c
            x[2] = u
        """
        if x.shape[0] < 3:
            raise ValueError("Latent state x must have dimension >= 3.")

        p, c, u = x[0], x[1], x[2]
        return np.array([p, c - u, u ** 2], dtype=float)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """
        Sample an observation y ~ p(y|x) using the mixture noise model.
        This is for simulation / synthetic experiments.
        """
        mu = self.h(x)
        std_in = np.array(self.cfg.noise_std, dtype=float)

        # Choose inlier vs outlier component
        if np.random.rand() < self.cfg.outlier_prob:
            std = std_in * np.sqrt(self.cfg.outlier_scale)
        else:
            std = std_in

        y = mu + np.random.randn(self.cfg.obs_dim) * std

        if self.cfg.clip_observation:
            y = np.clip(y, 0.0, 1.0)

        return y

    def log_likelihood(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Compute log p(y | x) under the robust mixture likelihood.

        p(y|x) = (1-eps) N(y; h(x), R_in) + eps N(y; h(x), R_out)
        with diagonal covariances.
        """
        mu = self.h(x)
        var_in = (np.array(self.cfg.noise_std, dtype=float) ** 2)
        var_out = var_in * self.cfg.outlier_scale

        eps = self.cfg.outlier_prob

        log_in = np.log(1.0 - eps) + _log_gaussian_diag(y, mu, var_in)
        log_out = np.log(eps) + _log_gaussian_diag(y, mu, var_out)

        return _logsumexp(np.array([log_in, log_out], dtype=float))