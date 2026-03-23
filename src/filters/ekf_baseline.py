import numpy as np
from dataclasses import dataclass

from src.simulation.slds import SLDSDynamics
from src.simulation.sensors import ReasoningSensors


def numerical_jacobian(f, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x), dtype=float)
    m = y0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        yi = np.asarray(f(xp), dtype=float)
        J[:, i] = (yi - y0) / eps
    return J


@dataclass
class EKFConfig:
    init_mean: tuple = (0.2, 0.5, 0.8)
    init_cov_diag: tuple = (0.02, 0.02, 0.02, 0.05, 0.06)


class EKFBaseline:
    """
    EKF baseline continuous-only nonlinear state estimation
    Uses SLDS drift only (no switching modes or impulses)
    Observation model is the same ReasoningSensors.h(x)
    """

    def __init__(self, dyn: SLDSDynamics, sensors: ReasoningSensors, cfg: EKFConfig = EKFConfig()):
        self.dyn = dyn
        self.sensors = sensors
        self.cfg = cfg

        self.d = int(self.dyn.cfg.state_dim)

        self.mu = np.zeros(self.d, dtype=float)
        self.mu[:3] = np.array(cfg.init_mean, dtype=float)
        if self.d > 3:
            self.mu[3] = 0.15
        if self.d > 4:
            self.mu[4] = 0.2
        
        self.Sigma = np.zeros((self.d, self.d), dtype=float)
        init_cov = np.array(cfg.init_cov_diag, dtype=float) ** 2
        self.Sigma[: len(init_cov), : len(init_cov)] = np.diag(init_cov)

        Q = np.zeros((self.d, self.d), dtype=float)
        q_diag = np.array(self.dyn.cfg.noise_std, dtype=float) ** 2
        Q[: len(q_diag), : len(q_diag)] = np.diag(q_diag)
        self.Q = Q

        r_diag = np.array(self.sensors.cfg.noise_std, dtype=float) ** 2
        self.R = np.diag(r_diag)

    def predict(self):
        # f(x) = x + drift(x)
        def f(x):
            return x + self.dyn.drift(x)

        mu_pred = f(self.mu)
        F = numerical_jacobian(f, self.mu)
        Sigma_pred = F @ self.Sigma @ F.T + self.Q

        if self.dyn.cfg.clip_state:
            mu_pred = np.clip(mu_pred, 0.0, 1.0)

        self.mu = mu_pred
        self.Sigma = Sigma_pred

    def update(self, y):
        y = np.asarray(y, dtype=float)

        def h(x):
            return self.sensors.h(x)

        y_pred = h(self.mu)
        H = numerical_jacobian(h, self.mu)

        S = H @ self.Sigma @ H.T + self.R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ (y - y_pred)
        self.Sigma = (np.eye(self.d) - K @ H) @ self.Sigma

        if self.dyn.cfg.clip_state:
            self.mu = np.clip(self.mu, 0.0, 1.0)

    def step(self, y):
        self.predict()
        self.update(y)
        return self.mu, self.Sigma
