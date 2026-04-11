import numpy as np
from dataclasses import dataclass
from typing import Tuple

from src.simulation.slds import SLDSDynamics, Mode
from src.simulation.sensors import ReasoningSensors


def numerical_jacobian(f, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x), dtype=float)
    m, n = y0.size, x.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        J[:, i] = (np.asarray(f(xp), dtype=float) - y0) / eps
    return J


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    N = len(weights)
    positions = (np.random.rand() + np.arange(N)) / N
    cum = np.cumsum(weights)
    idx = np.zeros(N, dtype=int)
    i = j = 0
    while i < N:
        if positions[i] < cum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def log_gaussian_full(y: np.ndarray, mu: np.ndarray, S: np.ndarray) -> float:
    y, mu, S = np.asarray(y, dtype=float), np.asarray(mu, dtype=float), np.asarray(S, dtype=float)
    d = y.size
    L = np.linalg.cholesky(S)
    sol = np.linalg.solve(L, y - mu)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (d * np.log(2.0 * np.pi) + logdet + sol @ sol))


@dataclass
class RBPFConfig:
    num_particles: int = 200
    resample_threshold: float = 0.5
    init_mean: Tuple[float, float, float] = (0.2, 0.5, 0.8)
    init_cov_diag: Tuple[float, float, float, float, float] = (0.02, 0.02, 0.02, 0.05, 0.06)


class RBPF_SLDS:
    """RBPF for SLDS. Particles track discrete modes; EKF tracks continuous state per particle."""

    def __init__(self, dyn: SLDSDynamics, sensors: ReasoningSensors, cfg: RBPFConfig = RBPFConfig()):
        self.dyn = dyn
        self.sensors = sensors
        self.cfg = cfg

        self.N = cfg.num_particles
        self.d = int(self.dyn.cfg.state_dim)
        self.m = int(self.sensors.cfg.obs_dim)

        self.weights = np.ones(self.N, dtype=float) / self.N
        self.modes = np.full(self.N, int(Mode.NORMAL), dtype=int)

        init_mu = np.zeros(self.d, dtype=float)
        init_mu[:3] = np.array(cfg.init_mean, dtype=float)
        if self.d > 3:
            init_mu[3] = 0.15
        if self.d > 4:
            init_mu[4] = 0.2
        self.mus = np.tile(init_mu, (self.N, 1))

        self.Sigmas = np.zeros((self.N, self.d, self.d), dtype=float)
        init_cov = np.zeros((self.d, self.d), dtype=float)
        init_diag = np.array(cfg.init_cov_diag, dtype=float) ** 2
        init_cov[: len(init_diag), : len(init_diag)] = np.diag(init_diag)
        for i in range(self.N):
            self.Sigmas[i] = init_cov

        self.Q = np.diag(np.array(self.dyn.cfg.noise_std, dtype=float) ** 2)

    def effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def _predict_particle(self, mu, Sigma, z_next: Mode):
        def fz(x):
            return self.dyn.transition_mean(x, z_next)
        mu_pred = fz(mu)
        F = numerical_jacobian(fz, mu)
        Sigma_pred = F @ Sigma @ F.T + self.Q
        if self.dyn.cfg.clip_state:
            mu_pred = np.clip(mu_pred, 0.0, 1.0)
        return mu_pred, Sigma_pred

    def _update_particle(self, mu_pred, Sigma_pred, y):
        def hx(x):
            return self.sensors.h(x)
        y_pred = hx(mu_pred)
        H = numerical_jacobian(hx, mu_pred)
        R = np.diag((np.array(self.sensors.cfg.noise_std, dtype=float) ** 2))
        S = H @ Sigma_pred @ H.T + R
        K = Sigma_pred @ H.T @ np.linalg.inv(S)
        mu_upd = mu_pred + K @ (y - y_pred)
        Sigma_upd = (np.eye(self.d) - K @ H) @ Sigma_pred
        if self.dyn.cfg.clip_state:
            mu_upd = np.clip(mu_upd, 0.0, 1.0)
        return mu_upd, Sigma_upd, y_pred, S

    def step(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)

        new_modes = np.zeros_like(self.modes)
        for i in range(self.N):
            new_modes[i] = int(self.dyn.sample_next_mode(Mode(int(self.modes[i]))))

        new_mus = np.zeros_like(self.mus)
        new_Sigmas = np.zeros_like(self.Sigmas)
        logw = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            mu_pred, Sigma_pred = self._predict_particle(self.mus[i], self.Sigmas[i], Mode(int(new_modes[i])))
            mu_upd, Sigma_upd, y_pred, S = self._update_particle(mu_pred, Sigma_pred, y)
            new_mus[i] = mu_upd
            new_Sigmas[i] = Sigma_upd

            eps = max(self.sensors.cfg.outlier_prob, 1e-12)
            scale = self.sensors.cfg.outlier_scale
            log_in  = np.log(1.0 - eps) + log_gaussian_full(y, y_pred, S)
            log_out = np.log(eps)        + log_gaussian_full(y, y_pred, S * scale)
            m = max(log_in, log_out)
            logw[i] = m + np.log(np.exp(log_in - m) + np.exp(log_out - m))

        logw -= np.max(logw)
        w = np.exp(logw) * self.weights
        w_sum = np.sum(w)
        w = np.ones(self.N, dtype=float) / self.N if (w_sum <= 0 or not np.isfinite(w_sum)) else w / w_sum

        self.weights = w
        self.modes = new_modes
        self.mus = new_mus
        self.Sigmas = new_Sigmas

        if self.effective_sample_size() / self.N < self.cfg.resample_threshold:
            idx = systematic_resample(self.weights)
            self.weights = np.ones(self.N, dtype=float) / self.N
            self.modes   = self.modes[idx]
            self.mus     = self.mus[idx]
            self.Sigmas  = self.Sigmas[idx]

        return self.estimate()

    def estimate(self):
        x_hat = np.average(self.mus, axis=0, weights=self.weights)
        mode_probs = np.zeros(3, dtype=float)
        for i in range(self.N):
            mode_probs[self.modes[i]] += self.weights[i]
        return x_hat, mode_probs
