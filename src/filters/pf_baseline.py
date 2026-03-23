import numpy as np
from dataclasses import dataclass
from typing import Tuple

from src.simulation.slds import SLDSDynamics, Mode
from src.simulation.sensors import ReasoningSensors


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


@dataclass
class PFConfig:
    num_particles: int = 2000
    resample_threshold: float = 0.5
    init_mean: Tuple[float, float, float] = (0.2, 0.5, 0.8)
    init_std: Tuple[float, float, float, float, float] = (0.05, 0.05, 0.05, 0.08, 0.1)


class PF_SLDS:
    """
    Baseline Particle Filter for SLDS
    Particles carry discrete mode z and continuous state x (dimension d)

    """

    def __init__(self, dyn: SLDSDynamics, sensors: ReasoningSensors, cfg: PFConfig = PFConfig()):
        self.dyn = dyn
        self.sensors = sensors
        self.cfg = cfg
        self.N = cfg.num_particles

        d = self.dyn.cfg.state_dim

        self.weights = np.ones(self.N, dtype=float) / self.N
        self.modes = np.full(self.N, int(Mode.NORMAL), dtype=int)

        mean = np.zeros(d, dtype=float)
        std = np.zeros(d, dtype=float)

        mean[:3] = np.array(cfg.init_mean, dtype=float)
        std[:3] = np.array(cfg.init_std[:3], dtype=float)
        if d > 3:
            mean[3] = 0.15
            std[3] = cfg.init_std[3]
        if d > 4:
            mean[4] = 0.2
            std[4] = cfg.init_std[4]
        
        self.xs = mean + np.random.randn(self.N, d) * std

        if self.dyn.cfg.clip_state:
            self.xs = np.clip(self.xs, 0.0, 1.0)

    def effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def step(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        d = self.dyn.cfg.state_dim

        # sample next modes
        new_modes = np.zeros_like(self.modes)
        for i in range(self.N):
            z = Mode(int(self.modes[i]))
            z_next = self.dyn.sample_next_mode(z)
            new_modes[i] = int(z_next)

        # propagate continuous states using the true simulator dynamics step (but without sampling z inside)
        new_xs = np.zeros_like(self.xs)
        logw = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            z_next = Mode(int(new_modes[i]))

            x = self.xs[i]
            x_pred = self.dyn.transition_mean(x, z_next)

            noise = np.random.randn(d) * np.array(self.dyn.cfg.noise_std, dtype=float)
            x_pred = x_pred + noise

            if self.dyn.cfg.clip_state:
                x_pred = np.clip(x_pred, 0.0, 1.0)

            new_xs[i] = x_pred

            # likelihood weight
            logw[i] = self.sensors.log_likelihood(y, x_pred)

        # normalize weights
        logw = logw - np.max(logw)
        w = np.exp(logw) * self.weights
        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.ones(self.N, dtype=float) / self.N
        else:
            w = w / w_sum

        self.weights = w
        self.modes = new_modes
        self.xs = new_xs

        # resample
        ess = self.effective_sample_size()
        if ess / self.N < self.cfg.resample_threshold:
            idx = systematic_resample(self.weights)
            self.weights = np.ones(self.N, dtype=float) / self.N
            self.modes = self.modes[idx]
            self.xs = self.xs[idx]

        return self.estimate()

    def estimate(self):
        x_hat = np.average(self.xs, axis=0, weights=self.weights)
        mode_probs = np.zeros(3, dtype=float)
        for i in range(self.N):
            mode_probs[self.modes[i]] += self.weights[i]
        return x_hat, mode_probs
