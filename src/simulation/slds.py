import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum


class Mode(IntEnum):
    NORMAL = 0
    INSIGHT = 1
    BACKTRACK = 2


@dataclass
class SLDSConfig:
    # Continuous drift coefficients (baseline / "normal" regime)
    alpha: float = 0.1
    beta: float = 0.05
    gamma: float = 0.1
    delta: float = 0.05
    eta: float = 0.05
    lam: float = 0.2
    state_dim: int = 3

    # Mode-dependent additive impulses (event-like effects)
    # These are applied in addition to the smooth drift
    insight_impulse: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.10, -0.15]))
    backtrack_impulse: np.ndarray = field(default_factory=lambda: np.array([-0.10, -0.08, 0.12]))

    # Process noise
    noise_std: tuple = (0.01, 0.01, 0.01)

    # Markov transition matrix over modes (rows sum to 1)
    # Default: mostly stay in same mode, occasional switches
    P: np.ndarray = field(
        default_factory=lambda: np.array(
            # (NORMAL, INSIGHT, BACKTRACK)
            [
                [0.90, 0.05, 0.05],
                [0.60, 0.35, 0.05],
                [0.60, 0.05, 0.35],
            ],
            dtype=float,
        )
    )

    clip_state: bool = True


class SLDSDynamics:
    """
    Switching Latent Dynamical System (SLDS):
      z_k is a discrete mode evolving as a Markov chain
      x_k is a continuous latent state evolving with mode-dependent dynamics

    x = [progress, coherence, uncertainty] in [0, 1]^3 (by default with clipping).
    """

    def __init__(self, config: SLDSConfig = SLDSConfig()):
        self.cfg = config
        self._validate_P(self.cfg.P)
        self._validate_shapes()

    @staticmethod
    def _validate_P(P: np.ndarray):
        if P.shape != (3, 3):
            raise ValueError(f"Transition matrix P must be shape (3,3), got {P.shape}.")
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"Each row of P must sum to 1. Row sums: {row_sums}")
        if np.any(P < 0):
            raise ValueError("Transition matrix P must have nonnegative entries")

    def _validate_shapes(self):
        d = self.cfg.state_dim

        if len(self.cfg.noise_std) != d:
            raise ValueError(f"noise_std must have length {d}")

        if self.cfg.insight_impulse.shape != (d,):
            raise ValueError(f"insight_impulse must have shape ({d},)")

        if self.cfg.backtrack_impulse.shape != (d,):
            raise ValueError(f"backtrack_impulse must have shape ({d},)")
    
    def drift(self, x: np.ndarray) -> np.ndarray:
        """Smooth drift g(x) shared across modes"""
        p, c, u = x
        dp = self.cfg.alpha * c - self.cfg.beta * u
        dc = self.cfg.gamma * (1.0 - u) - self.cfg.delta * c
        du = self.cfg.eta * (1.0 - c) - self.cfg.lam * p
        return np.array([dp, dc, du], dtype=float)

    def mode_impulse(self, z: Mode) -> np.ndarray:
        """Mode-dependent impulse (discrete event effect)"""
        if z == Mode.INSIGHT:
            return self.cfg.insight_impulse
        if z == Mode.BACKTRACK:
            return self.cfg.backtrack_impulse
        return np.zeros(self.cfg.state_dim, dtype=float)

    def sample_next_mode(self, z: Mode) -> Mode:
        """Sample z_{k+1} ~ P(z_{k+1} | z_k)"""
        probs = self.cfg.P[int(z)]
        z_next = np.random.choice([Mode.NORMAL, Mode.INSIGHT, Mode.BACKTRACK], p=probs)
        return Mode(int(z_next))

    def step(self, x: np.ndarray, z: Mode):
        """
        One SLDS step:
          z_{k+1} ~ P(. | z_k)
          x_{k+1} = x_k + drift(x_k) + impulse(z_{k+1}) + noise
        Note: impulse is applied for the new mode (interpretable as entering that regime)
        """
        d = self.cfg.state_dim

        z_next = self.sample_next_mode(z)

        x_next = x + self.drift(x)
        x_next = x_next + self.mode_impulse(z_next)

        noise = np.random.randn(d) * np.array(self.cfg.noise_std, dtype=float)
        x_next = x_next + noise

        if self.cfg.clip_state:
            x_next = np.clip(x_next, 0.0, 1.0)

        return x_next, z_next


class SLDSSimulator:
    """Convenience wrapper to simulate (x_0:T, z_0:T)"""

    def __init__(self, dynamics: SLDSDynamics):
        self.dyn = dynamics

    def run(self, T: int, x0=None, z0: Mode = Mode.NORMAL):
        d = self.dyn.cfg.state_dim

        if x0 is None:
            x = np.array([0.2, 0.5, 0.8], dtype=float)
        else:
            x = np.array(x0, dtype=float)

        z = Mode(int(z0))

        xs = np.zeros((T + 1, d), dtype=float)
        zs = np.zeros((T + 1,), dtype=int)

        xs[0] = x
        zs[0] = int(z)

        for k in range(T):
            x, z = self.dyn.step(x, z)
            xs[k + 1] = x
            zs[k + 1] = int(z)

        return xs, zs
