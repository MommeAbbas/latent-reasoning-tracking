import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum

class Mode(IntEnum):
    NORMAL = 0
    INSIGHT = 1
    BACKTRACK = 2

@dataclass
class SLDSConfig:
    state_dim: int = 5
    gamma: float = 0.1
    delta: float = 0.05
    eta: float = 0.05
    
    # Base drift vectors
    insight_drift: np.ndarray = field(default_factory=lambda: np.array([0.0]*5))
    backtrack_drift: np.ndarray = field(default_factory=lambda: np.array([0.0]*5))

    # Impulses (set to zero to prevent EKF seeing "jumps")
    insight_impulse: np.ndarray = field(default_factory=lambda: np.zeros(5))
    backtrack_impulse: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    # Transition Matrix
    P: np.ndarray = field(default_factory=lambda: np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ]))
    
    noise_std: tuple = (0.01, 0.01, 0.01, 0.01, 0.01)
    clip_state: bool = True

class SLDSDynamics:
    def __init__(self, config: SLDSConfig = SLDSConfig()):
        self.cfg = config

    def drift(self, x: np.ndarray) -> np.ndarray:

        dx = np.zeros(self.cfg.state_dim, dtype=float)
        p, c, u, f, m = x[0], x[1], x[2], x[3], x[4]

        # Progress naturally decays
        dx[0] = -0.04  

        # Other states evolve normally
        dx[1] = self.cfg.gamma * (1.0 - u) - self.cfg.delta * c
        dx[2] = self.cfg.eta * (1.0 - c) - 0.1 * f
        dx[3] = 0.02 + 0.05 * u
        dx[4] = 0.0
        return dx

    def mode_drift(self, z: Mode, x: np.ndarray) -> np.ndarray:
        dx = np.zeros(self.cfg.state_dim, dtype=float)
        
        if z == Mode.INSIGHT:
            # Insight boosts Progress (0)
            # But it does NOT boost Coherence(1) or Uncertainty(2)
            # To the EKF, x[1] and x[2] look "Normal" (no boost)
            dx[0] = 0.25 
            
        elif z == Mode.BACKTRACK:
            dx[0] = -0.05
            dx[1] = -0.05
            dx[2] = 0.05
            dx[3] = 0.05

        return dx

    def _get_mode_rotation(self, z: Mode) -> np.ndarray:

        R = np.eye(self.cfg.state_dim)
        theta = 0.0
        
        if z == Mode.INSIGHT:
            theta = 0.30
        elif z == Mode.BACKTRACK:
            theta = -0.30
            
        if theta != 0.0:
            # Rotate INDICES 1 and 2 (Coherence & Uncertainty)
            i, j = 1, 2
            c, s = np.cos(theta), np.sin(theta)
            R[i, i] = c
            R[i, j] = -s
            R[j, i] = s
            R[j, j] = c
            
        return R

    def mode_impulse(self, z: Mode) -> np.ndarray:
        if z == Mode.INSIGHT: return self.cfg.insight_impulse.copy()
        if z == Mode.BACKTRACK: return self.cfg.backtrack_impulse.copy()
        return np.zeros(self.cfg.state_dim, dtype=float)

    def sample_next_mode(self, z: Mode) -> Mode:
        probs = self.cfg.P[int(z)]
        z_next = np.random.choice([Mode.NORMAL, Mode.INSIGHT, Mode.BACKTRACK], p=probs)
        return Mode(int(z_next))

    def transition_mean(self, x: np.ndarray, z_next: Mode) -> np.ndarray:
        x_drift = x + self.drift(x)
        R = self._get_mode_rotation(z_next)
        x_rot = R @ x_drift
        x_next = x_rot + self.mode_impulse(z_next) + self.mode_drift(z_next, x)
        return x_next

    def step(self, x: np.ndarray, z: Mode):
        z_next = self.sample_next_mode(z)
        x_next = self.transition_mean(x, z_next)
        noise = np.random.randn(self.cfg.state_dim) * np.array(self.cfg.noise_std)
        x_next = x_next + noise
        if self.cfg.clip_state:
            x_next = np.clip(x_next, 0.0, 1.0)
        return x_next, z_next

# ---------------------------------------------------------------------------
# Ablation factory functions
# ---------------------------------------------------------------------------

def make_no_rotation_dynamics(base_cfg: SLDSConfig = None) -> "SLDSDynamics":
    """
    Returns an SLDSDynamics where _get_mode_rotation always returns the
    identity matrix, making INSIGHT and BACKTRACK structurally indistinguishable
    in the observation space (ablates the rotation component).
    """
    cfg = base_cfg if base_cfg is not None else SLDSConfig()

    class _NoRotDynamics(SLDSDynamics):
        def _get_mode_rotation(self, z: Mode) -> np.ndarray:
            return np.eye(self.cfg.state_dim)

    return _NoRotDynamics(cfg)


def make_two_mode_dynamics(base_cfg: SLDSConfig = None) -> "SLDSDynamics":
    """
    Returns an SLDSDynamics that only uses NORMAL and INSIGHT modes.
    BACKTRACK transitions are redirected to NORMAL, effectively removing
    the backtracking mode from the system.
    """
    import copy
    cfg = copy.deepcopy(base_cfg) if base_cfg is not None else SLDSConfig()

    # Redirect all BACKTRACK probability mass to NORMAL
    # Row 2 (from BACKTRACK): goes to NORMAL with p=1
    cfg.P[2, :] = [1.0, 0.0, 0.0]
    # Column 2 (to BACKTRACK): move that mass to NORMAL
    cfg.P[:, 0] += cfg.P[:, 2]
    cfg.P[:, 2]  = 0.0
    # Re-normalise each row
    cfg.P = cfg.P / cfg.P.sum(axis=1, keepdims=True)

    class _TwoModeDynamics(SLDSDynamics):
        def mode_drift(self, z: Mode, x: np.ndarray) -> np.ndarray:
            dx = np.zeros(self.cfg.state_dim, dtype=float)
            if z == Mode.INSIGHT:
                dx[0] = 0.25
            # BACKTRACK treated as NORMAL (no special drift)
            return dx

        def _get_mode_rotation(self, z: Mode) -> np.ndarray:
            R = np.eye(self.cfg.state_dim)
            if z == Mode.INSIGHT:
                theta = 0.30
                i, j = 1, 2
                c, s = np.cos(theta), np.sin(theta)
                R[i, i] = c; R[i, j] = -s
                R[j, i] = s; R[j, j] =  c
            return R

    return _TwoModeDynamics(cfg)


class SLDSSimulator:
    def __init__(self, dynamics: SLDSDynamics):
        self.dyn = dynamics

    def run(self, T: int, x0=None, z0: Mode = Mode.NORMAL):
        d = self.dyn.cfg.state_dim
        if x0 is None:
            x = np.zeros(d)
        else:
            x = np.array(x0, dtype=float)
        z = Mode(int(z0))
        xs, zs = np.zeros((T + 1, d)), np.zeros((T + 1,), dtype=int)
        xs[0], zs[0] = x, int(z)
        for k in range(T):
            x, z = self.dyn.step(x, z)
            xs[k + 1], zs[k + 1] = x, int(z)
        return xs, zs