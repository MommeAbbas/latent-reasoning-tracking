import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from datasets import load_dataset

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator, Mode
from src.simulation.sensors import SensorConfig, ReasoningSensors


@dataclass
class GSM8KSimConfig:
    min_steps: int = 2
    max_steps: int = 40
    progress_threshold: float = 0.5
    
    # Both winners and losers start here and outcome is determined by the dynamics
    x0_start: Tuple[float, float, float, float, float] = (0.2, 0.5, 0.8, 0.2, 0.1)


class GSM8KLoader:
    def __init__(
        self,
        config_name: str = "main",
        split: str = "test",
        seed: int = 42,
        sim_cfg: GSM8KSimConfig = GSM8KSimConfig(),
        slds_cfg: Optional[SLDSConfig] = None,
        sensor_cfg: Optional[SensorConfig] = None,
    ):
        print(f"[GSM8K] Loading config='{config_name}', split='{split}'")
        self.dataset = load_dataset("openai/gsm8k", config_name, split=split)
        self.rng = np.random.default_rng(seed)
        self.sim_cfg = sim_cfg

        if slds_cfg is None:
            slds_cfg = SLDSConfig(
                state_dim=5,
                noise_std=(0.02, 0.02, 0.02, 0.01, 0.01),
                P=np.array([
                    [0.90, 0.05, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                ]),
                # Silent insight (rotation only, no impulse jump)
                insight_impulse=np.zeros(5),
                backtrack_impulse=np.zeros(5),
            )

        if sensor_cfg is None:
            sensor_cfg = SensorConfig(
                obs_dim=3,
                # High noise prevents EKF from tracking micro-drifts
                noise_std=(0.08, 0.08, 0.08), 
                outlier_prob=0.05,
                outlier_scale=20.0,
            )

        self.dyn = SLDSDynamics(slds_cfg)
        self.sensors = ReasoningSensors(sensor_cfg)

    def __len__(self):
        return len(self.dataset)

    def _extract_T(self, answer_text: str) -> int:
        reasoning_part = answer_text.split("####")[0]
        steps = [s.strip() for s in reasoning_part.split("\n") if s.strip()]
        T = len(steps)
        return max(self.sim_cfg.min_steps, min(self.sim_cfg.max_steps, T))

    def get_trajectory(self, idx: int) -> Tuple[np.ndarray, int]:
        item = self.dataset[idx]
        T = self._extract_T(item["answer"])

        # EKF sees this and thinks "I have no idea who will win"
        x0 = np.array(self.sim_cfg.x0_start, dtype=float)

        # Decides if we get insights (win) or not (fail)
        sim = SLDSSimulator(self.dyn)
        xs, zs = sim.run(T=T, x0=x0, z0=Mode.NORMAL)

        # Generate observations
        ys = np.zeros((T, self.sensors.cfg.obs_dim), dtype=float)
        for t in range(1, T + 1):
            ys[t - 1] = self.sensors.observe(xs[t])

        # Label is 1 if final progress > threshold, else 0
        label = int(xs[-1, 0] > self.sim_cfg.progress_threshold)

        return ys, label