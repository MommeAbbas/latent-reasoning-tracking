# Loads real LLM traces saved by gsm8k_llm_runner.py.
# Same interface as GSM8KLoader so the eval loop works unchanged.

import os
import numpy as np
from typing import Tuple

_DATA_DIR = os.path.dirname(__file__)


class RealDataLoader:

    def __init__(
        self,
        data_dir: str = _DATA_DIR,
        split: str = "eval",
        n_eval: int = 200,
    ):
        traces_path  = os.path.join(data_dir, "real_traces.npy")
        labels_path  = os.path.join(data_dir, "real_labels.npy")
        lengths_path = os.path.join(data_dir, "real_trace_lengths.npy")

        for p in (traces_path, labels_path, lengths_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found. Run `python -m src.data.gsm8k_llm_runner` first."
                )

        traces  = np.load(traces_path)
        labels  = np.load(labels_path)
        lengths = np.load(lengths_path)

        N = len(labels)
        if split == "eval":
            idx = slice(0, min(n_eval, N))
        elif split == "held":
            idx = slice(min(n_eval, N), N)
        elif split == "all":
            idx = slice(0, N)
        else:
            raise ValueError(f"unknown split '{split}'")

        self.traces  = traces[idx]
        self.labels  = labels[idx]
        self.lengths = lengths[idx]

    def __len__(self) -> int:
        return len(self.labels)

    def get_trajectory(self, idx: int) -> Tuple[np.ndarray, int]:
        T   = int(self.lengths[idx])
        ys  = self.traces[idx, :T, :].astype(np.float64)
        lbl = int(self.labels[idx])
        return ys, lbl

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_trajectory(i)
