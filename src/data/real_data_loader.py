"""
Loader for the real LLM traces saved by gsm8k_llm_runner.py.

Yields (ys, label, T) tuples with the same interface as
GSM8KLoader.get_trajectory(), so the existing evaluation loop works unchanged.
"""

import os
import numpy as np
from typing import Tuple

_DATA_DIR = os.path.dirname(__file__)   # src/data/


class RealDataLoader:
    """
    Loads the pre-computed real LLM feature arrays and iterates over them.

    Parameters
    ----------
    data_dir : str
        Directory containing real_traces.npy, real_labels.npy,
        real_trace_lengths.npy.  Defaults to src/data/.
    split : str
        "eval"  — problems 0..n_eval-1   (default, used for filter evaluation)
        "held"  — problems n_eval..end   (held-out)
        "all"   — all problems
    n_eval : int
        Number of problems to treat as the eval split. Default 200.
    """

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
                    f"{p} not found.\n"
                    "Run `python -m src.data.gsm8k_llm_runner` first."
                )

        traces  = np.load(traces_path)           # (N, MAX_T, 3)
        labels  = np.load(labels_path)           # (N,)
        lengths = np.load(lengths_path)          # (N,)

        N = len(labels)
        if split == "eval":
            idx = slice(0, min(n_eval, N))
        elif split == "held":
            idx = slice(min(n_eval, N), N)
        elif split == "all":
            idx = slice(0, N)
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'eval', 'held', or 'all'.")

        self.traces  = traces[idx]
        self.labels  = labels[idx]
        self.lengths = lengths[idx]

    def __len__(self) -> int:
        return len(self.labels)

    def get_trajectory(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Returns (ys, label) where ys has shape (T, 3) — the actual steps only,
        without zero-padding.
        """
        T   = int(self.lengths[idx])
        ys  = self.traces[idx, :T, :].astype(np.float64)
        lbl = int(self.labels[idx])
        return ys, lbl

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_trajectory(i)
