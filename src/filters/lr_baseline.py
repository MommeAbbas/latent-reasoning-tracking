# Supervised LR baseline. Trains on first k obs per trajectory.

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List


class LRBaseline:
    """LR on the first k observations, zero-padded to fixed length."""

    def __init__(self, k: int = 5, C: float = 1.0):
        self.k   = k
        self.C   = C
        self.clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        self._obs_dim = None

    def _featurise(self, ys_list: List[np.ndarray]) -> np.ndarray:
        obs_dim = ys_list[0].shape[1] if ys_list else 3
        self._obs_dim = obs_dim
        out = np.zeros((len(ys_list), self.k * obs_dim), dtype=float)
        for i, ys in enumerate(ys_list):
            n = min(self.k, len(ys))
            out[i, : n * obs_dim] = ys[:n].flatten()
        return out

    def fit(self, ys_list: List[np.ndarray], labels: np.ndarray) -> "LRBaseline":
        X = self._featurise(ys_list)
        y = np.asarray(labels, dtype=int)
        self.clf.fit(X, y)
        return self

    def predict(self, ys_list: List[np.ndarray]) -> np.ndarray:
        X = self._featurise(ys_list)
        return self.clf.predict_proba(X)[:, 1]
