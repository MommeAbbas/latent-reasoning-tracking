"""
Logistic Regression baseline for online success prediction.

Unlike the Bayesian filters, this is a *supervised* method trained on labelled
trajectories.  It serves as a strong reference point: if the RBPF beats LR at
early steps (k=5) without any training data, that demonstrates the value of
the structured prior.

Interface
---------
    lr = LRBaseline(k=5, train_frac=0.7)
    lr.fit(all_ys, all_labels)   # all_ys: list of (T_i, 3) arrays
    probs = lr.predict(all_ys)   # np.ndarray of shape (N,)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List


class LRBaseline:
    """
    Logistic regression on the first k observations (zero-padded).

    Parameters
    ----------
    k : int
        Number of observation steps to use as features.
    C : float
        Inverse regularisation strength for LogisticRegression.
    """

    def __init__(self, k: int = 5, C: float = 1.0):
        self.k   = k
        self.C   = C
        self.clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        self._obs_dim = None

    def _featurise(self, ys_list: List[np.ndarray]) -> np.ndarray:
        """Flatten first k steps of each trajectory into a feature vector."""
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
        """Returns predicted P(correct) for each trajectory."""
        X = self._featurise(ys_list)
        return self.clf.predict_proba(X)[:, 1]
