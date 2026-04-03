"""
Extracts 3 scalar features per reasoning step from raw LLM text output.
No model weights required — pure string processing.

Feature map (matches the 3D observation space of ReasoningSensors):
  y[t, 0]  coherence proxy   — normalised step length
  y[t, 1]  uncertainty proxy — hedging word rate
  y[t, 2]  fatigue proxy     — bigram repetition rate vs. prior steps
"""

import re
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Hedging vocabulary used for the uncertainty proxy
# ---------------------------------------------------------------------------
_HEDGING_WORDS = {
    "wait", "actually", "hmm", "alternatively", "but", "however",
    "unless", "maybe", "perhaps", "reconsider", "unclear", "unsure",
    "doubt", "possibly", "might", "could", "uncertain", "mistake",
    "wrong", "oops",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _bigrams(tokens: List[str]):
    return list(zip(tokens[:-1], tokens[1:]))


class StepFeatureExtractor:
    """
    Takes a list of step strings (one per reasoning step) and returns a
    (T, 3) numpy array of normalised features.
    """

    def extract(self, steps: List[str]) -> np.ndarray:
        if not steps:
            return np.zeros((0, 3), dtype=float)

        token_lists = [_tokenize(s) for s in steps]
        lengths = [len(t) for t in token_lists]
        median_len = float(np.median(lengths)) if lengths else 1.0

        features = np.zeros((len(steps), 3), dtype=float)

        all_prior_bigrams: set = set()

        for t, (step, tokens) in enumerate(zip(steps, token_lists)):
            n = len(tokens) if tokens else 1

            # --- coherence proxy: normalised step length ---
            features[t, 0] = min(len(tokens) / (median_len + 1e-6), 2.0) / 2.0

            # --- uncertainty proxy: hedging word rate ---
            hedge_count = sum(1 for tok in tokens if tok in _HEDGING_WORDS)
            features[t, 1] = min(hedge_count / n, 1.0)

            # --- fatigue proxy: bigram repetition rate ---
            current_bigrams = set(_bigrams(tokens))
            if current_bigrams:
                repeated = current_bigrams & all_prior_bigrams
                features[t, 2] = len(repeated) / len(current_bigrams)
            else:
                features[t, 2] = 0.0

            all_prior_bigrams.update(current_bigrams)

        return features


class FeatureNormalizer:
    """
    Min-max normaliser fit on a collection of feature arrays.
    Ensures all features are in [0, 1] across the corpus.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, feature_arrays: List[np.ndarray]) -> "FeatureNormalizer":
        all_data = np.concatenate(feature_arrays, axis=0)   # (N_total, 3)
        self.min_ = all_data.min(axis=0)
        self.max_ = all_data.max(axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        denom = np.where(self.max_ - self.min_ > 1e-8, self.max_ - self.min_, 1.0)
        return np.clip((features - self.min_) / denom, 0.0, 1.0)

    def fit_transform(self, feature_arrays: List[np.ndarray]) -> List[np.ndarray]:
        self.fit(feature_arrays)
        return [self.transform(f) for f in feature_arrays]
