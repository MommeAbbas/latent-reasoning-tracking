import numpy as np


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class OnlineCorrectnessPredictor:
    """
    Maps a belief over latent state x_k to a probability
    P(final answer is correct | y_{1:k})

    This class is intentionally simple and explicit
    """

    def __init__(self, w=None, b=0.0):

        if w is None:
            # [progress, coherence, uncertainty, fatigue, momentum]
            self.w = np.array([2.0, 1.5, -1.8, -2.2, 1.0], dtype=float)
        else:
            self.w = np.asarray(w, dtype=float)

        self.b = float(b)

    def prob_correct_from_state(self, x):
        """
        x: latent state vector (>=3,)
        returns scalar probability in (0,1)
        """
        x = x[: self.w.shape[0]]
        logit = self.w @ x + self.b
        return 1.0 / (1.0 + np.exp(-logit))

    def prob_correct_from_particles(self, mus, weights):
        """
        mus are array of shape (N, 3) of particle means
        weights are array of shape (N,) summing to 1
        Returns E_{p(x|y)} [ P(correct | x) ]
        """
        probs = np.array(
            [self.prob_correct_from_state(mu) for mu in mus],
            dtype=float,
        )
        return float(np.sum(weights * probs))


class OnlinePredictionRecorder:
    """
    Stores predictions over time for a single trajectory
    """

    def __init__(self):
        self.p_hats = []

    def update(self, p_hat):
        self.p_hats.append(float(p_hat))

    def as_array(self):
        return np.array(self.p_hats, dtype=float)
