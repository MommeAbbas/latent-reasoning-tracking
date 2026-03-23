import numpy as np
from sklearn.metrics import roc_auc_score


def negative_log_likelihood(p_hat, y_true, eps=1e-12):
    """
    p_hat is array of predicted probabilities in (0,1)
    y_true is array of binary labels {0,1}
    """
    p_hat = np.clip(p_hat, eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=float)
    return -np.mean(
        y_true * np.log(p_hat) + (1.0 - y_true) * np.log(1.0 - p_hat)
    )


def brier_score(p_hat, y_true):
    p_hat = np.asarray(p_hat, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return np.mean((p_hat - y_true) ** 2)


def expected_calibration_error(p_hat, y_true, n_bins=10):
    p_hat = np.asarray(p_hat, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(p_hat)

    for i in range(n_bins):
        mask = (p_hat >= bins[i]) & (p_hat < bins[i + 1])
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(p_hat[mask])
            ece += (np.sum(mask) / N) * abs(acc - conf)

    return ece


def auc_by_prefix(predictions, labels):
    """
    predictions are list of arrays, one per prefix length k
    labels are binary labels per trajectory
    """
    aucs = []
    for p_k in predictions:
        if len(np.unique(labels)) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(labels, p_k))
    return np.array(aucs)