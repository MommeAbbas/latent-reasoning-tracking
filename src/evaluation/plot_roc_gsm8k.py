"""
Generates the ROC curve figure from actual filter runs.

Runs the simulation evaluation across N_SEEDS seeds, computes per-seed ROC
curves, then plots mean ± std confidence bands.  All AUC numbers printed here
are the values that should appear in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.evaluation.run_simulation_eval import run_eval

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "text.usetex": False,
})

N_SEEDS  = 3     # increase to 5-10 on Colab for tighter confidence bands
N_TRAJ   = 50    # increase to 150-200 on Colab
T        = 35
EARLY_K  = 4     # 0-indexed step for early-prediction AUC

BASE_FPR = np.linspace(0, 1, 101)


def collect_roc_curves(seeds, step: int):
    """
    For each seed, run the eval and compute ROC from predictions at `step`.
    Returns (tprs_rbpf, tprs_ekf, tprs_pf, aucs_rbpf, aucs_ekf, aucs_pf).
    Each tprs_* is shape (len(seeds), 101).
    """
    tprs_rbpf, tprs_ekf, tprs_pf = [], [], []
    aucs_rbpf, aucs_ekf, aucs_pf = [], [], []

    for seed in seeds:
        print(f"  Running seed {seed} ...")
        preds_rbpf, preds_ekf, preds_pf, labels = run_eval(
            seed=seed, N_traj=N_TRAJ, T=T, verbose=False
        )

        # Guard: skip seed if only one class present
        if len(np.unique(labels)) < 2:
            print(f"  [seed={seed}] Only one class — skipping.")
            continue

        scores_rbpf = preds_rbpf[:, step]
        scores_ekf  = preds_ekf[:, step]
        scores_pf   = preds_pf[:, step]

        for scores, tprs_list, aucs_list in [
            (scores_rbpf, tprs_rbpf, aucs_rbpf),
            (scores_ekf,  tprs_ekf,  aucs_ekf),
            (scores_pf,   tprs_pf,   aucs_pf),
        ]:
            fpr, tpr, _ = roc_curve(labels, scores)
            tpr_interp = np.interp(BASE_FPR, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs_list.append(tpr_interp)
            aucs_list.append(auc(BASE_FPR, tpr_interp))

    return (
        np.array(tprs_rbpf), np.array(tprs_ekf), np.array(tprs_pf),
        np.array(aucs_rbpf), np.array(aucs_ekf), np.array(aucs_pf),
    )


def plot_with_bands(ax, tprs, aucs, label_prefix, color, linestyle="-"):
    mean_tpr = tprs.mean(axis=0)
    std_tpr  = tprs.std(axis=0)
    mean_auc = aucs.mean()
    std_auc  = aucs.std()

    label = f"{label_prefix} AUC={mean_auc:.2f}±{std_auc:.2f}"
    ax.plot(BASE_FPR, mean_tpr, color=color, lw=2, ls=linestyle, label=label)
    ax.fill_between(
        BASE_FPR,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color=color, alpha=0.2,
    )
    return mean_auc, std_auc


def plot_publication_quality():
    seeds = list(range(N_SEEDS))
    step  = T - 1   # final-step predictions

    print(f"Collecting ROC data: {N_SEEDS} seeds × {N_TRAJ} trajectories (step={step}) ...")
    tprs_rbpf, tprs_ekf, tprs_pf, aucs_rbpf, aucs_ekf, aucs_pf = collect_roc_curves(seeds, step)

    print("\n=== ROC results (final-step predictions) ===")
    print(f"RBPF  AUC = {aucs_rbpf.mean():.4f} ± {aucs_rbpf.std():.4f}")
    print(f"PF    AUC = {aucs_pf.mean():.4f}   ± {aucs_pf.std():.4f}")
    print(f"EKF   AUC = {aucs_ekf.mean():.4f}  ± {aucs_ekf.std():.4f}")

    # Also report early-step AUC
    print(f"\nCollecting early-step (k={EARLY_K}) AUC ...")
    _, _, _, aucs_rbpf_e, aucs_ekf_e, aucs_pf_e = collect_roc_curves(seeds, EARLY_K)
    print(f"\n=== ROC results (early-step k={EARLY_K}) ===")
    print(f"RBPF  AUC = {aucs_rbpf_e.mean():.4f} ± {aucs_rbpf_e.std():.4f}")
    print(f"PF    AUC = {aucs_pf_e.mean():.4f}   ± {aucs_pf_e.std():.4f}")
    print(f"EKF   AUC = {aucs_ekf_e.mean():.4f}  ± {aucs_ekf_e.std():.4f}")

    # Plot final-step ROC
    fig, ax = plt.subplots(figsize=(6, 5))

    plot_with_bands(ax, tprs_rbpf, aucs_rbpf, "RBPF",  "#004488", "-")
    plot_with_bands(ax, tprs_pf,   aucs_pf,   "PF",    "#DDAA33", "--")
    plot_with_bands(ax, tprs_ekf,  aucs_ekf,  "EKF",   "#BB5566", "-.")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Early Success Prediction (SLDS Simulation)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="white")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_roc_early_prediction.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    plot_publication_quality()
