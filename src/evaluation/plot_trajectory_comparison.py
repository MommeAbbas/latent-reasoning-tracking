"""
Dual-panel trajectory comparison figure.

Left panel : one correctly-solved problem (label=1)
Right panel: one incorrectly-solved problem (label=0)

Each panel shows:
  Top    — stacked area of mode posteriors P(Normal/Insight/Backtrack) over steps
  Bottom — RBPF progress estimate p_t with success threshold

Saved to figures/fig_trajectory_comparison.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.simulation.slds import SLDSConfig, SLDSDynamics
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.evaluation.online_prediction import OnlineCorrectnessPredictor
from src.data.real_data_loader import RealDataLoader

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": False,
})

MODE_COLORS = {
    "Normal":    "#4477AA",
    "Insight":   "#228833",
    "Backtrack": "#CC3311",
}
PROGRESS_COLOR  = "#222222"
THRESHOLD_COLOR = "#888888"

N_PARTICLES = 200


def _build_dynamics_sensors():
    d_cfg = SLDSConfig()
    d_cfg.P = np.array([[0.96, 0.02, 0.02],
                        [0.05, 0.90, 0.05],
                        [0.05, 0.05, 0.90]])
    d_cfg.noise_std = (0.015, 0.015, 0.015, 0.01, 0.01)
    dyn = SLDSDynamics(d_cfg)

    s_cfg = SensorConfig()
    s_cfg.noise_std    = (0.10, 0.10, 0.10)
    s_cfg.outlier_prob  = 0.05
    s_cfg.outlier_scale = 20.0
    sensors = ReasoningSensors(s_cfg)
    return dyn, sensors


def _run_rbpf(ys, dyn, sensors):
    """Return (progress_list, mode_posteriors_list).
    mode_posteriors_list[t] = [P(Normal), P(Insight), P(Backtrack)]."""
    predictor = OnlineCorrectnessPredictor()
    rbpf = RBPF_SLDS(dyn=dyn, sensors=sensors,
                     cfg=RBPFConfig(num_particles=N_PARTICLES))
    progress, posteriors = [], []
    for y in ys:
        x_hat, mode_probs = rbpf.step(y)
        progress.append(predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights))
        posteriors.append(mode_probs.copy())
    return np.array(progress), np.array(posteriors)


def _find_examples(loader, rng, min_len=6):
    """Return (correct_idx, incorrect_idx) with length >= min_len."""
    correct_idx = incorrect_idx = None
    perm = rng.permutation(len(loader))
    for idx in perm:
        ys, label = loader.get_trajectory(int(idx))
        if label == 1 and correct_idx is None and len(ys) >= min_len:
            correct_idx = int(idx)
        if label == 0 and incorrect_idx is None and len(ys) >= min_len:
            incorrect_idx = int(idx)
        if correct_idx is not None and incorrect_idx is not None:
            break
    if correct_idx is None or incorrect_idx is None:
        raise RuntimeError("Could not find suitable trajectories in dataset.")
    return correct_idx, incorrect_idx


def _draw_panel(axes_top, axes_bot, ys, progress, posteriors, title):
    steps = np.arange(len(progress))

    # ── mode posteriors (stacked area) ───────────────────────────────────────
    p_normal    = posteriors[:, 0]
    p_insight   = posteriors[:, 1]
    p_backtrack = posteriors[:, 2]

    axes_top.stackplot(
        steps,
        p_normal, p_insight, p_backtrack,
        labels=["Normal", "Insight", "Backtrack"],
        colors=[MODE_COLORS["Normal"], MODE_COLORS["Insight"], MODE_COLORS["Backtrack"]],
        alpha=0.75,
    )
    axes_top.set_ylim(0, 1)
    axes_top.set_ylabel("Mode posterior")
    axes_top.set_title(title, fontweight="bold")
    axes_top.set_xticks([])
    axes_top.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ── progress estimate ─────────────────────────────────────────────────────
    axes_bot.plot(steps, progress, color=PROGRESS_COLOR, lw=1.8, label="RBPF $\\hat{p}_t$")
    axes_bot.axhline(0.5, color=THRESHOLD_COLOR, lw=1.2, ls=":", label="Threshold 0.5")
    axes_bot.set_ylim(-0.05, 1.05)
    axes_bot.set_ylabel("P(correct)")
    axes_bot.set_xlabel("Reasoning step $t$")
    axes_bot.legend(loc="lower right" if progress[-1] > 0.5 else "upper right",
                    fontsize=8, framealpha=0.85)


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    print("[plot_trajectory_comparison] Loading data ...")
    loader = RealDataLoader(split="all")
    dyn, sensors = _build_dynamics_sensors()

    correct_idx, incorrect_idx = _find_examples(loader, rng)
    ys_c, _ = loader.get_trajectory(correct_idx)
    ys_w, _ = loader.get_trajectory(incorrect_idx)

    print(f"  Correct   problem idx={correct_idx}  T={len(ys_c)}")
    print(f"  Incorrect problem idx={incorrect_idx} T={len(ys_w)}")

    print("[plot_trajectory_comparison] Running RBPF ...")
    prog_c, post_c = _run_rbpf(ys_c, dyn, sensors)
    prog_w, post_w = _run_rbpf(ys_w, dyn, sensors)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [1.4, 1]},
        sharey="row",
    )

    _draw_panel(axes[0, 0], axes[1, 0], ys_c, prog_c, post_c,
                title="Correct solution")
    _draw_panel(axes[0, 1], axes[1, 1], ys_w, prog_w, post_w,
                title="Incorrect solution")

    plt.tight_layout(pad=1.5)

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fig_trajectory_comparison.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"[plot_trajectory_comparison] Saved -> {out_path}")


if __name__ == "__main__":
    main()
