"""
Mode posterior heatmap.

Runs the RBPF over all trajectories in the GSM8K eval set, collects per-step
mode posteriors, then plots:
  - Left  : mean P(mode | step, correct)   — correctly-solved problems
  - Right : mean P(mode | step, incorrect) — incorrectly-solved problems

Rows = mode (Normal / Insight / Backtrack)
Cols = normalised time step (0 … N_BINS-1)

Saved to figures/fig_mode_heatmap.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.simulation.slds import SLDSConfig, SLDSDynamics
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.data.real_data_loader import RealDataLoader

# ── config ────────────────────────────────────────────────────────────────────
N_PARTICLES = 125
N_BINS      = 12       # number of normalised time bins
MODE_NAMES  = ["Normal", "Insight", "Backtrack"]


def _build_dynamics_sensors():
    d_cfg = SLDSConfig()
    d_cfg.P = np.array([[0.96, 0.02, 0.02],
                        [0.05, 0.90, 0.05],
                        [0.05, 0.05, 0.90]])
    d_cfg.noise_std = (0.015, 0.015, 0.015, 0.01, 0.01)
    dyn = SLDSDynamics(d_cfg)

    s_cfg = SensorConfig()
    s_cfg.noise_std     = (0.10, 0.10, 0.10)
    s_cfg.outlier_prob  = 0.05
    s_cfg.outlier_scale = 20.0
    sensors = ReasoningSensors(s_cfg)
    return dyn, sensors


def _collect_posteriors(loader, dyn, sensors):
    """
    Returns two arrays of shape (N_problems, N_BINS, 3):
      correct_bins[i, b, m]   — mean mode posterior for correct problem i, bin b, mode m
      incorrect_bins[i, b, m] — same for incorrect problems
    """
    correct_post, incorrect_post = [], []

    for i in range(len(loader)):
        ys, label = loader.get_trajectory(i)
        T = len(ys)
        if T < 2:
            continue

        rbpf = RBPF_SLDS(dyn=dyn, sensors=sensors,
                         cfg=RBPFConfig(num_particles=N_PARTICLES))
        step_posts = []
        for y in ys:
            _, mode_probs = rbpf.step(y)
            step_posts.append(mode_probs.copy())

        step_posts = np.array(step_posts)   # (T, 3)

        # bin into N_BINS equally-sized time windows
        binned = np.zeros((N_BINS, 3), dtype=float)
        for b in range(N_BINS):
            t_start = int(b * T / N_BINS)
            t_end   = int((b + 1) * T / N_BINS)
            t_end   = max(t_end, t_start + 1)
            binned[b] = step_posts[t_start:t_end].mean(axis=0)

        if label == 1:
            correct_post.append(binned)
        else:
            incorrect_post.append(binned)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] correct={len(correct_post)}  incorrect={len(incorrect_post)}")

    return np.array(correct_post), np.array(incorrect_post)


def _draw_heatmap(ax, grid, title, vmin, vmax):
    """grid: (3, N_BINS)"""
    im = ax.imshow(
        grid,
        aspect="auto",
        vmin=vmin, vmax=vmax,
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(MODE_NAMES, fontsize=9)
    ax.set_xticks(np.arange(0, N_BINS, max(1, N_BINS // 6)))
    ax.set_xticklabels(
        [f"{int(b * 100 / N_BINS)}%" for b in np.arange(0, N_BINS, max(1, N_BINS // 6))],
        fontsize=8,
    )
    ax.set_xlabel("Relative step (%)")
    ax.set_title(title, fontweight="bold")
    return im


def main():
    np.random.seed(42)
    print("[plot_mode_heatmap] Loading data ...")
    loader = RealDataLoader(split="eval", n_eval=200)
    dyn, sensors = _build_dynamics_sensors()

    print(f"[plot_mode_heatmap] Running RBPF on {len(loader)} trajectories ...")
    correct_post, incorrect_post = _collect_posteriors(loader, dyn, sensors)

    if len(correct_post) == 0 or len(incorrect_post) == 0:
        print("[plot_mode_heatmap] Not enough examples of both classes. Aborting.")
        return

    print(f"  Correct  : {len(correct_post)} problems")
    print(f"  Incorrect: {len(incorrect_post)} problems")

    mean_c = correct_post.mean(axis=0).T    # (3, N_BINS)
    mean_w = incorrect_post.mean(axis=0).T  # (3, N_BINS)

    vmin = min(mean_c.min(), mean_w.min())
    vmax = max(mean_c.max(), mean_w.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    im_c = _draw_heatmap(axes[0], mean_c, "Correct solutions",   vmin, vmax)
    im_w = _draw_heatmap(axes[1], mean_w, "Incorrect solutions", vmin, vmax)

    # shared colorbar
    cbar = fig.colorbar(im_w, ax=axes, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Mean mode posterior", fontsize=9)

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/fig_mode_heatmap.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"[plot_mode_heatmap] Saved -> {out_path}")


if __name__ == "__main__":
    main()
