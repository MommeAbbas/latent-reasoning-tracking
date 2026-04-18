"""
Results bar chart.

Reads tables/real_llm_results.csv (produced by run_real_eval.py) and plots
AUC@final with 95%-CI error bars for all methods, coloured by method type:
  - Dark blue : RBPF (ours)
  - Mid grey  : filter baselines (EKF, PF)
  - Orange    : supervised baselines (LR-k5, LR-full)
  - Green     : PRM

Also reads tables/ablation_results.csv for the simulation ablation inset (optional).

Saved to figures/fig_results_barchart.pdf
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ── colour scheme ─────────────────────────────────────────────────────────────
METHOD_STYLE = {
    "RBPF":        {"color": "#004488", "label": "RBPF (ours)",      "zorder": 5},
    "EKF":         {"color": "#888888", "label": "EKF",              "zorder": 3},
    "PF":          {"color": "#AAAAAA", "label": "PF",               "zorder": 3},
    "LR-k5":       {"color": "#EE7733", "label": "LR (k=5)",         "zorder": 4},
    "LR-full":     {"color": "#CC3311", "label": "LR-full",          "zorder": 4},
    "MeanEntropy": {"color": "#BBBBBB", "label": "Mean Entropy",      "zorder": 2},
    "PRM":         {"color": "#228833", "label": "PRM (supervised)", "zorder": 4},
}

# Display order (most interesting first)
DISPLAY_ORDER = ["RBPF", "PRM", "LR-full", "LR-k5", "EKF", "PF", "MeanEntropy"]


def _load_csv(path, metric="auc_final"):
    """Returns {method: list_of_values}."""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["method"]
            if metric not in row:
                continue
            val = float(row[metric])
            data.setdefault(m, []).append(val)
    return data


def _mean_ci(vals, z=1.96):
    arr = np.array(vals)
    n = len(arr)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, z * se


def plot_barchart(csv_path, out_path, title):
    data = _load_csv(csv_path, metric="auc_final")
    if not data:
        print(f"[plot_results_barchart] No data at {csv_path} — skipping.")
        return

    methods = [m for m in DISPLAY_ORDER if m in data]
    # append any methods in data but not in DISPLAY_ORDER
    for m in data:
        if m not in methods:
            methods.append(m)

    means, errs, colors, labels = [], [], [], []
    for m in methods:
        mu, ci = _mean_ci(data[m])
        means.append(mu)
        errs.append(ci)
        style = METHOD_STYLE.get(m, {"color": "#999999", "label": m})
        colors.append(style["color"])
        labels.append(style["label"])

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(x, means, color=colors, width=0.55,
                  yerr=errs, capsize=4,
                  error_kw={"elinewidth": 1.4, "ecolor": "#444444"})

    # chance line
    ax.axhline(0.5, color="#CC3311", lw=1.0, ls="--", alpha=0.6, label="Chance (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AUC (ROC)")
    ax.set_ylim(0.40, min(1.0, max(means) + max(errs) + 0.08))
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")

    # annotate bar tops
    for rect, mu, ci in zip(bars, means, errs):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            mu + ci + 0.008,
            f"{mu:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="#222222",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"[plot_results_barchart] Saved -> {out_path}")
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)

    plot_barchart(
        csv_path="tables/real_llm_results.csv",
        out_path="figures/fig_results_barchart_gsm8k.pdf",
        title="AUC@final — GSM8K (DeepSeek-R1-Distill-Qwen-1.5B)",
    )

    # Ablation simulation chart (uses same metric structure)
    ablation_path = "tables/ablation_results.csv"
    if os.path.exists(ablation_path):
        plot_barchart(
            csv_path=ablation_path,
            out_path="figures/fig_results_barchart_ablation.pdf",
            title="AUC@final — Simulation ablation",
        )


if __name__ == "__main__":
    main()
