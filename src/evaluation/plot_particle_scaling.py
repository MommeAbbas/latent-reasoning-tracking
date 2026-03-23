import os
import numpy as np
import matplotlib.pyplot as plt

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
    "text.usetex": False
})

def get_stats(metric_list, key="auc_early"):
    """
    Extracts mean and std from a list of metric dictionaries
    """
    if not metric_list:
        return np.nan, np.nan
        
    vals = [m[key] for m in metric_list if key in m]
    if not vals:
        return np.nan, np.nan

    return np.mean(vals), np.std(vals)

def plot_scaling():
    try:
        results = np.load("particle_sweep_results.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("Data not found. Run src.experiments.run_particle_sweep first.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    
    if "RBPF" in results:
        xs, means, stds = [], [], []
        for N in sorted(results["RBPF"].keys()):
            m, s = get_stats(results["RBPF"][N]["metrics"])
            xs.append(N)
            means.append(m)
            stds.append(s)
        
        means = np.array(means)
        stds = np.array(stds)
        
        if len(xs) > 0:
            ax.plot(xs, means, 'o-', color='#004488', label='RBPF')
            ax.fill_between(xs, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), 
                            color='#004488', alpha=0.15)

    if "PF" in results:
        xs, means, stds = [], [], []
        for N in sorted(results["PF"].keys()):
            m, s = get_stats(results["PF"][N]["metrics"])
            xs.append(N)
            means.append(m)
            stds.append(s)
            
        means = np.array(means)
        stds = np.array(stds)
        
        if len(xs) > 0:
            ax.plot(xs, means, 's--', color='#DDAA33', label='PF')
            ax.fill_between(xs, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), 
                            color='#DDAA33', alpha=0.15)

    if "EKF" in results and results["EKF"]["metrics"]:
        ekf_mean, ekf_std = get_stats(results["EKF"]["metrics"])
        if not np.isnan(ekf_mean):
            ax.axhline(ekf_mean, color='#BB5566', linestyle='-.', label=f'EKF (AUC={ekf_mean:.2f})')
            ax.fill_between([1, 2000], np.clip(ekf_mean - ekf_std, 0, 1), np.clip(ekf_mean + ekf_std, 0, 1),
                            color='#BB5566', alpha=0.05)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Particles (Log Scale)")
    ax.set_ylabel("AUC (Early Success Prediction)")
    ax.set_title("Scaling of Inference Accuracy with Compute")
    ax.legend(loc="lower right")
    
    ax.set_xlim(left=8, right=250)
    
    plt.tight_layout()
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_particle_scaling.png")
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved publication-ready plot to {output_path}")

if __name__ == "__main__":
    plot_scaling()