import os
import numpy as np
import matplotlib.pyplot as plt
from src.data.gsm8k_loader import GSM8KLoader
from src.simulation.slds import Mode
from src.simulation.sensors import SensorConfig
from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.filters.ekf_baseline import EKFBaseline, EKFConfig

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

def get_insight_prob(f):
    """
    Extracts P(z=INSIGHT) using the key 'modes'
    """
    if hasattr(f, 'modes'): 
        zs = np.atleast_1d(f.modes)
        return np.sum(f.weights[zs == 1])

    if hasattr(f, 'zs'): 
        zs = np.atleast_1d(f.zs)
        return np.sum(f.weights[zs == 1])
                
    return 0.0

def plot_single_trace():
    loader = GSM8KLoader(config_name="main", split="test")
    
    loader.sensors.cfg.noise_std = (0.01, 0.01, 0.01) 
    
    found_insight = False
    print("Searching for a clean 'Textbook' trajectory (Success + Length > 5)...")
    
    for _ in range(1000):
        idx = np.random.randint(0, len(loader))
        ys, label = loader.get_trajectory(idx) 
        
        if label == 1 and len(ys) >= 6:
            found_insight = True
            break
            
    if not found_insight:
        print("Could not find suitable trajectory. Please run again.")
        return

    print(f"Found trajectory at index {idx} (T={len(ys)}). Running filters...")
    T = len(ys)
    
    rbpf = RBPF_SLDS(loader.dyn, loader.sensors, RBPFConfig(num_particles=500))
    ekf = EKFBaseline(loader.dyn, loader.sensors, EKFConfig())
    
    rbpf_progress = []
    rbpf_prob_insight = []
    ekf_progress = []
    
    for y in ys:
        # RBPF Step
        rbpf.step(y)
        p_mean = np.sum(rbpf.mus[:, 0] * rbpf.weights)
        rbpf_progress.append(p_mean)
        prob = get_insight_prob(rbpf)
        rbpf_prob_insight.append(prob)

        # EKF Step
        mu, _ = ekf.step(y)
        ekf_progress.append(mu[0])

    time_steps = np.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    
    # observations
    axes[0].plot(time_steps, ys[:, 0], color='#44AA99', label='Observed Coherence', marker='o', markersize=4)
    axes[0].plot(time_steps, ys[:, 1], color='#882255', label='Observed Uncertainty', marker='o', markersize=4)
    axes[0].set_ylabel("Sensor Values")
    axes[0].set_title(f"A. The Raw Signal (Length T={T})")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    
    # interpretation
    axes[1].plot(time_steps, rbpf_prob_insight, color='#004488', lw=2, label="RBPF P(Insight)")
    axes[1].fill_between(time_steps, 0, rbpf_prob_insight, color='#004488', alpha=0.1)
    axes[1].set_ylabel("Probability of Insight")
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_title("B. Latent Mode Detection")
    axes[1].legend(loc="upper left", fontsize=9)
    
    # estimated progress
    axes[2].plot(time_steps, rbpf_progress, color='#004488', lw=2, label="RBPF Estimated Progress")
    axes[2].plot(time_steps, ekf_progress, color='#BB5566', lw=2, ls='-.', label="EKF Estimated Progress")
    axes[2].axhline(0.5, color='gray', linestyle=':', label="Success Threshold")
    
    axes[2].set_ylabel("Latent Progress")
    axes[2].set_xlabel("Reasoning Step (t)")
    axes[2].set_title("C. Final Trajectory Prediction")
    axes[2].legend(loc="upper left", fontsize=9)
    
    plt.tight_layout()
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_trace_dynamics.png")
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved qualitative trace to {output_path}")

if __name__ == "__main__":
    plot_single_trace()