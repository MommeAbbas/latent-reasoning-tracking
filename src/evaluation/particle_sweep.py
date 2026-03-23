import time
import numpy as np
import gc  # Garbage Collection
from sklearn.metrics import roc_auc_score

# Data & Physics
from src.data.gsm8k_loader import GSM8KLoader
from src.evaluation.online_prediction import OnlineCorrectnessPredictor

# Filters
from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.filters.pf_baseline import PF_SLDS, PFConfig
from src.filters.ekf_baseline import EKFBaseline, EKFConfig

def run_filter_on_trace(ys, filter_obj, predictor, is_particle_filter=False, is_rbpf=False):
    """
    Runs filter on the full trajectory (or up to T=20) to ensure the event is seen
    """
    limit_steps = 20 
    
    pred_at_k = 0.5
    
    # Run filter up to the limit or end of data
    steps = min(len(ys), limit_steps + 1)
    
    for t in range(steps):
        # Update Filter
        if is_particle_filter:
            res = filter_obj.step(ys[t])
            x_est = res[0] if isinstance(res, tuple) else res
        elif is_rbpf:
            filter_obj.step(ys[t])
        else:
            mu, _ = filter_obj.step(ys[t])
            x_est = mu

        # Update Prediction (continuously update until the end)
        if is_rbpf:
            pred_at_k = predictor.prob_correct_from_particles(filter_obj.mus, filter_obj.weights)
        elif is_particle_filter:
            pred_at_k = predictor.prob_correct_from_state(x_est)
        else:
            # EKF
            pred_at_k = predictor.prob_correct_from_state(x_est)
                
    return pred_at_k

def main():
    np.random.seed(42)
    print("Initializing components...")
    predictor = OnlineCorrectnessPredictor()
    loader = GSM8KLoader(config_name="main", split="test")
    
    cached_data = []
    print("Caching 50 evaluation trajectories...")
    while len(cached_data) < 50:
        idx = np.random.randint(0, len(loader))
        ys, label = loader.get_trajectory(idx)
        if len(ys) >= 6:
            cached_data.append((ys, label))
            
    print(f"Dataset ready. Starting Optimized Sweep...")

    N_TRIALS = 3
    RBPF_particles = [10, 25, 50, 100, 200]
    PF_particles = [10, 25, 50, 100, 200]
    
    results = {
        "RBPF": {N: {"metrics": [], "runtimes": []} for N in RBPF_particles},
        "PF":   {N: {"metrics": [], "runtimes": []} for N in PF_particles},
        "EKF":  {"metrics": [], "runtimes": []}
    }

    # Run EKF once (deterministic)
    print("\n=== Running Baseline (EKF) ===")
    preds, labels, times = [], [], []
    for ys, label in cached_data:
        ekf = EKFBaseline(loader.dyn, loader.sensors, EKFConfig())
        t0 = time.perf_counter()
        p = run_filter_on_trace(ys, ekf, predictor, is_particle_filter=False)
        dt = time.perf_counter() - t0
        preds.append(p)
        labels.append(label)
        times.append(dt)
    
    ekf_auc = roc_auc_score(labels, preds)
    ekf_time = np.mean(times)
    # Replicate for N_TRIALS so the plotter works
    results["EKF"]["metrics"] = [{"auc_early": ekf_auc}] * N_TRIALS
    results["EKF"]["runtimes"] = [ekf_time] * N_TRIALS
    print(f"EKF Baseline AUC: {ekf_auc:.3f}")
    
    del ekf, preds, labels, times
    gc.collect()

    # Run PFs
    for trial in range(N_TRIALS):
        print(f"\n=== Trial {trial+1}/{N_TRIALS} ===")
        
        # RBPF
        for N in RBPF_particles:
            preds, labels, times = [], [], []
            for ys, label in cached_data:
                rbpf = RBPF_SLDS(loader.dyn, loader.sensors, RBPFConfig(num_particles=N))
                t0 = time.perf_counter()
                p = run_filter_on_trace(ys, rbpf, predictor, is_particle_filter=False, is_rbpf=True)
                dt = time.perf_counter() - t0
                preds.append(p)
                labels.append(label)
                times.append(dt)
            
            auc = roc_auc_score(labels, preds)
            results["RBPF"][N]["metrics"].append({"auc_early": auc})
            results["RBPF"][N]["runtimes"].append(np.mean(times))
            print(f"  RBPF (N={N}): {auc:.3f}")
            
            del rbpf, preds, labels, times
            gc.collect()

        # PF
        for N in PF_particles:
            preds, labels, times = [], [], []
            for ys, label in cached_data:
                pf = PF_SLDS(loader.dyn, loader.sensors, PFConfig(num_particles=N))
                t0 = time.perf_counter()
                p = run_filter_on_trace(ys, pf, predictor, is_particle_filter=True)
                dt = time.perf_counter() - t0
                preds.append(p)
                labels.append(label)
                times.append(dt)
            
            auc = roc_auc_score(labels, preds)
            results["PF"][N]["metrics"].append({"auc_early": auc})
            results["PF"][N]["runtimes"].append(np.mean(times))
            print(f"  PF (N={N}):   {auc:.3f}")
            
            del pf, preds, labels, times
            gc.collect()

    np.save("particle_sweep_results.npy", results)
    print("\nSaved particle_sweep_results.npy")

if __name__ == "__main__":
    main()