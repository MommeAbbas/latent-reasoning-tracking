"""
Particle scaling study: AUC vs. particle count for RBPF and PF.
Uses the same dynamics/sensors as run_simulation_eval.py for consistency.
Results saved to particle_sweep_results.npy and used by plot_particle_scaling.py.
"""

import time
import numpy as np
import gc
from sklearn.metrics import roc_auc_score

from src.simulation.slds import SLDSSimulator, Mode
from src.evaluation.online_prediction import OnlineCorrectnessPredictor
from src.evaluation.run_simulation_eval import make_dynamics_and_sensors, compute_label
from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.filters.pf_baseline import PF_SLDS, PFConfig
from src.filters.ekf_baseline import EKFBaseline, EKFConfig


X0 = np.array([0.2, 0.5, 0.8, 0.2, 0.1])
N_TRAJ      = 100
T           = 35
N_TRIALS    = 3


def run_filter_on_trace(ys, filter_obj, predictor, is_rbpf=False, is_pf=False):
    for y in ys:
        if is_rbpf:
            filter_obj.step(y)
        elif is_pf:
            x_est, _ = filter_obj.step(y)
        else:
            mu, _ = filter_obj.step(y)
            x_est = mu

    if is_rbpf:
        return predictor.prob_correct_from_particles(filter_obj.mus, filter_obj.weights)
    return predictor.prob_correct_from_state(x_est)


def main():
    np.random.seed(42)
    dyn, sensors = make_dynamics_and_sensors()
    predictor    = OnlineCorrectnessPredictor()

    # Cache trajectories so all filter variants see the same data
    print(f"Generating {N_TRAJ} trajectories ...")
    cached = []
    while len(cached) < N_TRAJ:
        xs, zs = SLDSSimulator(dyn).run(T=T, x0=X0)
        label  = compute_label(xs, zs)
        ys     = np.array([sensors.observe(xs[k + 1]) for k in range(T)])
        cached.append((ys, label))

    PARTICLE_COUNTS = [10, 25, 50, 100, 200]

    results = {
        "RBPF": {N: {"metrics": [], "runtimes": []} for N in PARTICLE_COUNTS},
        "PF":   {N: {"metrics": [], "runtimes": []} for N in PARTICLE_COUNTS},
        "EKF":  {"metrics": [], "runtimes": []},
    }

    # EKF (deterministic — run once)
    print("Running EKF ...")
    preds, labels, times = [], [], []
    for ys, label in cached:
        ekf = EKFBaseline(dyn, sensors, EKFConfig())
        t0  = time.perf_counter()
        p   = run_filter_on_trace(ys, ekf, predictor)
        times.append(time.perf_counter() - t0)
        preds.append(p); labels.append(label)
    ekf_auc = roc_auc_score(labels, preds)
    results["EKF"]["metrics"]  = [{"auc_early": ekf_auc}] * N_TRIALS
    results["EKF"]["runtimes"] = [np.mean(times)] * N_TRIALS
    print(f"  EKF AUC: {ekf_auc:.3f}")

    for trial in range(N_TRIALS):
        print(f"\nTrial {trial + 1}/{N_TRIALS}")

        for N in PARTICLE_COUNTS:
            # RBPF
            preds, labels, times = [], [], []
            for ys, label in cached:
                rbpf = RBPF_SLDS(dyn, sensors, RBPFConfig(num_particles=N))
                t0   = time.perf_counter()
                p    = run_filter_on_trace(ys, rbpf, predictor, is_rbpf=True)
                times.append(time.perf_counter() - t0)
                preds.append(p); labels.append(label)
            auc = roc_auc_score(labels, preds)
            results["RBPF"][N]["metrics"].append({"auc_early": auc})
            results["RBPF"][N]["runtimes"].append(np.mean(times))
            print(f"  RBPF N={N:3d}: AUC={auc:.3f}")
            del rbpf; gc.collect()

            # PF
            preds, labels, times = [], [], []
            for ys, label in cached:
                pf = PF_SLDS(dyn, sensors, PFConfig(num_particles=N))
                t0 = time.perf_counter()
                p  = run_filter_on_trace(ys, pf, predictor, is_pf=True)
                times.append(time.perf_counter() - t0)
                preds.append(p); labels.append(label)
            auc = roc_auc_score(labels, preds)
            results["PF"][N]["metrics"].append({"auc_early": auc})
            results["PF"][N]["runtimes"].append(np.mean(times))
            print(f"  PF   N={N:3d}: AUC={auc:.3f}")
            del pf; gc.collect()

    np.save("particle_sweep_results.npy", results)
    print("\nSaved particle_sweep_results.npy")


if __name__ == "__main__":
    main()
