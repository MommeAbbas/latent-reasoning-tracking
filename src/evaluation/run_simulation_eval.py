import numpy as np
import gc

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator, Mode
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds import RBPFConfig, RBPF_SLDS
from src.filters.ekf_baseline import EKFBaseline, EKFConfig
from src.filters.pf_baseline import PF_SLDS, PFConfig
from src.evaluation.online_prediction import (
    OnlineCorrectnessPredictor,
    OnlinePredictionRecorder,
)
from src.evaluation.metrics import (
    negative_log_likelihood,
    brier_score,
    expected_calibration_error,
    auc_by_prefix,
)

def compute_label(xs, zs):
    """
    A trajectory is labelled correct (1) if final progress exceeds 0.5.
    Progress starts at x0[0]=0.2 and only exceeds 0.5 if INSIGHT events occur.
    This gives roughly balanced classes under the default SLDS config.
    """
    return int(xs[-1, 0] > 0.5)

def run_single_trajectory(T, dyn, sensors, rbpf, ekf, pf, predictor, x0=None):
    xs, zs = SLDSSimulator(dyn).run(T=T, x0=x0)
    c = compute_label(xs, zs)
    rec_rbpf = OnlinePredictionRecorder()
    rec_ekf = OnlinePredictionRecorder()
    rec_pf = OnlinePredictionRecorder()

    for k in range(1, T + 1):
        y = sensors.observe(xs[k])
        # RBPF
        rbpf.step(y)
        p_rbpf = predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights)
        rec_rbpf.update(p_rbpf)
        # EKF
        mu, Sigma = ekf.step(y)
        p_ekf = predictor.prob_correct_from_state(mu)
        rec_ekf.update(p_ekf)
        # PF
        x_hat_pf, _ = pf.step(y)
        p_pf = predictor.prob_correct_from_state(x_hat_pf)
        rec_pf.update(p_pf)

    return rec_rbpf.as_array(), rec_ekf.as_array(), rec_pf.as_array(), c

def make_dynamics_and_sensors():
    """Return the canonical (dyn, sensors) pair used across all evals."""
    d_cfg = SLDSConfig()
    d_cfg.P = np.array([
        [0.96, 0.02, 0.02],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ])
    d_cfg.noise_std = (0.015, 0.015, 0.015, 0.01, 0.01)
    dyn = SLDSDynamics(d_cfg)

    s_cfg = SensorConfig()
    s_cfg.noise_std = (0.008, 0.008, 0.008)
    s_cfg.outlier_prob = 0.05
    s_cfg.outlier_scale = 50.0
    sensors = ReasoningSensors(s_cfg)
    return dyn, sensors


def run_eval(seed: int = 42, N_traj: int = 100, T: int = 35, PARTICLES: int = 75, verbose: bool = False):
    """
    Run the full simulation evaluation for a single seed.
    Returns (all_preds_rbpf, all_preds_ekf, all_preds_pf, all_labels)
    each of shape (N_traj, T).
    """
    np.random.seed(seed)
    dyn, sensors = make_dynamics_and_sensors()
    predictor = OnlineCorrectnessPredictor()

    all_preds_rbpf, all_preds_ekf, all_preds_pf, all_labels = [], [], [], []

    # Starting state: progress=0.2 so threshold 0.5 is reachable via insight events
    X0 = np.array([0.2, 0.5, 0.8, 0.2, 0.1])

    for n in range(N_traj):
        rbpf = RBPF_SLDS(dyn=dyn, sensors=sensors, cfg=RBPFConfig(num_particles=PARTICLES))
        pf   = PF_SLDS(dyn=dyn, sensors=sensors, cfg=PFConfig(num_particles=PARTICLES))
        ekf  = EKFBaseline(dyn=dyn, sensors=sensors, cfg=EKFConfig())

        preds_rbpf, preds_ekf, preds_pf, label = run_single_trajectory(
            T=T, dyn=dyn, sensors=sensors, rbpf=rbpf, ekf=ekf, pf=pf,
            predictor=predictor, x0=X0,
        )

        all_preds_rbpf.append(preds_rbpf)
        all_preds_ekf.append(preds_ekf)
        all_preds_pf.append(preds_pf)
        all_labels.append(label)

        del rbpf, pf, ekf, preds_rbpf, preds_ekf, preds_pf
        if verbose and n % 20 == 0:
            gc.collect()
            print(f"  [seed={seed}] Trajectory {n}/{N_traj}")

    return (
        np.array(all_preds_rbpf),
        np.array(all_preds_ekf),
        np.array(all_preds_pf),
        np.array(all_labels),
    )


def compute_and_print_metrics(name, all_preds, all_labels, T):
    nll   = negative_log_likelihood(all_preds.flatten(), np.repeat(all_labels, T))
    brier = brier_score(all_preds.flatten(), np.repeat(all_labels, T))
    ece   = expected_calibration_error(all_preds.flatten(), np.repeat(all_labels, T))
    aucs  = auc_by_prefix([all_preds[:, k] for k in range(T)], all_labels)

    print(f"=== Evaluation results ({name}) ===")
    print(f"NLL:             {nll:.4f}")
    print(f"Brier:           {brier:.4f}")
    print(f"ECE:             {ece:.4f}")
    print(f"AUC@final:       {aucs[-1]:.4f}")
    print(f"AUC@early (k=5): {aucs[4]:.4f}")
    print()
    return aucs


def main():
    T = 35
    print("Running Simulation: Hidden Accumulator Regime...")
    preds_rbpf, preds_ekf, preds_pf, labels = run_eval(seed=42, N_traj=200, T=T, verbose=True)
    compute_and_print_metrics("RBPF", preds_rbpf, labels, T)
    compute_and_print_metrics("EKF",  preds_ekf,  labels, T)
    compute_and_print_metrics("PF",   preds_pf,   labels, T)


if __name__ == "__main__":
    main()