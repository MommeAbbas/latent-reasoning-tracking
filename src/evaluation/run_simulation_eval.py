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

def compute_label(xs, zs, tail=8, backtrack_limit=2):
    progress = xs[-1, 0]
    fatigue_tail = np.mean(xs[-tail:, 3])
    backtrack_tail = np.sum(zs[-tail:] == int(Mode.BACKTRACK))
    recent_insight = np.sum(zs[-tail:] == int(Mode.INSIGHT))
    
    backtrack_streak = 0
    for z in zs[-tail:][::-1]:
        if z == int(Mode.BACKTRACK):
            backtrack_streak += 1
        else:
            break

    passed_progress = progress > 0.50 
    
    low_fatigue = fatigue_tail < 0.40
    limited_backtrack = backtrack_tail <= backtrack_limit
    no_streak = backtrack_streak <= 1
    
    return int(passed_progress and low_fatigue and limited_backtrack and no_streak)

def run_single_trajectory(T, dyn, sensors, rbpf, ekf, pf, predictor):
    xs, zs = SLDSSimulator(dyn).run(T=T)
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

def main():
    np.random.seed(42)
    T = 35
    N_traj = 200

    # Setup Dynamics (Physics)
    d_cfg = SLDSConfig()
    d_cfg.P = np.array([
        [0.96, 0.02, 0.02],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ])
    d_cfg.noise_std = (0.015, 0.015, 0.015, 0.01, 0.01)
    dyn = SLDSDynamics(d_cfg)

    # Setup Sensors (Observation)
    s_cfg = SensorConfig()
    s_cfg.noise_std = (0.008, 0.008, 0.008) # Tight noise
    s_cfg.outlier_prob = 0.05
    s_cfg.outlier_scale = 50.0 
    sensors = ReasoningSensors(s_cfg)

    predictor = OnlineCorrectnessPredictor()

    all_preds_rbpf = []
    all_preds_ekf = []
    all_preds_pf = []
    all_labels = []

    print("Running Simulation: Hidden Accumulator Regime...")

    PARTICLES = 125

    for n in range(N_traj):
        rbpf = RBPF_SLDS(dyn=dyn, sensors=sensors, cfg=RBPFConfig(num_particles=PARTICLES))
        pf = PF_SLDS(dyn=dyn, sensors=sensors, cfg=PFConfig(num_particles=PARTICLES))
        ekf = EKFBaseline(dyn=dyn, sensors=sensors, cfg=EKFConfig())

        preds_rbpf, preds_ekf, preds_pf, label = run_single_trajectory(
            T=T, dyn=dyn, sensors=sensors, rbpf=rbpf, ekf=ekf, pf=pf, predictor=predictor
        )

        all_preds_rbpf.append(preds_rbpf)
        all_preds_ekf.append(preds_ekf)
        all_preds_pf.append(preds_pf)
        all_labels.append(label)

        del rbpf, pf, ekf, preds_rbpf, preds_ekf, preds_pf
        if n % 5 == 0:
            gc.collect()
            print(f"  Trajectory {n}/{N_traj} complete")

    all_preds_rbpf = np.array(all_preds_rbpf)
    all_preds_ekf = np.array(all_preds_ekf)
    all_preds_pf = np.array(all_preds_pf)
    all_labels = np.array(all_labels)

    def compute_all_metrics(name, all_preds, all_labels, T):
        nll = negative_log_likelihood(all_preds.flatten(), np.repeat(all_labels, T))
        brier = brier_score(all_preds.flatten(), np.repeat(all_labels, T))
        ece = expected_calibration_error(all_preds.flatten(), np.repeat(all_labels, T))
        aucs = auc_by_prefix([all_preds[:, k] for k in range(T)], all_labels)

        print(f"=== Evaluation results ({name}) ===")
        print(f"NLL:    {nll:.4f}")
        print(f"Brier: {brier:.4f}")
        print(f"ECE:    {ece:.4f}")
        print(f"AUC@final: {aucs[-1]:.4f}")
        print(f"AUC@early (k=5): {aucs[4]:.4f}")
        print()

    compute_all_metrics("RBPF", all_preds_rbpf, all_labels, T)
    compute_all_metrics("EKF",  all_preds_ekf,  all_labels, T)
    compute_all_metrics("PF",   all_preds_pf,   all_labels, T)

if __name__ == "__main__":
    main()