import numpy as np
import gc

from src.data.gsm8k_loader import GSM8KLoader
from src.simulation.slds import SLDSConfig, SLDSDynamics
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.filters.pf_baseline import PF_SLDS, PFConfig
from src.filters.ekf_baseline import EKFBaseline, EKFConfig
from src.evaluation.online_prediction import (
    OnlineCorrectnessPredictor,
    OnlinePredictionRecorder,
)
from src.evaluation.metrics import (
    negative_log_likelihood,
    brier_score,
    expected_calibration_error,
)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def run_single_problem(ys, label, dyn, sensors, predictor):
    """
    Run one GSM8K trajectory through all filters
    """
    rbpf = RBPF_SLDS(dyn, sensors, RBPFConfig(num_particles=150))
    pf   = PF_SLDS(dyn, sensors, PFConfig(num_particles=250))
    ekf  = EKFBaseline(dyn, sensors, EKFConfig())

    rec_rbpf = OnlinePredictionRecorder()
    rec_pf   = OnlinePredictionRecorder()
    rec_ekf  = OnlinePredictionRecorder()

    for y in ys:
        # RBPF
        rbpf.step(y)
        p_rbpf = predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights)
        rec_rbpf.update(p_rbpf)

        # PF
        x_hat_pf, _ = pf.step(y)
        p_pf = predictor.prob_correct_from_state(x_hat_pf)
        rec_pf.update(p_pf)

        # EKF
        mu, _ = ekf.step(y)
        p_ekf = predictor.prob_correct_from_state(mu)
        rec_ekf.update(p_ekf)

    res = (rec_rbpf.as_array(), rec_pf.as_array(), rec_ekf.as_array(), label)
    
    del rbpf, pf, ekf, rec_rbpf, rec_pf, rec_ekf
    return res


def main():
    np.random.seed(42)

    # Start with a small N to verify it works
    gsm8k = GSM8KLoader(config_name="main", split="test")
    N = 100
    print(f"Running on first {N} examples from GSM8K...")

    dyn = SLDSDynamics(SLDSConfig(
        state_dim=5,
        noise_std=(0.02, 0.02, 0.02, 0.01, 0.01)
    ))

    sensors = ReasoningSensors(SensorConfig(
        obs_dim=3,
        noise_std=(0.05, 0.05, 0.05), 
    ))

    predictor = OnlineCorrectnessPredictor()

    # Store results as lists (because lengths vary)
    all_rbpf, all_pf, all_ekf = [], [], []
    all_labels = []

    for i in range(N):
        ys, label = gsm8k.get_trajectory(i)

        # Skip degenerate cases (0 or 1 step is too short to filter)
        if ys.shape[0] < 2:
            continue

        preds_rbpf, preds_pf, preds_ekf, lbl = run_single_problem(
            ys, label, dyn, sensors, predictor
        )

        all_rbpf.append(preds_rbpf)
        all_pf.append(preds_pf)
        all_ekf.append(preds_ekf)
        all_labels.append(lbl)

        if i % 10 == 0:
            gc.collect()
            print(f"  Processed {i}/{N}...")
    
    def report(name, list_of_preds):
        if not list_of_preds:
            print(f"No results for {name}")
            return

        # Concatenate all steps from all trajectories into one massive 1D array
        flat_preds = np.concatenate(list_of_preds)
        
        # Repeat the label for every step in that specific trajectory
        flat_labels = np.concatenate([
            np.repeat(lbl, len(p)) for lbl, p in zip(all_labels, list_of_preds)
        ])

        print(f"=== GSM8K Results ({name}) ===")
        print(f"NLL:    {negative_log_likelihood(flat_preds, flat_labels):.4f}")
        print(f"Brier:  {brier_score(flat_preds, flat_labels):.4f}")
        print(f"ECE:    {expected_calibration_error(flat_preds, flat_labels):.4f}")
        
        # Final prediction (the very last step)
        final_preds = np.array([p[-1] for p in list_of_preds])
        final_labels = np.array(all_labels)
        
        # Early prediction (k=4, or last step if length < 5)
        early_preds = np.array([p[min(4, len(p)-1)] for p in list_of_preds])
        
        if roc_auc_score is not None:
            # Check if we have both classes (0 and 1)
            if len(np.unique(final_labels)) > 1:
                auc_final = roc_auc_score(final_labels, final_preds)
                auc_early = roc_auc_score(final_labels, early_preds)
                print(f"AUC@final: {auc_final:.4f}")
                print(f"AUC@early: {auc_early:.4f}")
            else:
                print("AUC: N/A (All labels are 'Correct' in this subset)")
        else:
            print("AUC: Skipped (scikit-learn not installed)")
        
        print()

    report("RBPF", all_rbpf)
    report("PF",   all_pf)
    report("EKF",  all_ekf)


if __name__ == "__main__":
    main()