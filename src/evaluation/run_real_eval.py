import argparse
import gc
import os
import csv
import numpy as np

from src.simulation.slds  import SLDSConfig, SLDSDynamics
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds  import RBPF_SLDS, RBPFConfig
from src.filters.ekf_baseline import EKFBaseline, EKFConfig
from src.filters.pf_baseline  import PF_SLDS, PFConfig
from src.filters.lr_baseline  import LRBaseline, LRFullBaseline
from src.evaluation.online_prediction import OnlineCorrectnessPredictor, OnlinePredictionRecorder
from src.data.real_data_loader import RealDataLoader
from src.data.math500_data_loader import Math500DataLoader
from src.data import real_data_loader as _rdl_module

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    def safe_auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    def safe_aupr(y, s):
        return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
except ImportError:
    def safe_auc(y, s):  return float("nan")
    def safe_aupr(y, s): return float("nan")

def brier(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    return float(np.mean((s - y) ** 2))

N_SEEDS    = 5
PARTICLES  = 125
EARLY_K    = 4
STEP_AUCS  = [2, 4, 6, 8, 10]   # steps at which to report AUC@k


def _select_noise_std(val_loader, base_dyn):
    candidates = [0.05, 0.10, 0.20]
    best_auc, best_cfg = -1.0, None
    val_items = [val_loader.get_trajectory(i) for i in range(min(50, len(val_loader)))]

    for ns in candidates:
        s_cfg = SensorConfig()
        s_cfg.noise_std     = (ns, ns, ns)
        s_cfg.outlier_prob  = 0.05
        s_cfg.outlier_scale = 20.0
        sensors   = ReasoningSensors(s_cfg)
        predictor = OnlineCorrectnessPredictor()

        preds_final, lbls = [], []
        for ys, label in val_items:
            if len(ys) < 2:
                continue
            rbpf = RBPF_SLDS(dyn=base_dyn, sensors=sensors, cfg=RBPFConfig(num_particles=PARTICLES))
            for k in range(len(ys)):
                rbpf.step(ys[k])
            preds_final.append(predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights))
            lbls.append(label)

        a = safe_auc(np.array(lbls), np.array(preds_final))
        if a > best_auc:
            best_auc = a
            best_cfg = s_cfg

    print(f"  [noise selection] best noise_std={best_cfg.noise_std[0]:.2f}, AUC={best_auc:.3f}")
    return best_cfg


def _eval_seed(seed, loader, base_dyn, sensors, prm_scores=None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(loader)).tolist()
    predictor = OnlineCorrectnessPredictor()

    preds_rbpf, preds_ekf, preds_pf = [], [], []
    all_ys, labels = [], []
    mean_entropy_scores = []
    prm_scores_seed = [] if prm_scores is not None else None

    for idx in indices:
        ys, label = loader.get_trajectory(idx)
        if len(ys) < 2:
            continue

        labels.append(label)
        all_ys.append(ys)
        mean_entropy_scores.append(-float(np.mean(ys[:, 0])))
        if prm_scores_seed is not None:
            prm_scores_seed.append(float(prm_scores[idx]))

        rbpf = RBPF_SLDS(dyn=base_dyn, sensors=sensors, cfg=RBPFConfig(num_particles=PARTICLES))
        pf   = PF_SLDS(dyn=base_dyn,   sensors=sensors, cfg=PFConfig(num_particles=PARTICLES))
        ekf  = EKFBaseline(dyn=base_dyn, sensors=sensors, cfg=EKFConfig())

        rec_rbpf = OnlinePredictionRecorder()
        rec_ekf  = OnlinePredictionRecorder()
        rec_pf   = OnlinePredictionRecorder()

        for k in range(len(ys)):
            y = ys[k]
            rbpf.step(y)
            rec_rbpf.update(predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights))
            mu, _ = ekf.step(y)
            rec_ekf.update(predictor.prob_correct_from_state(mu))
            x_hat, _ = pf.step(y)
            rec_pf.update(predictor.prob_correct_from_state(x_hat))

        preds_rbpf.append(rec_rbpf.as_array())
        preds_ekf.append(rec_ekf.as_array())
        preds_pf.append(rec_pf.as_array())

        del rbpf, pf, ekf
        gc.collect()

    labels = np.array(labels)

    def _final(pred_list):
        return np.array([p[-1] for p in pred_list])

    def _early(pred_list, k):
        return np.array([p[min(k, len(p) - 1)] for p in pred_list])

    n_train = int(0.70 * len(labels))
    lr_k5 = LRBaseline(k=EARLY_K + 1)
    lr_k5.fit(all_ys[:n_train], labels[:n_train])
    lr_k5_probs = lr_k5.predict(all_ys[n_train:])
    lr_labels   = labels[n_train:]

    lr_full = LRFullBaseline()
    lr_full.fit(all_ys[:n_train], labels[:n_train])
    lr_full_probs = lr_full.predict(all_ys[n_train:])

    me_scores = np.array(mean_entropy_scores)

    def _at_step(pred_list, k):
        return np.array([p[min(k, len(p) - 1)] for p in pred_list])

    def _filter_metrics(y, pred_list):
        d = {
            "auc_final":  safe_auc(y, _final(pred_list)),
            "aupr_final": safe_aupr(y, _final(pred_list)),
            "brier":      brier(y, _final(pred_list)),
            "auc_early":  safe_auc(y, _early(pred_list, EARLY_K)),
            "aupr_early": safe_aupr(y, _early(pred_list, EARLY_K)),
        }
        for k in STEP_AUCS:
            d[f"auc_k{k}"] = safe_auc(y, _at_step(pred_list, k))
        return d

    def _static_metrics(y, s):
        d = {
            "auc_final":  safe_auc(y, s),
            "aupr_final": safe_aupr(y, s),
            "brier":      brier(y, s),
            "auc_early":  safe_auc(y, s),
            "aupr_early": safe_aupr(y, s),
        }
        for k in STEP_AUCS:
            d[f"auc_k{k}"] = safe_auc(y, s)   # static: same score at every step
        return d

    results = {}
    for name, plist, y in [
        ("RBPF", preds_rbpf, labels),
        ("EKF",  preds_ekf,  labels),
        ("PF",   preds_pf,   labels),
    ]:
        results[name] = _filter_metrics(y, plist)

    for name, probs, y in [
        ("LR-k5",      lr_k5_probs,   lr_labels),
        ("LR-full",    lr_full_probs, lr_labels),
    ]:
        results[name] = _static_metrics(y, probs)

    results["MeanEntropy"] = _static_metrics(labels, me_scores)

    if prm_scores_seed is not None:
        results["PRM"] = _static_metrics(labels, np.array(prm_scores_seed))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], default="gsm8k")
    args = parser.parse_args()

    if args.dataset == "math500":
        loader     = Math500DataLoader(split="eval", n_eval=350)
        val_loader = Math500DataLoader(split="held", n_eval=350)
        tag        = "MATH500"
        csv_name   = "math500_results.csv"
    else:
        loader     = RealDataLoader(split="eval", n_eval=200)
        val_loader = RealDataLoader(split="held", n_eval=200)
        tag        = "GSM8K"
        csv_name   = "real_llm_results.csv"

    print(f"[run_real_eval] Loading {tag} traces ...")
    print(f"  {len(loader)} trajectories loaded.")
    print(f"  Class balance: {loader.labels.mean()*100:.1f}% correct")

    d_cfg = SLDSConfig()
    d_cfg.P = np.array([[0.96, 0.02, 0.02], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]])
    d_cfg.noise_std = (0.015, 0.015, 0.015, 0.01, 0.01)
    base_dyn = SLDSDynamics(d_cfg)

    print(f"[run_real_eval] Selecting sensor noise_std ...")
    sensors = ReasoningSensors(_select_noise_std(val_loader, base_dyn))

    prm_path = os.path.join(os.path.dirname(_rdl_module.__file__), "real_prm_scores.npy")
    prm_scores = np.load(prm_path) if (args.dataset == "gsm8k" and os.path.exists(prm_path)) else None
    if prm_scores is not None:
        print(f"  PRM scores loaded: {len(prm_scores)} problems")

    variant_names = ["RBPF", "EKF", "PF", "LR-k5", "LR-full", "MeanEntropy"]
    if prm_scores is not None:
        variant_names.append("PRM")

    step_keys   = [f"auc_k{k}" for k in STEP_AUCS]
    metric_keys = ["auc_final", "aupr_final", "brier", "auc_early", "aupr_early"] + step_keys
    all_results = {n: {m: [] for m in metric_keys} for n in variant_names}

    for seed in range(N_SEEDS):
        print(f"[run_real_eval] Seed {seed+1}/{N_SEEDS} ...")
        res = _eval_seed(seed, loader, base_dyn, sensors, prm_scores=prm_scores)
        for name in variant_names:
            if name in res:
                for m in metric_keys:
                    all_results[name][m].append(res[name].get(m, float("nan")))

    # ── print summary ──────────────────────────────────────────────────────────
    step_header = "".join(f"  @{k:>2}" for k in STEP_AUCS)
    print(f"\n=== {tag} Eval Results (mean ± std, {N_SEEDS} seeds) ===")
    print(f"{'Method':<12} {'AUC@final':>14} {'AUPR@final':>12} {'Brier':>10}{step_header}")
    print("-" * (52 + 6 * len(STEP_AUCS)))
    for name in variant_names:
        af  = np.nanmean(all_results[name]["auc_final"])
        af_s= np.nanstd(all_results[name]["auc_final"])
        apr = np.nanmean(all_results[name]["aupr_final"])
        br  = np.nanmean(all_results[name]["brier"])
        step_vals = "".join(
            f"  {np.nanmean(all_results[name][k]):.3f}" for k in step_keys
        )
        print(f"{name:<12} {af:.3f}±{af_s:.3f}   {apr:.3f}±{np.nanstd(all_results[name]['aupr_final']):.3f}  "
              f"{br:.3f}±{np.nanstd(all_results[name]['brier']):.3f}{step_vals}")

    os.makedirs("tables", exist_ok=True)
    csv_path = os.path.join("tables", csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed"] + metric_keys)
        for name in variant_names:
            for i in range(len(all_results[name]["auc_final"])):
                row = [name, i] + [f"{all_results[name][m][i]:.4f}" for m in metric_keys]
                writer.writerow(row)
    print(f"\n[run_real_eval] Saved -> {csv_path}")


if __name__ == "__main__":
    main()
