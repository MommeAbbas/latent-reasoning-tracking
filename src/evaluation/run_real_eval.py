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
    from sklearn.metrics import roc_auc_score
    def safe_auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
except ImportError:
    def safe_auc(y, s): return float("nan")

N_SEEDS   = 5
PARTICLES = 125
EARLY_K   = 4


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

    results = {
        "RBPF":         {"auc_final": safe_auc(labels, _final(preds_rbpf)),  "auc_early": safe_auc(labels, _early(preds_rbpf, EARLY_K))},
        "EKF":          {"auc_final": safe_auc(labels, _final(preds_ekf)),   "auc_early": safe_auc(labels, _early(preds_ekf,  EARLY_K))},
        "PF":           {"auc_final": safe_auc(labels, _final(preds_pf)),    "auc_early": safe_auc(labels, _early(preds_pf,   EARLY_K))},
        "LR-k5":        {"auc_final": safe_auc(lr_labels, lr_k5_probs),      "auc_early": safe_auc(lr_labels, lr_k5_probs)},
        "LR-full":      {"auc_final": safe_auc(lr_labels, lr_full_probs),    "auc_early": safe_auc(lr_labels, lr_full_probs)},
        "MeanEntropy":  {"auc_final": safe_auc(labels, me_scores),           "auc_early": safe_auc(labels, me_scores)},
    }
    if prm_scores_seed is not None:
        ps = np.array(prm_scores_seed)
        results["PRM"] = {"auc_final": safe_auc(labels, ps), "auc_early": safe_auc(labels, ps)}
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
    all_results = {n: {"auc_final": [], "auc_early": []} for n in variant_names}

    for seed in range(N_SEEDS):
        print(f"[run_real_eval] Seed {seed+1}/{N_SEEDS} ...")
        res = _eval_seed(seed, loader, base_dyn, sensors, prm_scores=prm_scores)
        for name in variant_names:
            if name in res:
                all_results[name]["auc_final"].append(res[name]["auc_final"])
                all_results[name]["auc_early"].append(res[name]["auc_early"])

    print(f"\n=== {tag} Eval Results (mean ± std, 5 seeds) ===")
    print(f"{'Method':<12} {'AUC@early':>12} {'AUC@final':>12}  Data")
    print("-" * 55)
    for name in variant_names:
        ef = np.array(all_results[name]["auc_early"])
        ff = np.array(all_results[name]["auc_final"])
        supervised = " (supervised)" if name in ("LR-k5", "LR-full") else ""
        print(f"{name:<12} {ef.mean():.3f}±{ef.std():.3f}   {ff.mean():.3f}±{ff.std():.3f}  {tag}{supervised}")

    os.makedirs("tables", exist_ok=True)
    csv_path = os.path.join("tables", csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "auc_final", "auc_early"])
        for name in variant_names:
            for i, (af, ae) in enumerate(zip(all_results[name]["auc_final"], all_results[name]["auc_early"])):
                writer.writerow([name, i, f"{af:.4f}", f"{ae:.4f}"])
    print(f"\n[run_real_eval] Saved -> {csv_path}")


if __name__ == "__main__":
    main()
