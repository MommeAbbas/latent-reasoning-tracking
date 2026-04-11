# Tracking Latent Reasoning Trajectories with Rao-Blackwellized Particle Filtering

Models LLM reasoning as a Switching Linear Dynamical System (SLDS) and estimates latent progress online using Bayesian filters. The goal is to predict whether a reasoning chain will succeed before it finishes, without access to ground-truth labels at test time.

## Results

**Simulation** (1000 trajectories, 10 seeds):

| Method | AUC@final |
|---|---|
| RBPF | 0.662 ± 0.043 |
| PF | 0.659 ± 0.040 |
| EKF | 0.050 ± 0.010 |

**Ablation** — removing rotational dynamics (RBPF-NoRot) collapses AUC to 0.130, identifying the mode geometry as the critical component.

**Real LLM** (DeepSeek-R1 on GSM8K, 200 problems):

| Method | AUC@final |
|---|---|
| RBPF | 0.649 ± 0.006 |
| PF | 0.651 ± 0.023 |
| EKF | 0.597 ± 0.000 |
| LR (supervised) | 0.518 ± 0.055 |

## Model

The latent state is 5-dimensional: progress, coherence, uncertainty, fatigue, momentum. Three discrete modes (Normal, Insight, Backtrack) are distinguished by rotational dynamics in the coherence-uncertainty subspace. The RBPF samples discrete modes via particles while tracking the continuous state analytically per particle using EKF updates.

Observations are derived from model logits at each reasoning step: token entropy, answer consistency, and step perplexity.

## Structure

```
src/
  filters/       rbpf_slds.py, pf_baseline.py, ekf_baseline.py, lr_baseline.py
  simulation/    slds.py, sensors.py
  evaluation/    run_simulation_eval.py, run_ablation_eval.py, run_real_eval.py
  data/          gsm8k_llm_runner.py, real_data_loader.py
```

## Reproducing Results

```bash
pip install -r requirements.txt

# simulation eval
python -m src.evaluation.run_simulation_eval

# ablation
python -m src.evaluation.run_ablation_eval

# collect real LLM traces (needs GPU)
python -m src.data.gsm8k_llm_runner

# evaluate on real traces
python -m src.evaluation.run_real_eval
```

Figures are saved to `figures/`, tables to `tables/`.
