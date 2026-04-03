# Tracking Latent Reasoning Trajectories with Rao-Blackwellized Particle Filtering

This repository implements a framework for modeling LLM reasoning as a Switching Linear Dynamic System (SLDS) and estimating latent progress using Bayesian filtering methods. The goal is to predict whether a reasoning trajectory will succeed before completion.

## Key Result

The Rao-Blackwellized Particle Filter (RBPF) significantly outperforms unimodal estimators in identifying successful trajectories on a synthetic SLDS benchmark calibrated to the step-count distribution of GSM8K:

* **RBPF**: AUC reported in paper (run `src/evaluation/plot_roc_gsm8k.py` to reproduce)
* **EKF**: substantially lower AUC — see paper for details

This improvement arises from explicitly modeling discrete reasoning transitions (insight events and backtracking), which traditional Gaussian filters cannot track.

> **Note on experiments:** The simulation benchmark generates synthetic SLDS trajectories whose *lengths* are drawn from the GSM8K dataset's step-count distribution. Real LLM outputs are used in a separate experiment via `src/data/gsm8k_llm_runner.py`.



## Overview

Reasoning is modeled as a five-dimensional latent state capturing progress, consistency, uncertainty, fatigue, and momentum. The system evolves according to a Markov-switching process with modes corresponding to steady progression, insight, and backtracking.



The RBPF partitions the problem by sampling discrete modes while analytically tracking continuous states via EKF updates. This allows the filter to maintain multiple hypotheses over reasoning trajectories and revise progress estimates upward when a mode switch is detected.

## Implementation

* `src/filters/`: Core implementations of the RBPF, standard Particle Filter, and EKF baselines.

* `src/simulation/`: SLDS dynamics and Reasoning Sensors (Hidden Accumulator Model).

* `src/evaluation/`: Online correctness predictors and metric recorders.

## Reproduction

To generate the ROC figure and AUC scores from real filter runs:

```bash
python -m src.evaluation.plot_roc_gsm8k
```

To run the full simulation evaluation (single seed, prints metrics table):

```bash
python -m src.evaluation.run_simulation_eval
```

To run ablation studies across 5 SLDS variants:

```bash
python -m src.evaluation.run_ablation_eval
```

To collect real LLM reasoning traces (requires HuggingFace `transformers`):

```bash
python -m src.data.gsm8k_llm_runner
```

To evaluate filters on the collected real traces:

```bash
python -m src.evaluation.run_real_eval
```

To visualize latent mode detection (Figure 2) and particle scaling (Figure 3):

```bash
python -m src.evaluation.plot_insight_trace
python -m src.evaluation.plot_particle_scaling
```

## Paper

For a full description of the model, mathematical derivations, and experimental results, see `paper.pdf`.


