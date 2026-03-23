# Tracking Latent Reasoning Trajectories with Rao-Blackwellized Particle Filtering



This repository implements a framework for modeling LLM reasoning as a Switching Linear Dynamic System (SLDS) and estimating latent progress using Bayesian filtering methods. The goal is to predict whether a reasoning trajectory will succeed before completion.



## Key Result

The Rao-Blackwellized Particle Filter (RBPF) significantly outperforms unimodal estimators in identifying successful trajectories:



* **RBPF**: $AUC \approx 0.71$

* **EKF**: $AUC \approx 0.13$



This improvement arises from explicitly modeling discrete reasoning transitions, such as insight events and backtracking, which traditional Gaussian filters fail to track.



## Overview

Reasoning is modeled as a five-dimensional latent state capturing progress, consistency, uncertainty, fatigue, and momentum. The system evolves according to a Markov-switching process with modes corresponding to steady progression, insight, and backtracking.



The RBPF partitions the problem by sampling discrete modes while analytically tracking continuous states via EKF updates. This allows the filter to maintain multiple hypotheses over reasoning trajectories and revise progress estimates upward when a mode switch is detected.

## Implementation

* `src/filters/`: Core implementations of the RBPF, standard Particle Filter, and EKF baselines.

* `src/simulation/`: SLDS dynamics and Reasoning Sensors (Hidden Accumulator Model).

* `src/evaluation/`: Online correctness predictors and metric recorders.

## Reproduction

To generate the metrics and AUC scores reported in the paper:

```bash

python -m src.evaluation.run_gsm8k_eval.py

```



To visualize latent mode detection (Figure 2) and particle scaling (Figure 3):

```bash

python -m src.evaluation.plot_insight_trace.py

python -m src.evaluation.plot_particle_scaling.py

```

## Paper

For a full description of the model, mathematical derivations, and detailed experimental results, see `paper.pdf`.


