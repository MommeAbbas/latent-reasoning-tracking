# Tracking Latent Reasoning Trajectories with RBPF

This repository contains the implementation and evaluation suite for modeling LLM reasoning as a Switching Linear Dynamic System (SLDS)[cite: 6, 40]. The project compares an Extended Kalman Filter (EKF), a standard Particle Filter (PF), and a Rao-Blackwellized Particle Filter (RBPF) for early success prediction[cite: 7, 17].

## Overview
The core of this work is a 5D latent state space representing progress ($p$), consistency ($c$), uncertainty ($u$), fatigue ($f$), and momentum ($m$)[cite: 20, 42]. We utilize a mode-dependent rotation matrix $R(\theta)$ to model discrete cognitive shifts, specifically "Insights" ($\theta = 0.30$) and "Backtracks" ($\theta = -0.30$)[cite: 49, 51].

The primary result demonstrates that analytical marginalization in the **RBPF** ($AUC \approx 0.71$) significantly outperforms unimodal Gaussian estimators like the **EKF** ($AUC \approx 0.13$) when tracking these non-linear transitions[cite: 8, 99, 100].

## Implementation Detail
* **RBPF Logic:** Located in `src/filters/rbpf_slds.py`. It partitions the state into a discrete sampled mode and a continuous analytical state updated via EKF equations[cite: 75, 77, 79].
* **Observation Model:** Found in `src/simulation/sensors.py`. It implements a "Hidden Accumulator" where progress $p$ is unobservable, forcing the filter to infer success from cross-correlations in coherence and uncertainty[cite: 58, 61].

## Reproduction
To generate the metrics and AUC scores reported in the paper:
\`\`\`bash
python run_gsm8k_eval.py
\`\`\`

To visualize the latent mode detection (Figure 2):
\`\`\`bash
python plot_insight_trace.py
\`\`\`
