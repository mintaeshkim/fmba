# Finite Memory Belief Approximation (FMBA)

This repository contains a minimal, reproducible experimental framework for studying
**finite memory belief approximation** in partially observable stochastic control,
with a concrete specialization to **LQG systems**.

The code is designed to directly validate theoretical results relating
finite memory belief mismatch to control performance degradation under a **fixed controller**.

---

## What This Repo Does

- Simulates a partially observable LQG system under a fixed LQR controller
- Computes the **true belief** using the full Kalman filter
- Constructs **finite-memory beliefs** using a sliding-window / restart Kalman filter
- Measures:
  - belief mismatch (Wasserstein-2 distance)
  - value mismatch under the same controller
- Empirically verifies:
  - exponential decay of belief error with memory length
  - linear scaling between belief error and value degradation

---

## Repository Structure

```bash
fmba/
├── lqg.py  # LQG model and closed-loop simulation
├── kf.py  # Kalman filter + finite-memory (window-restart) filter
├── metrics.py  # Belief-level cost and W2 distance
└── run_exp.py  # Main experiment runner and plotting

out/
├── plots/  # Paper-ready figures
└── results.npz  # Aggregated experiment results
```


The `archive/` directory contains legacy code and is not used in current experiments.

---

## Model and Setup

- System: discrete-time double integrator
- Observations: partial and noisy
- Noise: Gaussian process and observation noise
- Controller: infinite-horizon discrete-time LQR
- Policy is **fixed across all experiments**; only the belief representation changes

---

## Finite-Memory Belief Approximation

Finite memory belief approximations are constructed using a **window-restart Kalman filter**:

- At time `t-H`, the belief is initialized using only the single observation at that time
- The Kalman filter is then run forward over the most recent `H` steps
- This approximates the belief using only a truncated input–output history

Implementation is in `fmba/kf.py`.

---

## Running the Experiments

### Install
```bash
pip install -e .
```

### Run full sweep
```bash
python fmba/run_exp.py
```

This will:
- Sweep memory lengths H = [0, 1, 2, 5, 10, 20, 50, 100]
- Run multiple random seeds
- Save figures and results to out/

---

## Generated Plots

- The script automatically generates paper-ready plots, including:
- Belief mismatch vs memory length (semi-log)
- Value gap vs memory length (semi-log)
- Value gap vs belief mismatch (log-log)
- Time evolution of belief mismatch
- Example trajectories and stage costs
- All figures are saved under out/plots/.

---

## Citation
If you use this code in academic work, please cite the corresponding paper.

```bash

```