# Finite memory belief approximation (LQG) - fixed policy experiment

## Requirements
- numpy
- scipy

## Run
python run_exp.py

Outputs:
- prints a table of H vs eps_H (max_t E[W2]) and gap_H = E[|J - Jhat_H|]
- saves results to ./out/results.npz