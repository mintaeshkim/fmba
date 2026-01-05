# fmba/run_exp.py
import os
import numpy as np
from lqg import make_default_config, simulate_closed_loop
from kf import window_restart_kf
from metrics import belief_stage_cost, gaussian_w2


def main():
    config = make_default_config()

    H_list = [0, 1, 2, 5, 10, 20, 50, 100]
    seeds = list(range(50))  # concise: 50 seeds

    os.makedirs("out", exist_ok=True)

    # Accumulators across seeds
    # We'll compute:
    #  - J_true per seed
    #  - J_hat[H] per seed
    #  - eps_hat[H] per seed: max_t W2(b_t, bhat_t^H)
    J_true_all = []
    J_hat_all = {H: [] for H in H_list}
    eps_all = {H: [] for H in H_list}

    for seed in seeds:
        sim = simulate_closed_loop(config, seed=seed)
        A, B, C = sim["A"], sim["B"], sim["C"]
        Σ_w, Σ_v = sim["Σ_w"], sim["Σ_v"]
        x, y, u = sim["x"], sim["y"], sim["u"]
        m_true, P_true = sim["m_true"], sim["P_true"]

        T = config.T
        gamma = config.gamma

        # True-belief cost functional along the realized u_t = pi(b_t)
        J_true = 0.0
        for t in range(T):
            J_true += (gamma**t) * belief_stage_cost(m_true[t], P_true[t], u[t], config.Q, config.R)
        J_true_all.append(J_true)

        # For each H: window-restart KF -> (mhat, Phat)
        for H in H_list:
            mhat, Phat = window_restart_kf(
                m0=config.m0, P0=config.P0,
                A=A, B=B, C=C,
                Σ_w=Σ_w, Σ_v=Σ_v,
                y=y, u=u,
                H=H
            )

            # Approx-belief cost functional under same realized inputs u_t
            J_hat = 0.0
            w2_t = np.zeros(T)
            for t in range(T):
                J_hat += (gamma**t) * belief_stage_cost(mhat[t], Phat[t], u[t], config.Q, config.R)
                w2_t[t] = gaussian_w2(m_true[t], P_true[t], mhat[t], Phat[t])

            J_hat_all[H].append(J_hat)
            eps_all[H].append(float(np.max(w2_t)))  # sup_t approx by max over horizon

    # Aggregate
    J_true_all = np.array(J_true_all)
    out_rows = []
    for H in H_list:
        J_hat = np.array(J_hat_all[H])
        eps = np.array(eps_all[H])

        gap = np.abs(J_true_all - J_hat)

        row = dict(
            H=H,
            eps_mean=float(eps.mean()),
            eps_se=float(eps.std(ddof=1) / np.sqrt(len(eps))),
            gap_mean=float(gap.mean()),
            gap_se=float(gap.std(ddof=1) / np.sqrt(len(gap))),
            J_true_mean=float(J_true_all.mean()),
            J_hat_mean=float(J_hat.mean()),
        )
        out_rows.append(row)

    # Print concise table
    print("H sweep (fixed controller), reporting eps_H ≈ max_t E[W2] and gap=E[|J - Jhat_H|]")
    print(f"{'H':>4} | {'eps_mean':>10} {'eps_se':>10} | {'gap_mean':>12} {'gap_se':>10}")
    print("-"*60)
    for r in out_rows:
        print(f"{r['H']:>4d} | {r['eps_mean']:>10.4e} {r['eps_se']:>10.4e} | {r['gap_mean']:>12.4e} {r['gap_se']:>10.4e}")

    # Save
    np.savez(
        "out/results.npz",
        config=dict(dt=config.dt, T=config.T, gamma=config.gamma, sigma_w=config.σ_w, sigma_v=config.σ_v),
        H_list=np.array(H_list, dtype=int),
        J_true=J_true_all,
        J_hat={str(H): np.array(J_hat_all[H]) for H in H_list},
        eps={str(H): np.array(eps_all[H]) for H in H_list},
        rows=out_rows,
    )
    print("\nSaved: out/results.npz")

if __name__ == "__main__":
    main()