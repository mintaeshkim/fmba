import os
import numpy as np
import matplotlib.pyplot as plt

from lqg import Config, simulate_closed_loop
from kf import window_restart_kf
from metrics import belief_stage_cost, gaussian_w2


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _mean_se(x: np.ndarray):
    x = np.asarray(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return mean, se


def _safe_positive(y, eps=1e-12):
    """Avoid log(0) in plots/fits."""
    y = np.asarray(y, dtype=float)
    return np.maximum(y, eps)


def _fit_line_x_logy(x, y):
    """
    Fit: log(y) = a + b*x  (least squares).
    Returns (a, b) in natural log.
    """
    x = np.asarray(x, dtype=float)
    y = _safe_positive(y)
    ly = np.log(y)
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    a, b = coef[0], coef[1]
    return float(a), float(b)


def _fit_line_logx_logy(x, y):
    """
    Fit: log(y) = a + b*log(x)  (least squares).
    Returns (a, b) in natural log.
    """
    x = _safe_positive(x)
    y = _safe_positive(y)
    lx = np.log(x)
    ly = np.log(y)
    A = np.stack([np.ones_like(lx), lx], axis=1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    a, b = coef[0], coef[1]
    return float(a), float(b)


def _plot_errorbar_vs_H(
    H_list, mean_list, se_list,
    ylabel, title, save_path,
    xscale="linear", yscale="linear",
    add_fit_logy=False,
    fit_start_H=None,
):
    H = np.asarray(H_list, dtype=float)
    y = np.asarray(mean_list, dtype=float)
    e = np.asarray(se_list, dtype=float)

    plt.figure()
    plt.errorbar(H, y, yerr=e, fmt="o-", capsize=3)

    # Optional: add exponential-fit line in semilogy sense (log y vs H).
    if add_fit_logy:
        mask = np.ones_like(H, dtype=bool)
        if fit_start_H is not None:
            mask &= (H >= float(fit_start_H))
        # H can include 0; that's fine for x. y must be positive.
        a, b = _fit_line_x_logy(H[mask], y[mask])
        H_fit = np.linspace(H.min(), H.max(), 200)
        y_fit = np.exp(a + b * H_fit)
        plt.plot(H_fit, y_fit, "--", linewidth=1.5, label=rf"fit: $\log y = {a:.2f} + ({b:.2f})H$")

    plt.xlabel("Memory length H")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.xscale(xscale)
    plt.yscale(yscale)

    if add_fit_logy:
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_scatter(
    x, y, xlabel, ylabel, title, save_path,
    xscale="linear", yscale="linear",
    add_fit_loglog=False,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    plt.figure()
    plt.scatter(x, y, s=30)

    if add_fit_loglog:
        # Fit in log-log. Need positive x,y.
        a, b = _fit_line_logx_logy(x, y)
        x_fit = np.geomspace(_safe_positive(x).min(), _safe_positive(x).max(), 200)
        y_fit = np.exp(a) * (x_fit ** b)
        plt.plot(x_fit, y_fit, "--", linewidth=1.5, label=rf"fit: $\log y = {a:.2f} + ({b:.2f})\log x$")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.xscale(xscale)
    plt.yscale(yscale)

    if add_fit_loglog:
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_time_curves(t, curves_mean, curves_se, labels, title, ylabel, save_path, logy=False):
    plt.figure()
    for mean, se, lab in zip(curves_mean, curves_se, labels):
        mean = np.asarray(mean, dtype=float)
        se = np.asarray(se, dtype=float)
        plt.plot(t, mean, label=lab)
        plt.fill_between(t, mean - se, mean + se, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    config = Config()

    H_list = [0, 1, 2, 5, 10, 20, 50, 100]
    seeds = list(range(50))

    out_dir = "out"
    plot_dir = os.path.join(out_dir, "plots")
    _ensure_dir(out_dir)
    _ensure_dir(plot_dir)

    T = config.T
    gamma = config.gamma

    J_true_all = []
    J_hat_all = {H: [] for H in H_list}
    eps_all = {H: [] for H in H_list}

    w2_sum = {H: np.zeros(T) for H in H_list}
    w2_sumsq = {H: np.zeros(T) for H in H_list}

    example = None
    example_seed = seeds[0]

    for seed in seeds:
        sim = simulate_closed_loop(config, seed=seed)
        A, B, C = sim["A"], sim["B"], sim["C"]
        Σ_w, Σ_v = sim["Σ_w"], sim["Σ_v"]
        x, y, u = sim["x"], sim["y"], sim["u"]
        m_true, P_true = sim["m_true"], sim["P_true"]

        J_true = 0.0
        stage_true = np.zeros(T)
        for t in range(T):
            ct = belief_stage_cost(m_true[t], P_true[t], u[t], config.Q, config.R)
            stage_true[t] = ct
            J_true += (gamma**t) * ct
        J_true_all.append(J_true)

        if seed == example_seed:
            example = dict(
                x=x.copy(), y=y.copy(), u=u.copy(),
                m_true=m_true.copy(), P_true=P_true.copy(),
                stage_true=stage_true.copy(),
            )

        for H in H_list:
            mhat, Phat = window_restart_kf(
                m0=config.m0, P0=config.P0,
                A=A, B=B, C=C,
                Σ_w=Σ_w, Σ_v=Σ_v,
                y=y, u=u,
                H=H
            )

            J_hat = 0.0
            w2_t = np.zeros(T)
            stage_hat = None
            if seed == example_seed:
                stage_hat = np.zeros(T)

            for t in range(T):
                cht = belief_stage_cost(mhat[t], Phat[t], u[t], config.Q, config.R)
                J_hat += (gamma**t) * cht
                if stage_hat is not None:
                    stage_hat[t] = cht

                w2 = gaussian_w2(m_true[t], P_true[t], mhat[t], Phat[t])
                w2_t[t] = w2

            J_hat_all[H].append(J_hat)
            eps_all[H].append(float(np.max(w2_t)))

            w2_sum[H] += w2_t
            w2_sumsq[H] += w2_t**2

            if seed == example_seed:
                if "stage_hat_by_H" not in example:
                    example["stage_hat_by_H"] = {}
                example["stage_hat_by_H"][H] = stage_hat

    J_true_all = np.array(J_true_all)
    out_rows = []

    eps_mean_list, eps_se_list = [], []
    gap_mean_list, gap_se_list = [], []

    for H in H_list:
        J_hat = np.array(J_hat_all[H])
        eps_seedwise = np.array(eps_all[H])
        gap = np.abs(J_true_all - J_hat)

        eps_mean, eps_se = _mean_se(eps_seedwise)
        gap_mean, gap_se = _mean_se(gap)

        out_rows.append(dict(
            H=H,
            eps_mean=eps_mean,
            eps_se=eps_se,
            gap_mean=gap_mean,
            gap_se=gap_se,
            J_true_mean=float(J_true_all.mean()),
            J_hat_mean=float(J_hat.mean()),
        ))

        eps_mean_list.append(eps_mean)
        eps_se_list.append(eps_se)
        gap_mean_list.append(gap_mean)
        gap_se_list.append(gap_se)

    print("H sweep (fixed controller), reporting eps_H ≈ E[max_t W2] and gap=E[|J - Jhat_H|]")
    print(f"{'H':>4} | {'eps_mean':>10} {'eps_se':>10} | {'gap_mean':>12} {'gap_se':>10}")
    print("-"*60)
    for r in out_rows:
        print(f"{r['H']:>4d} | {r['eps_mean']:>10.4e} {r['eps_se']:>10.4e} | {r['gap_mean']:>12.4e} {r['gap_se']:>10.4e}")

    # ==========================
    # Plot 1 (PAPER): semilogy => straight line for exponential decay
    # x: H (linear), y: eps (log)
    # ==========================
    _plot_errorbar_vs_H(
        H_list, eps_mean_list, eps_se_list,
        ylabel=r"$\widehat{\varepsilon}_H(\pi)$",
        title="Finite-memory belief mismatch vs H (semi-log; exponential decay appears linear)",
        save_path=os.path.join(plot_dir, "eps_vs_H_semilogy.png"),
        xscale="linear",
        yscale="log",
        add_fit_logy=True,
        fit_start_H=5,   # optional: fit on tail only (edit as you like)
    )

    # Plot 2 (optional): gap vs H (also semilogy often looks linear in tail)
    _plot_errorbar_vs_H(
        H_list, gap_mean_list, gap_se_list,
        ylabel=r"$\mathbb{E}[|J(\pi)-\hat J_H(\pi)|]$",
        title="Fixed-policy cost mismatch vs H (semi-log)",
        save_path=os.path.join(plot_dir, "gap_vs_H_semilogy.png"),
        xscale="linear",
        yscale="log",
        add_fit_logy=True,
        fit_start_H=5,
    )

    # ==========================
    # Plot 3 (PAPER): log-log => straight line for linear scaling gap ~ eps
    # ==========================
    _plot_scatter(
        x=eps_mean_list,
        y=gap_mean_list,
        xlabel=r"$\widehat{\varepsilon}_H(\pi)$",
        ylabel=r"$\mathbb{E}[|J-\hat J_H|]$",
        title="Cost mismatch vs belief mismatch (log-log; linear scaling appears as slope-1 line)",
        save_path=os.path.join(plot_dir, "gap_vs_eps_loglog.png"),
        xscale="log",
        yscale="log",
        add_fit_loglog=True,
    )

    # ----- Time curves (unchanged) -----
    N = len(seeds)
    w2_mean_by_t = {}
    w2_se_by_t = {}
    for H in H_list:
        mean = w2_sum[H] / N
        var = np.maximum(w2_sumsq[H] / N - mean**2, 0.0)
        se = np.sqrt(var / max(N - 1, 1))
        w2_mean_by_t[H] = mean
        w2_se_by_t[H] = se

    H_show = [0, 5, 20, 100] if 100 in H_list else [H_list[0], H_list[len(H_list)//2], H_list[-1]]
    t_axis = np.arange(T)

    _plot_time_curves(
        t=t_axis,
        curves_mean=[w2_mean_by_t[H] for H in H_show],
        curves_se=[w2_se_by_t[H] for H in H_show],
        labels=[f"H={H}" for H in H_show],
        title="Time profile of belief mismatch (mean ± SE)",
        ylabel=r"$\mathbb{E}[W_2(b_t,\hat b_t^{(H)})]$",
        save_path=os.path.join(plot_dir, "w2_time_avg.png"),
        logy=False,
    )

    # example plots (unchanged)
    if example is not None:
        x_ex = example["x"]
        plt.figure()
        plt.plot(t_axis, x_ex[:, 0], label="position")
        plt.plot(t_axis, x_ex[:, 1], label="velocity")
        plt.xlabel("t")
        plt.ylabel("state")
        plt.title(f"Example trajectory (seed={example_seed})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "traj_example.png"), dpi=200)
        plt.close()

    if example is not None and "stage_hat_by_H" in example:
        H_ex = 20 if 20 in H_list else H_list[min(3, len(H_list)-1)]
        stage_true = example["stage_true"]
        stage_hat = example["stage_hat_by_H"][H_ex]

        plt.figure()
        plt.plot(t_axis, stage_true, label=r"$\bar c(b_t,u_t)$ (true belief)")
        plt.plot(t_axis, stage_hat, label=rf"$\bar c(\hat b_t^{{(H)}},u_t)$ (H={H_ex})")
        plt.xlabel("t")
        plt.ylabel("stage cost")
        plt.title(f"Example belief-level stage cost (seed={example_seed})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "cost_time_example.png"), dpi=200)
        plt.close()

    np.savez(
        os.path.join(out_dir, "results.npz"),
        config=dict(dt=config.dt, T=config.T, gamma=config.gamma, sigma_w=config.σ_w, sigma_v=config.σ_v),
        H_list=np.array(H_list, dtype=int),
        J_true=J_true_all,
        J_hat={str(H): np.array(J_hat_all[H]) for H in H_list},
        eps={str(H): np.array(eps_all[H]) for H in H_list},
        rows=out_rows,
        w2_mean_by_t={str(H): w2_mean_by_t[H] for H in H_list},
        w2_se_by_t={str(H): w2_se_by_t[H] for H in H_list},
        example_seed=int(example_seed),
    )
    print("\nSaved: out/results.npz")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()