# fmba/metrics.py
import numpy as np
from scipy.linalg import sqrtm


def belief_stage_cost(m, P, u, Q, R):
    # E[x'Qx] = m'Qm + tr(QP)
    return float(m.T @ Q @ m + np.trace(Q @ P) + u.reshape(-1,1).T @ R @ u.reshape(-1,1))

def gaussian_w2(m1, P1, m2, P2):
    """
    W2 distance between N(m1,P1) and N(m2,P2).
    W2^2 = ||m1-m2||^2 + tr(P1 + P2 - 2*(P1^{1/2} P2 P1^{1/2})^{1/2})
    """
    dm = (m1 - m2).reshape(-1, 1)
    term_mean = float(dm.T @ dm)

    # Symmetrize to reduce numerical issues
    P1s = 0.5 * (P1 + P1.T)
    P2s = 0.5 * (P2 + P2.T)

    S1 = sqrtm(P1s)
    # (P1^{1/2} P2 P1^{1/2})^{1/2}
    M = S1 @ P2s @ S1
    SM = sqrtm(0.5 * (M + M.T))

    # sqrtm can return tiny imaginary parts numerically
    tr = np.trace(P1s + P2s - 2.0 * SM)
    tr = np.real_if_close(tr, tol=1e8)
    w2_sq = term_mean + float(tr)
    w2_sq = max(w2_sq, 0.0)
    return float(np.sqrt(w2_sq))