# fmba/kf.py
import numpy as np


def kf_predict(m, P, A, B, Σ_w, u):
    m_pred = A @ m + (B @ u.reshape(-1, 1)).reshape(-1)
    P_pred = A @ P @ A.T + Σ_w
    return m_pred, P_pred

def kf_update(m_pred, P_pred, C, Σ_v, y):
    # Innovation
    S = C @ P_pred @ C.T + Σ_v
    K = P_pred @ C.T @ np.linalg.inv(S)
    innov = y.reshape(-1, 1) - (C @ m_pred.reshape(-1, 1))
    m = (m_pred.reshape(-1, 1) + K @ innov).reshape(-1)
    P = (np.eye(P_pred.shape[0]) - K @ C) @ P_pred
    return m, P

def kf_update_only(m0, P0, C, Σ_v, y):
    """Posterior given a single obseΣ_vation y, starting from prior (m0,P0)."""
    # This is the boundary belief in the paper: P(x | y)
    S = C @ P0 @ C.T + Σ_v
    K = P0 @ C.T @ np.linalg.inv(S)
    innov = y.reshape(-1, 1) - (C @ m0.reshape(-1, 1))
    m = (m0.reshape(-1, 1) + K @ innov).reshape(-1)
    P = (np.eye(P0.shape[0]) - K @ C) @ P0
    return m, P

def window_restart_kf(m0, P0, A, B, C, Σ_w, Σ_v, y, u, H):
    """
    Build hat b_t^(H) via 'window-restart KF' for each t:
      - boundary belief at time s=t-H:  tilde b_s = P(x_s | y_s) using prior (m0,P0) + y_s only
      - apply KF over suffix (u_{s:s+H-1}, y_{s+1:s+H}) to reach time t
    For t < H, uses s=0 and window length = t.

    Returns:
      mhat[t], Phat[t]
    """
    T = y.shape[0]
    xdim = m0.shape[0]
    mhat = np.zeros((T, xdim))
    Phat = np.zeros((T, xdim, xdim))

    for t in range(T):
        s = max(0, t - H)
        # boundary belief at s: posterior from single obseΣ_vation y[s]
        m, P = kf_update_only(m0, P0, C, Σ_v, y[s])

        # now run KF from s+1 ... t using realized u and y
        # if s==t, no update needed; hat belief at t is boundary itself
        for k in range(s + 1, t + 1):
            # predict using u[k-1]
            m_pred, P_pred = kf_predict(m, P, A, B, Σ_w, u[k - 1])
            # update with y[k]
            m, P = kf_update(m_pred, P_pred, C, Σ_v, y[k])

        mhat[t] = m
        Phat[t] = P

    return mhat, Phat