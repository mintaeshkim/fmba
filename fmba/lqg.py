# fmba/lqg.py
import numpy as np
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are
from kf import kf_update, kf_predict, kf_update_only


@dataclass(frozen=True)
class Config:
    dt: float = 0.1
    T: int = 1000
    gamma: float = 0.99

    σ_w: float = 0.05
    σ_v: float = 0.20

    # cost
    Q: np.ndarray = None
    R: np.ndarray = None

    # initial belief
    m0: np.ndarray = None
    P0: np.ndarray = None

def make_default_config():
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    m0 = np.array([0.0, 0.0])
    P0 = np.diag([1.0**2, 0.5**2])
    return Config(Q=Q, R=R, m0=m0, P0=P0)

def make_double_integrator(dt):
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5*dt*dt],
                  [dt]])
    C = np.array([[1.0, 0.0]])
    return A, B, C

def dlqr(A, B, Q, R):
    """Discrete-time infinite-horizon LQR gain u = -K x."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K

def simulate_closed_loop(config, seed):
    """
    Simulate a single trajectory under the fixed policy pi(b_t)= -K m_t,
    where (m_t, P_t) are from the *true* Kalman filter.
    Returns arrays: x[t], y[t], u[t], (m_true[t], P_true[t]).
    """
    rng = np.random.default_rng(seed)
    A, B, C = make_double_integrator(config.dt)

    Σ_w = (config.σ_w**2) * np.eye(2)
    Σ_v = np.array([[config.σ_v**2]])

    # LQR gain (state-feedback on estimate)
    K = dlqr(A, B, config.Q, config.R)

    T = config.T
    x = np.zeros((T, 2))
    y = np.zeros((T, 1))
    u = np.zeros((T, 1))
    m_true = np.zeros((T, 2))
    P_true = np.zeros((T, 2, 2))

    # sample initial true state from prior belief
    x0 = rng.multivariate_normal(config.m0, config.P0)
    x[0] = x0
    y[0] = (C @ x[0]).reshape(-1) + rng.normal(0.0, config.σ_v, size=(1,))

    # initialize KF belief from prior + first obseΣ_vation (time 0 update)
    m, P = kf_update_only(config.m0, config.P0, C, Σ_v, y[0])

    m_true[0] = m
    P_true[0] = P

    # control at t=0 uses m_0
    u[0] = -(K @ m.reshape(-1, 1)).reshape(-1)

    for t in range(1, T):
        # propagate true dynamics with process noise
        w = rng.multivariate_normal(np.zeros(2), Σ_w)
        x[t] = (A @ x[t-1] + (B @ u[t-1]).reshape(-1) + w)

        # obseΣ_vation
        v = rng.normal(0.0, config.σ_v, size=(1,))
        y[t] = (C @ x[t]).reshape(-1) + v

        # Kalman filter step
        m_pred, P_pred = kf_predict(m, P, A, B, Σ_w, u[t-1])
        m, P = kf_update(m_pred, P_pred, C, Σ_v, y[t])

        m_true[t] = m
        P_true[t] = P

        # fixed policy uses true belief mean
        u[t] = -(K @ m.reshape(-1, 1)).reshape(-1)

    return dict(A=A, B=B, C=C, Σ_w=Σ_w, Σ_v=Σ_v, K=K, x=x, y=y, u=u, m_true=m_true, P_true=P_true)