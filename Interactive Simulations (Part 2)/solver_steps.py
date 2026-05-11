from collections.abc import Callable, Sequence, Iterable
import numpy as np

# -------------------------------
# Solver Steps
# -------------------------------

def euler_step(
    func: Callable,
    y_curr: Sequence[float],
    t,
    dt: float = 1e-4,
):
    if isinstance(y_curr, Iterable):    # Input is vectorized
        y_curr = np.array(y_curr)
        func_ = lambda t, ys: np.array(func(t, *ys))
    else:
        func_ = func

    y_next = y_curr + dt * func_(t, y_curr)
    t_next = t + dt

    return t_next, y_next

def improved_euler_step(
    func: Callable,
    y_curr: Sequence[float],
    t,
    dt: float = 1e-4,
):
    if isinstance(y_curr, Iterable):    # Input is vectorized
        y_curr = np.array(y_curr)
        func_ = lambda t, ys: np.array(func(t, *ys))
    else:
        func_ = func

    y_half = y_curr + dt / 2 * func_(t, y_curr)
    y_next = y_curr + dt * func_(t + dt/2, y_half)
    t_next = t + dt

    return t_next, y_next


def rk4_step(
    func: Callable,
    y_curr: Sequence[float] | float,
    t,
    dt = 1e-4
):
    if isinstance(y_curr, Iterable):    # Input is vectorized
        y_curr = np.array(y_curr)
        func_ = lambda t, ys: np.array(func(t, *ys))
    else:
        func_ = func

    k1 = func_(t, y_curr)
    k2 = func_(t + dt / 2, y_curr + dt / 2 * k1)
    k3 = func_(t + dt / 2, y_curr + dt / 2 * k2)
    k4 = func_(t + dt, y_curr + dt * k3)

    y_next = y_curr + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    t_next = t + dt

    return t_next, y_next


# -------------------------------
# Matrix Solver Steps
# -------------------------------
"""
Steps for the IVP solver of the type dy/dt = A @ y with A being A matrix and y a vector.
"""

def matrix_euler_step(A, y_curr, dt):
    return y_curr + dt * A @ y_curr

def matrix_mod_euler_step(A, y_curr, dt):
    y_half = y_curr + dt/2 * A @ y_curr
    return y_curr + dt * A @ y_half

def matrix_rk4_step_old(A, y_curr, dt):
    K = np.zeros((4, *y_curr.shape))
    K[0] = A @ y_curr
    K[1] = A @ (y_curr + dt/2 * K[0])
    K[2] = A @ (y_curr + dt/2 * K[1])
    K[3] = A @ (y_curr + dt   * K[2])

    weights = [1,2,2,1]
    y_next = y_curr + dt/6 * np.tensordot(weights, K, axes=(0,0))

    return y_next

def matrix_rk4_step(A, y_curr, dt):
    C = np.array([  # Equivalent to A in the Butcher Tableau
        [0,   0,   0, 0],
        [1/2, 0,   0, 0],
        [0,   1/2, 0, 0],
        [0,   0,   1, 0]
    ])
    b = np.array([1/6, 1/3, 1/3, 1/6])

    K = np.zeros((4, *y_curr.shape))
    K[0] = A @ y_curr
    for i in range(1,4):
        K[i] = A @ (y_curr + dt * np.tensordot(C[i], K, axes=(0,0)))

    y_next = y_curr + dt * np.tensordot(b, K, axes=(0,0))

    return y_next

# TODO: Remove
# def solve_matrix_ipv(
#     A,
#     y0,
#     t0=0,
#     t1=10,
#     dt=0.25
# ):
#     A = np.array(A)
#     y0 = np.array(y0)

#     t_values = np.arange(t0, t1, dt)
#     y_curr = y0
#     y_values = [y0]
#     for _ in range(len(t_values)):
        
#         y_next = matrix_rk4_step(A, y_curr, dt)
#         y_values.append(y_next)
#         y_curr = y_next

#     return np.array(y_values)
