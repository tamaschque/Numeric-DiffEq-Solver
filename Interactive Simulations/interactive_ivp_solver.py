import numpy as np
from collections.abc import Callable, Sequence, Iterable
from solver_steps import *

solver_steps = {
    "euler": euler_step,
    "mod_euler": improved_euler_step,
    "rk4": rk4_step
}

def solve_ivp_interact(
    func: Callable,
    y_curr: Sequence[float] | float,
    t: float = 0.0,
    method: str = "rk4",
    dt: float = 1/60,
    dt_intern: float = 1e-4,
):
    t_curr = t
    solver = solver_steps[method]
    t_max = t + dt

    while t_curr < t_max:
        
        if t + dt_intern > t_max:
            dt_intern = t_max - t

        t_curr, y_curr = solver(
                            func,
                            y_curr,
                            t_curr,
                            dt_intern
                        )

    return t_curr, y_curr

def solve_2nd_order_ivp_interact(
    func: Callable,
    y_curr: Sequence[float] | float,
    yp_curr: Sequence[float] | float,
    t: float = 0.0,
    method: str = "rk4",
    dt: float = 1/60,
    dt_intern: float = 1e-4,
):
    if isinstance(y_curr, Iterable) or isinstance(yp_curr, Iterable):
        dim = len(y_curr)
        reduced_func = lambda t, *vars: np.array([
            *vars[dim:],
            *func(t, *vars)
        ])
    else:
        reduced_func = lambda t, y, yp: np.array([
            yp,
            func(t, y, yp)
        ])

    reduced_y_curr = np.array([y_curr, yp_curr]).flatten()

    return solve_ivp_interact(
        reduced_func,
        reduced_y_curr,
        t,
        method,
        dt,
        dt_intern
    )


# -------------------------------
# Matrix Solve
# -------------------------------

matrix_solver_steps = {
    "euler": matrix_euler_step,
    "mod_euler": matrix_euler_step,
    "rk4": matrix_rk4_step,
}

def solve_matrix_ivp_interact(
    A,
    y_curr,
    dt=1/60,
    method="rk4",
    dt_intern=1e-6,
):
    y_curr = np.array(y_curr)
    solver = matrix_solver_steps[method]
    t = 0
    while t < dt:
        
        if t + dt_intern > dt:
            dt_intern = dt - t

        y_curr = solver(A, y_curr, dt_intern)
        t += dt_intern

    return np.array(y_curr)
