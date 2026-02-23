from collections.abc import Callable, Iterable, Sequence
from typing import Tuple
import numpy as np

def solve_ivp_euler(
    func: Callable,
    dt: float = 1e-4,
    t0: float = 0,
    t1: float = 1,
    y0: Sequence[float] | float = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Uses the (forward) Euler method to solve the initial value problem (IVP) of the form dy/dt = func(t,y) with y(t0) = y0. Also accepts vector quantities. See https://en.wikipedia.org/wiki/Euler_method.

    Parameters
    ----------
    func : Callable
        Right hand side of the IVP dy/dt = func(t,y). If dealing with vector quantities it should have the from func(t, y1, y2, ..., yn).
    dt : float
        Discretized time step for IVP.
    t0 : float
        Initial time.
    t1 : float
        Ending time of the calculation.
    y0 : Sequence[float] | float
        Initial value(s) of func at the time t0.

    Returns
    -------
    list
        List of times.
    list
        List of calculated points.
    """
    if isinstance(y0, Iterable):    # Input is vectorized
        y0 = np.array(y0)
        func_ = lambda t, ys: np.array(func(t, *ys))
    else:
        func_ = func

    t_values = np.arange(t0, t1+dt, dt)

    y_values = [y0]
    y_curr = y0

    for t in t_values[:-1]:
        
        y_next = y_curr + dt * func_(t, y_curr)

        y_values.append(y_next)
        y_curr = y_next

    return t_values, np.array(y_values)


def solve_ivp_improved_euler(
    func: Callable,
    dt: float = 1e-4,
    t0: float = 0,
    t1: float = 1,
    y0: Sequence[float] | float = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Uses the improved/modified Euler method to solve the initial value problem (IVP) of the form dy/dt = func(t,y) with y(t0) = y0. Also accepts vector quantities. The only change from the original euler method is that we use an intermediate step y_half to get a better estimate of y_next.

    Parameters
    ----------
    func : Callable
        Right hand side of the IVP dy/dt = func(t,y). If dealing with vector quantities it should have the from func(t, y1, y2, ..., yn).
    dt : float
        Discretized time step for IVP.
    t0 : float
        Initial time.
    t1 : float
        Ending time of the calculation.
    y0 : Sequence[float] | float
        Initial value(s) of func at the time t0.

    Returns
    -------
    list
        List of times.
    list
        List of calculated points.
    """
    if isinstance(y0, Iterable):    # Input is vectorized
        y0 = np.array(y0)
        func_ = lambda t, ys: np.array(func(t, *ys))
    else:
        func_ = func

    t_values = np.arange(t0, t1+dt, dt)
    y_values = [y0]
    y_curr = y0

    for t in t_values[:-1]:
        
        y_half = y_curr + dt / 2 * func_(t, y_curr)
        y_next = y_curr + dt * func_(t + dt/2, y_half)

        y_values.append(y_next)
        y_curr = y_next

    return t_values, np.array(y_values)