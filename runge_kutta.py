from collections.abc import Callable, Sequence, Iterable
import numpy as np

def solve_ivp_rk4(
    func: Callable,
    dt: float = 1e-4,
    t0: float = 0,
    t1: float = 1,
    y0: float | Sequence[float] = 0,
    ) -> list:
    """Uses the explicit Runga-Kutta Method of order 4 for solving the IVP dy/dt = func(t, y). Also accepts vectorized input. See https://en.wikipedia.org/wiki/Runge–Kutta_methods.

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
        k1 = func_(t, y_curr)
        k2 = func_(t + dt / 2, y_curr + dt / 2 * k1)
        k3 = func_(t + dt / 2, y_curr + dt / 2 * k2)
        k4 = func_(t + dt, y_curr + dt * k3)

        y_next = y_curr + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        y_values.append(y_next)
        y_curr = y_next

    return t_values, np.array(y_values)


def solve_ivp_rk45(
    func: Callable,
    dt: float = 1e-4,
    t0: float = 0,
    t1: float = 1,
    y0: float = 0,
    a_tol: float = 1e-6,
    r_tol: float = 1e-3,
    dt_max: float = 2,
    ) -> list:
    """Uses the explicit Runga-Kutta 4(5) Method with integrated stepsize control for solving the IVP dy/dt = func(t, y). Also accepts vectorized input. See https://en.wikipedia.org/wiki/Runge–Kutta–Fehlberg_method.

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
    a_tol: float
        Absolute tolarance of the truncated error.
    r_tol: float
        Relative tolarance compared to the next y_value added to the absolute tolerance.

    Returns
    -------
    list
        List of times.
    list
        List of calculated points.
    """
    if isinstance(y0, Iterable):
        y_curr = np.array(y0)
        func_ = lambda t, ys: np.array(func(t, *ys))
    else:
        y_curr = y0
        func_ = func
    
    y_values = [y0]
    t_values = [t0]
    t = t0
    while t < t1:
        if t + dt > t1:
            dt = t1 - t

        k1 = func_(t, y_curr)
        k2 = func_(t + dt*1/4  , y_curr + dt * (k1*1/4))
        k3 = func_(t + dt*3/8  , y_curr + dt * (k1*3/32 + k2*9/32))
        k4 = func_(t + dt*12/13, y_curr + dt * (k1*1932/2197 - k2*7200/2197 + k3*7296/2197))
        k5 = func_(t + dt      , y_curr + dt * (k1*439/216 - k2*8 + k3*3680/513 - k4*845/4104))
        k6 = func_(t + dt*1/2  , y_curr + dt * (-k1*8/27 + k2*2 - k3*3544/2565 + k4*1859/4104 - k5*11/40))

        k  = np.array([k1, k2, k3, k4, k5, k6])

        c4 = np.array([25/216, 0, 1408/2565 , 2197/4104  , -1/5 , 0])
        c5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])

        y_next4 = y_curr + dt * np.dot(c4,k)
        y_next5 = y_curr + dt * np.dot(c5,k)

        err = np.linalg.norm(y_next5 - y_next4)

        err_scale = a_tol + r_tol * np.linalg.norm(y_next5)
        # Accepted Step -> Increase Step Size
        if err <= err_scale:
            if err <= 1e-10:
                scale = 2
            else:
                scale = 0.9 * (err_scale / err)**0.2

            # Store accepted values
            y_values.append(y_next5)
            t_values.append(t + dt)

            # Update Values for next iteration
            t += dt
            dt *= np.clip(scale, 0.1, 2.0)
            dt = min(dt, dt_max)
            y_curr = y_next5

        # Rejected Step -> Decrease Step Size
        else:
            scale = 0.9 * (err_scale / err)**0.2
            dt *= np.clip(scale, 0.1, 2.0)
            dt = min(dt, dt_max)

    return np.array(t_values), np.array(y_values)
