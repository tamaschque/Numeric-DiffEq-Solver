from collections.abc import Iterable

from euler_method import *
from runge_kutta import *
from caching import *

solvers = {
    "euler" : solve_ivp_euler,
    "mod_euler": solve_ivp_improved_euler,
    "rk4": solve_ivp_rk4,
    "rk45": solve_ivp_rk45
}

def solve_ivp(
    func: Callable,
    y0: float,
    method: str = "rk45",
    t0: float = 0,
    t1: float = 1,
    dt: float = 1e-4,
    cache_result = True,
    check_cache: bool = True,
    cache_location: str = None,
    cache_options: dict = None,
    **kwargs
):
    """
    Solves the IVP dy/dt = func(t, y). Also accepts vectorized intput
    
    Parameters
    ----------
    func : Callable
        Right hand side of the equation. For vectorized input it should have the form f(t, y1, y2, ..., yn).
    y0 : float
        Initial value for the y at t0.
    yp0 : float
        Initial values for y' = dy/dt at t0.
    method : str | Callable
        Method used to solve the reduced IVP. Available methods are:
        * "rk45" (default)  - Explicit Runge-Kutta method of order 5(4)
        * "euler"           - Forward euler method
        * "mod_euler"       - Improved/Modified euler method
        * "rk4"             - Runge Kutta of order 4
    t0 : float
        Initial time.
    t1 : float
        Ending time.
    dt : float
        (Initial) Time step size.
    cache_result: bool
        If set to `True` and a `cache_location` is given the output will be cached there.
    check_cache: bool
        If set to True it will first check the file a `cache_location` and compare.
    cache_location: str
        Path to json file serving as the cache.
    cache_options: dict
        Further options to be tracked in the cache. Eg if the function has variables. For the funciton only the bytecode gets stored and compared so the variable values don't actually get updated automatically.
    kwargs:
        Further kwargs to be passed into the solver. Eg absolute and relative tolerances of rk45.
        
    Returns
    -------
    list
        List of times.
    list
        List of points of y and y' in order [y1, y2, ..., yn]
    """
    if cache_options is None:
        cache_options = {}
    
    cache_options.update(
        {
            "t0": t0,
            "t1": t1,
            "y0": list(y0) if isinstance(y0, Iterable) else y0,
            "dt": dt,
            "method": method,
            "func_id": str(func.__code__.co_code)
        }
    )

    if check_cache and cache_location:
        if matching_cache_options(cache_location, **cache_options):
            print("Matching Cache")
            return load_chached_result(cache_location)

    solver = solvers[method]
    t_values, y_values = solver(func, y0=y0, t0=t0, t1=t1, dt=dt, **kwargs)

    if cache_result and cache_location:
        cache(
            filename=cache_location,
            t_values=t_values,
            y_values=y_values,
            **cache_options
        )

    return t_values, y_values


def solve_2nd_order_ivp(
    func: Callable,
    y0: Sequence[float] | float,
    yp0: Sequence[float] | float,
    method: str = "rk45",
    t0: float = 0,
    t1: float = 1,
    dt: float = 1e-4,
    cache_result = True,
    check_cache: bool = True,
    cache_location: str = None,
    cache_options: dict = None,
    **kwargs
    ):
    """
    Solves the IVP d²y/d²t = func(t, y, y') by reducing it to a system of first order IVPs of the Form:
    I : dv/dt = func
    II: dy/dt = v
    
    Parameters
    ----------
    func : Callable
        Right hand side of the equation. For vectorized input it should have the form f(t, y1, y2, ... yn, y1', y2', ... yn').
    y0 : float
        Initial value for the y at t0.
    yp0 : float
        Initial values for y' = dy/dt at t0.
    method : str | Callable
        Method used to solve the reduced IVP. Available methods are:
        * "rk45" (default)  - Explicit Runge-Kutta method of order 5(4)
        * "euler"           - Forward euler method
        * "mod_euler"       - Improved/Modified euler method
        * "rk4"             - Runge Kutta of order 4
    t0 : float
        Initial time.
    t1 : float
        Ending time.
    dt : float
        (Initial) Time step size.
    cache_result: bool
        If set to `True` and a `cache_location` is given the output will be cached there.
    check_cache: bool
        If set to True it will first check the file a `cache_location` and compare.
    cache_location: str
        Path to json file serving as the cache.
    cache_options: dict
        Further options to be tracked in the cache. Eg if the function has variables. For the funciton only the bytecode gets stored and compared so the variable values don't actually get updated automatically.
    kwargs:
        Further kwargs to be passed into the solver. Eg absolute and relative tolerances of rk45.
        
    Returns
    -------
    list
        List of times.
    list
        List of points of y and y' in order [y1, y2, ..., yn, y1', y2', ... yn']
    """   
    if isinstance(y0, Iterable) or isinstance(yp0, Iterable):
        dim = len(y0)
        reduced_func = lambda t, *vars: np.array([
            *vars[dim:],
            *func(t, *vars)
        ])
    else:
        reduced_func = lambda t, y, yp: np.array([
            yp,
            func(t, y, yp)
        ])

    reduced_y0 = np.array([y0, yp0]).flatten()

    # Since we only look at the byte code we need to look at the underlying function. reduced_func is a wrapper and so its byte code doesnt change even if we change func
    if cache_options is None:
        cache_options = {}
    cache_options.update(
        {
            "unwrapped_func": str(func.__code__.co_code)
        }
    )

    t_values, y_values = solve_ivp(
        reduced_func,
        y0=reduced_y0.tolist(),
        method=method,
        t0=t0,
        t1=t1,
        dt=dt,
        cache_result=cache_result,
        check_cache=check_cache,
        cache_location=cache_location,
        cache_options=cache_options,
        **kwargs
    )

    return t_values, y_values