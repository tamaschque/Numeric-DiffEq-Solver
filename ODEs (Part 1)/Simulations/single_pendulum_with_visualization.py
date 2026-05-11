import numpy as np
from numeric_de_solver import *

# Parameters
m = 2
l = 3
g = 9.81
gamma = 1.5

T0 = np.pi/4
Tp0 = 0

t1 = 60

def pend_func(t, T, Tp):
    return -g/l * np.sin(T) - gamma/(m*l**2) * Tp

cache_options = {
    "m": m,
    "l": l,
    "g": g,
    "gamma": gamma
}

t_values, y_values = solve_2nd_order_ivp(
    pend_func,
    T0,
    Tp0,
    t1=t1,
    cache_options=cache_options,
    cache_location="numeric_diffeq_solver\\single_pendulum.json",
    a_tol=1e-6,
    r_tol=0
)