import numpy as np
from numeric_de_solver import *

# Parameters
m = 1
g = 9.81
k = 10
l = 3

t1 = 60

r0 = 0
T0 = np.pi/4

rp0 = 0
Tp0 = -1

def pend_func(t, r, T, rp, Tp):
    return [
        (l + r) * Tp**2 - k/m * r + g * np.cos(T),
        - g/(l+r) * np.sin(T) - 2*rp/(l+r) * Tp 
    ]

cache_options = {
    "m": m,
    "g": g,
    "k": k,
    "l": l
}

t_values, y_values = solve_2nd_order_ivp(
    pend_func,
    [r0, T0],
    [rp0, Tp0],
    t1=t1,
    cache_options=cache_options,
    cache_location="numeric_diffeq_solver\\springed_pendulum.json",
    a_tol=1e-10,
    r_tol=0
)