from numeric_de_solver import *

# Parameters
a = 10
b = 28
c = 8/3

r0 = [
    1.1,    # x0
    0,      # y0
    0       # z0
]

t1 = 60

def lorenz_func(t, x, y, z):
    return [
        a * (y-x),
        x *(b-z) - y,
        x*y - c*z
    ]

cache_options = {
    "a": a,
    "b": b,
    "c": c
}

t_values, y_values = solve_ivp(
    lorenz_func,
    r0,
    t1=t1,
    cache_options=cache_options,
    cache_location="numeric_diffeq_solver\\lorenz_attractor.json",
    a_tol=1e-10,
    r_tol=0
)