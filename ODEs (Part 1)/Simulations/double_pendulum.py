import numpy as np
from numeric_de_solver import *

# Parameters
m1 = 1
m2 = 1
l1 = 3
l2 = 3
g = 9.81

v10 = -2
v20 = 3
T10 = np.pi/4
T20 = 1.2*np.pi/4

t1 = 50

def double_pend_rhs(t, T1, T2, T1p, T2p):
    alpha = m2/(m1+m2) * l2/l1
    beta = l1/l2
    dT = T1 - T2

    R1 = - alpha * T2p**2 * np.sin(dT) - g/l1 * np.sin(T1)
    R2 = beta * T1p**2 * np.sin(dT) - g/l2 * np.sin(T2)
    D = 1 - alpha * beta * np.cos(dT)**2

    return [
        (R1 - alpha * R2 * np.cos(dT)) / D,
        (R2 - beta * R1 * np.cos(dT)) / D
    ]

cache_options = {
    "m1": m1,
    "m2": m2,
    "l1": l1,
    "l2": l2,
    "g": g,
    "t1": t1,
}

t_values, y_values = solve_2nd_order_ivp(
    func=double_pend_rhs,
    y0=[T10, T20],
    yp0=[v10,v20],
    t0=0,
    t1=t1,
    cache_location=r"numeric_diffeq_solver/double_pendulum.json",
    cache_options=cache_options,
    a_tol=1e-10,
    r_tol=0
)
