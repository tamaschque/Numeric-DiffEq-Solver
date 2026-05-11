import numpy as np
from numeric_de_solver import *

"""
The initial values for the periodic solutions were taken from Matthew Sheens project.
    Video:          https://www.youtube.com/watch?v=8_RRZcqBEAc
    GitHub Repo:    https://github.com/mws262/MAE5730_examples/tree/master/3BodySolutions    
"""

t1 = 60

G = 1
m1 = 1
m2 = 1
m3 = 1

def dist(r1, r2):
    return np.linalg.norm(r1-r2)

f1 = lambda r1, r2, r3: -G * m2 * (r1-r2) / dist(r1,r2)**3 - G * m3 * (r1-r3) / dist(r1,r3)**3

f2 = lambda r1, r2, r3: -G * m1 * (r2-r1) / dist(r2,r1)**3 - G * m3 * (r2-r3) / dist(r2,r3)**3

f3 = lambda r1, r2, r3: -G * m1 * (r3-r1) / dist(r3,r1)**3 - G * m2 * (r3-r2) / dist(r3,r2)**3

def threebodyfunc(t, r1x , r1y , r2x , r2y , r3x , r3y, r1px, r1py, r2px, r2py, r3px, r3py):
    
    r1 = np.array([r1x,r1y])
    r2 = np.array([r2x,r2y])
    r3 = np.array([r3x,r3y])
                
    return [
        *f1(r1,r2, r3),
        *f2(r1,r2, r3),
        *f3(r1,r2, r3),
    ]


cache_options = {
    "G": G,
    "m1": m1,
    "m2": m2,
    "m3": m3,
    "t1": t1
}

# Chaotic Solution
y0 = [
        -2 , 0,
        0  ,-1,
        2  , 0
    ]

yp0=[
        0.5, 0.25,
        -1 , 0,
        0.5, -0.25
    ]


# Periodic: Flower
y0_per_flower = [
    -0.602885898116520,
    1.059162128863347e-1,
    0.252709795391000,
    1.058254872224370-1,
    -0.355389016941814,
    1.038323764315145e-1
]
yp0_per_flower = [
    0.122913546623784,
    0.747443868604908,
    -0.019325586404545,
    1.369241993562101,
    -0.103587960218793,
    -2.116685862168820
]

# Periodic: Flourished
y0_per_flourish = [
    0.716248295712871,
    0.384288553041130,
    0.086172594591232,
    1.342795868576616,
    0.538777980807643,
    0.481049882655556
]
yp0_per_flourish = [
    1.245268230895990,
    2.444311951776573,
    -0.675224323690062,
    -0.962879613630031,
    -0.570043907205925,
    -1.481432338146543
]

# Periodic: Triangles in Circle
y0_per_triag = [
    1.666163752077218-1,
    -1.081921852656887+1,
    0.974807336315507-1,
    -0.545551424117481+1,
    0.896986706257760-1,
    -1.765806200083609+1
]
yp0_per_triag = [
    0.841202975403070,
    0.029746212757039,
    0.142642469612081,
    -0.492315648524683,
    -0.983845445011510,
    0.462569435774018
]

# Periodic: Oval, Cat, Starship
y0_per_cat =[
    0.536387073390469,
    0.054088605007709,
    -0.252099126491433,
    0.694527327749042,
    -0.275706601688421,
    -0.335933589317989
]

yp0_per_cat = [
    -0.569379585580752,
    1.255291102530929,
    0.079644615251500,
    -0.458625997341406,
    0.489734970329286,
    -0.796665105189482
]

# Computation
t_values, y_values = solve_2nd_order_ivp(
    threebodyfunc,
    y0=y0_per_cat,
    yp0=yp0_per_cat,
    t0=0,
    t1=t1,
    cache_location=r"numeric_diffeq_solver\three_body_prob_periodic_cache\3bp_per_cat.json",
    cache_options=cache_options,
    check_cache=False,
    a_tol=1e-12,
    r_tol=0
)

