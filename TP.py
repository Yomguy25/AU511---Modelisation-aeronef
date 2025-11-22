# -*- coding: utf-8 -*-
"""
CONTROL OF AIRCRAFT - PRACTICAL WORK
Author:
    FAIVRE Guillaume
    SEITZ Lucas
    Choosed number : 67
"""

import numpy as np
from control import StateSpace
#%% 1)
# -----------------------------
# CONSTANTS
# -----------------------------

g = 9.81                     # gravitational acceleration (m/s^2)
m = 8400                     # aircraft mass (kg)
S = 34                       # wing reference area (m^2)
alt_ft = 15855               # altitude (ft)
alt_m = alt_ft * 0.3048      # altitude converted to meters (m)
M = 1.92                     # Mach number (dimensionless)
R = 287.05                   # specific gas constant for air (J/(kg·K))
l_ref = 5.24
rg = 2.65 

# -----------------------------
# GRAPHICAL DATA (from aerodynamic plots)
# -----------------------------

Cx0 = 0.28                   # zero-lift drag coefficient Cx0 (-)
k = 0.53                     # polar coefficient k (-)
Cz_alpha = 2.1               # lift curve slope w.r.t incidence Cz_alpha (1/rad)
Cz_delta_m = 0.3             # lift curve slope w.r.t elevator Cz_delta_m (1/rad)
delta_m0 = -0.017            # equilibrium elevator deflection for null lift δm0 (rad)
alpha0 = 0.0105              # incidence for zero lift at δm = 0 α0 (rad)
f = -0.609                    # aerodynamic center coefficient of body and wings (−)
f_delta = -0.9                # aerodynamic center coefficient of fins (−)
Cmq = -0.27


# -----------------------------
# COMPUTED DATA 
# -----------------------------

P = 55239.3                  # static pressure (Pa)
T = 256.738                  # temperature (K)
rho = 0.749542               # air density (kg/m^3)
a = 321.211                  # speed of sound at altitude (m/s)
V_eq = M * a                 # true airspeed (m/s)
Q = 0.5 * rho * V_eq**2      # dynamic pressure (Pa = N/m^2)
l_t = (3/2) * l_ref          # total aircraft length lt = 3/2 * l_ref (m)
X = f * l_t                  # aerodynamic center position of wings/body (m)
Y = f_delta * l_t            # aerodynamic center position of fins (m)


# -----------------------------
# ALGORITHM FOR COMPUTING THE EQUILIBRIUM POINT
# -----------------------------
eps=1e-6
alpha_eq = 0.0
Fpx_eq = 0.0

diff = 100000000  # initial value to enter in the while
k=0

while diff > eps:

    Czeq = (m*g - Fpx_eq * np.sin(alpha_eq)) / (Q * S)

    Cxeq = Cx0 + k * Czeq**2

    Cx_delta_m = 2 * k * Czeq * Cz_delta_m

    num = Cxeq * np.sin(alpha_eq) + Czeq * np.cos(alpha_eq)
    den = Cx_delta_m * np.sin(alpha_eq) + Cz_delta_m * np.cos(alpha_eq)
    delta_meq = delta_m0 - (num / den) * (X / (Y - X))

    alpha_next = alpha0 + (Czeq / Cz_alpha) - (Cz_delta_m / Cz_alpha) * delta_meq

    Fpx_next = Q * S * Cxeq / np.cos(alpha_next)

    diff = abs(alpha_next - alpha_eq)

    alpha_eq = alpha_next
    Fpx_eq = Fpx_next
    k+=1
    
print("number iteration: ",k)
print("alpha_eq =", alpha_eq)
print("delta_meq =", delta_meq)
print("Fpx_eq =", Fpx_eq)


#%% 2) State space representation
# -----------------------------
# Additional needed definitions
# -----------------------------

gamma_eq = 0.0                     # level flight (rad)
F_tau = 0.0                        # no thrust variation in simplified model

F_eq = Q * S * Cxeq / np.cos(alpha_eq)


# Aerodynamic derivatives
Cx_alpha = 2 * k * Czeq * Cz_alpha
CN_alpha = Cx_alpha * np.sin(alpha_eq) + Cz_alpha * np.cos(alpha_eq)
CN_delta_m = Cx_delta_m * np.sin(alpha_eq) + Cz_delta_m * np.cos(alpha_eq)

Cm_alpha = (X / l_ref) * CN_alpha
Cm_delta_m = (Y / l_ref) * CN_delta_m

Iyy = m * rg**2 


# -----------------------------
# X-coefficients of simplified longitudinal model
# -----------------------------
X_V = (2 * Q * S * Cxeq) / (m * V_eq)
X_alpha = (F_eq * np.sin(alpha_eq)) / (m * V_eq) + (Q * S * Cx_alpha) / (m * V_eq)
X_gamma = (g * np.cos(gamma_eq)) / V_eq
X_delta_m = (Q * S * Cx_delta_m) / (m * V_eq)
X_tau = -(F_tau * np.cos(alpha_eq)) / (m * V_eq)

# -----------------------------
# m-coefficients of simplified longitudinal model
# -----------------------------
m_V = 0
m_alpha = (Q * S * l_ref * Cm_alpha) / Iyy
m_q = (Q * S * l_ref**2 * Cmq) / (V_eq * Iyy)
m_delta_m = (Q * S * l_ref * Cm_delta_m) / Iyy


# -----------------------------
# Z-coefficients of simplified longitudinal model
# -----------------------------
Z_V = (2 * Q * S * Czeq) / (m * V_eq)
Z_alpha = (F_eq * np.cos(alpha_eq) / (m * V_eq)) + (Q * S * Cz_alpha) / (m * V_eq)
Z_gamma = (g * np.sin(gamma_eq)) / V_eq
Z_delta_m = (Q * S * Cz_delta_m) / (m * V_eq)
Z_tau = (F_tau * np.sin(alpha_eq)) / (m * V_eq)

# -----------------------------------------
# MATRIX 
# -----------------------------------------
A = np.array([
    [-X_V,      -X_gamma,   -X_alpha,   0, 0, 0],
    [ Z_V,       0,          Z_alpha,    0, 0, 0],
    [-Z_V,       0,         -Z_alpha,    1, 0, 0],
    [ 0,         0,          m_alpha,    m_q, 0, 0],
    [ 0,         0,          0,          1,   0, 0],
    [ 0,         V_eq,       0,          0,   0, 0]
])

B = np.array([
    [0],
    [Z_delta_m],
    [-Z_delta_m],
    [m_delta_m],
    [0],
    [0]
])

C = np.eye(6)
D = np.zeros((6,1))


sys = StateSpace(A, B, C, D)

#%% 3)

import control.matlab as cmat
wn, zeta, pole = cmat.damp(sys)


#%% 4)
#%% reduced model 
import math
def mdamp(A):
    roots = np.linalg.eigvals(A)
    
    ri = []
    a = []
    b = []
    w = []
    xi = []
    st = []
    
    for i in range(roots.size):
        ri.append(roots[i])
        
        a.append(roots[i].real)
        b.append(roots[i].imag)
        
        w.append(math.sqrt(a[i]**2 + b[i]**2))
        xi.append(-a[i] / w[i])
        
        signb = '+' if b[i] > 0 else '-'
        
        st.append(
            f"{a[i]:.5f}{signb}j{abs(b[i]):.5f}  "
            f"xi = {xi[i]:.5f}  w = {w[i]:.5f} rad/s"
        )
    
    print(st)
    
    
Ar = np.array([
    [-X_V,     -X_gamma,   -X_alpha,   0],
    [ Z_V,      0,          Z_alpha,    0],
    [-Z_V,      0,         -Z_alpha,    1],
    [ 0,        0,          m_alpha,    m_q]
])
# mdamp(Ar)


Br = np.array([
    [0],
    [Z_delta_m],
    [-Z_delta_m],
    [m_delta_m]
])
# mdamp(Br)

Cr = np.eye(4)
Dr = np.zeros((4,1))

sys_r = StateSpace(Ar, Br, Cr, Dr)
wn_r, zeta_r, pole_r = cmat.damp(sys_r)

#%% 
from __future__ import unicode_literals
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import interpolate
from control import matlab
import control
from pylab import * 
from atm_std import *
from sisopy31 import * 

Ai = Ar[2:4, 2:4]
Bi = Br[2:4, 0:1]

mdamp(Ai)

Cia = np.matrix([[1, 0]])
Ciq = np.matrix([[0, 1]])
Di = np.matrix([[0]])

TaDm_ss = control.ss(Ai, Bi, Cia, Di)
print("Transfer function alpha / delta m =")
TaDm_tf = control.tf(TaDm_ss)
print(TaDm_tf)

print("Static gain of alpha / delta m = %f" % (control.dcgain(TaDm_tf)))

TqDm_ss = control.ss(Ai, Bi, Ciq, Di)
print("Transfer function q / delta m =")
TqDm_tf = control.ss2tf(TqDm_ss)
print(TqDm_tf)

print("Static gain of q / delta m = %f" % (dcgain(TqDm_tf)))

figure(1)

Ya, Ta = control.matlab.step(TaDm_tf, arange(0, 10, 0.01))
Yq, Tq = control.matlab.step(TqDm_tf, arange(0, 10, 0.01))

plot(Ta, Ya, 'b', Tq, Yq, 'r', lw=2)

plot([0, Ta[-1]], [Ya[-1], Ya[-1]], 'k--', lw=1)
plot([0, Ta[-1]], [1.05 * Ya[-1], 1.05 * Ya[-1]], 'k--', lw=1)
plot([0, Ta[-1]], [0.95 * Ya[-1], 0.95 * Ya[-1]], 'k--', lw=1)

plot([0, Tq[-1]], [Yq[-1], Yq[-1]], 'k--', lw=1)
plot([0, Tq[-1]], [1.05 * Yq[-1], 1.05 * Yq[-1]], 'k--', lw=1)
plot([0, Tq[-1]], [0.95 * Yq[-1], 0.95 * Yq[-1]], 'k--', lw=1)

minorticks_on()
grid(visible=True, which='both')


title(r"Step response $\alpha/\delta m$ et $q/\delta m$")
legend((r'$\alpha/\delta m$', r'$q/\delta m$'))
xlabel("Time (s)")
ylabel(r'$\alpha$ (rad) & $q$ (rad/s)')

Osa, Tra, Tsa = step_info(Ta, Ya)
Osq, Trq, Tsq = step_info(Tq, Yq)

yya = interp1d(Ta, Ya)
plot(Tsa, yya(Tsa), 'bs')
text(Tsa, yya(Tsa) - 0.2, Tsa)

yyq = interp1d(Tq, Yq)
plot(Tsq, yyq(Tsq), 'r s')
text(Tsq, yyq(Tsq) - 0.2, Tsq)

print("Alpha Settling time 5%% = %f s" % Tsa)
print("q Settling time 5%% = %f s" % Tsq)

savefig("stepalphaq.pdf")


