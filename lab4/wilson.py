import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

T = 298.15
R = 8.314462618
R_bar = 0.08314462618

A1, B1, C1 = 15.87, 2808.19, -45.99
Tc1, Pc1, w1 = 556.40, 45.60, 0.19

A2, B2, C2 = 15.90, 2788.51, -52.36
Tc2, Pc2, w2 = 562.10, 48.94, 0.21

x1 = np.array([0.100, 0.300, 0.500, 0.700, 0.900])
y1 = np.array([0.130, 0.342, 0.562, 0.706, 0.908])
Pexp = np.array([0.1310, 0.1379, 0.1435, 0.1478, 0.1510])

x2 = 1 - x1
y2 = 1 - y1

P1sat = np.exp(A1 - B1 / (T + C1)) * 133.322 / 1e5
P2sat = np.exp(A2 - B2 / (T + C2)) * 133.322 / 1e5

Z1 = 0.29056 - 0.08775 * w1
Z2 = 0.29056 - 0.08775 * w2

V1 = (R_bar * Tc1 / Pc1) * Z1 ** (1 + (1 - T / Tc1) ** (2 / 7))
V2 = (R_bar * Tc2 / Pc2) * Z2 ** (1 + (1 - T / Tc2) ** (2 / 7))

g1_exp = y1 * Pexp / (x1 * P1sat)
g2_exp = y2 * Pexp / (x2 * P2sat)
gE_exp = R * T * (x1 * np.log(g1_exp) + x2 * np.log(g2_exp))


def ln_gamma(x, a12, a21):
    x = np.asarray(x)
    x2 = 1 - x
    L12 = (V2 / V1) * np.exp(-a12 / (R * T))
    L21 = (V1 / V2) * np.exp(-a21 / (R * T))
    d1 = x + L12 * x2
    d2 = x2 + L21 * x
    ln1 = -np.log(d1) + x2 * (L12 / d1 - L21 / d2)
    ln2 = -np.log(d2) - x * (L12 / d1 - L21 / d2)
    return ln1, ln2


def f(a):
    a12, a21 = a
    ln1, ln2 = ln_gamma(x1, a12, a21)
    gE = R * T * (x1 * ln1 + x2 * ln2)
    return np.sum(np.abs(gE_exp - gE))


res = minimize(f, [10000, 5000], method='Nelder-Mead')
a12, a21 = res.x

ln1, ln2 = ln_gamma(x1, a12, a21)
g1 = np.exp(ln1)
g2 = np.exp(ln2)

Pcalc = x1 * g1 * P1sat + x2 * g2 * P2sat
y1calc = x1 * g1 * P1sat / Pcalc

L12 = (V2 / V1) * np.exp(-a12 / (R * T))
L21 = (V1 / V2) * np.exp(-a21 / (R * T))

print(f'alpha12 = {a12:.6f} J/mol')
print(f'alpha21 = {a21:.6f} J/mol')
print(f'Lambda12 = {L12:.6f}')
print(f'Lambda21 = {L21:.6f}')
print(' x1      y1_exp   y1_calc   P_exp     P_calc')
for i in range(len(x1)):
    print(f'{x1[i]:.3f}   {y1[i]:.3f}    {y1calc[i]:.3f}    {Pexp[i]:.4f}   {Pcalc[i]:.4f}')

x = np.linspace(0, 1, 200)
x2 = 1 - x
ln1, ln2 = ln_gamma(x, a12, a21)
g1 = np.exp(ln1)
g2 = np.exp(ln2)
P = x * g1 * P1sat + x2 * g2 * P2sat
y = x * g1 * P1sat / P

plt.figure(figsize=(7, 5))
plt.plot(x, y)
plt.scatter(x1, y1)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('y-x диаграмма')
plt.grid()
plt.tight_layout()
plt.savefig('diagram_yx.png', dpi=300)

plt.figure(figsize=(7, 5))
plt.plot(x, P)
plt.plot(y, P)
plt.scatter(x1, Pexp)
plt.scatter(y1, Pexp, marker='s')
plt.xlabel('x1, y1')
plt.ylabel('P, bar')
plt.title('P-x-y диаграмма')
plt.grid()
plt.tight_layout()
plt.savefig('diagram_pxy.png', dpi=300)

plt.show()
