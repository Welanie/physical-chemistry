import math

R = 8.314
T = 298
P = 101000

V_A = 2e-4
V_B = 4e-4

n_A = P * V_A / (R * T)
n_B = P * V_B / (R * T)
n_total = n_A + n_B

x_A = n_A / n_total
x_B = n_B / n_total

delta_S = -R * (n_A * math.log(x_A) + n_B * math.log(x_B))

print("дельта S = {:.3f} J/K".format(delta_S))
