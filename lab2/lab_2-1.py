import numpy as np

R = 8.314472
T = 340.0

a1_f = 3.04396646E+00
a2_f = 1.05333467E-02
a3_f = 1.96574996E-05
a4_f = -3.42001077E-08
a5_f = 1.48155667E-11
a6_f = 2.06456740E+04
a7_f = 1.05816246E+01


def Cp_NASA(T, a1, a2, a3, a4, a5):
    return R * (a1 + a2 * T + a3 * T ** 2 + a4 * T ** 3 + a5 * T ** 4)


def H_NASA(T, a1, a2, a3, a4, a5, a6):
    return R * T * (
            a1
            + a2 * T / 2
            + a3 * T ** 2 / 3
            + a4 * T ** 3 / 4
            + a5 * T ** 4 / 5
            + a6 / T
    )


def S_NASA(T, a1, a2, a3, a4, a5, a7):
    return R * (
            a1 * np.log(T)
            + a2 * T
            + a3 * T ** 2 / 2
            + a4 * T ** 3 / 3
            + a5 * T ** 4 / 4
            + a7
    )


Cp_f = Cp_NASA(T, a1_f, a2_f, a3_f, a4_f, a5_f)
H_f = H_NASA(T, a1_f, a2_f, a3_f, a4_f, a5_f, a6_f)
S_f = S_NASA(T, a1_f, a2_f, a3_f, a4_f, a5_f, a7_f)

print("Cp (340 K) = {:.3f} J/(mol*K)".format(Cp_f))
print("S (340 K) = {:.3f} J/(mol*K)".format(S_f))
