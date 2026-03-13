import numpy as np

R = 8.314472
T = 333.1

a1_g = 9.87121167E+00
a2_g = -9.36699002E-03
a3_g = 1.69887865E-04
a4_g = -2.15019520E-07
a5_g = 8.45407091E-11
a6_g = -2.37185495E+04

a1_l = 3.23581200E+01
a2_l = -1.55919703E-01
a3_l = 6.05367043E-04
a4_l = -5.71237410E-07
a5_l = -1.30759900E-10
a6_l = -3.07686562E+04


def H_NASA(T, a1, a2, a3, a4, a5, a6):
    return R * T * (
            a1
            + a2 * T / 2
            + a3 * T ** 2 / 3
            + a4 * T ** 3 / 4
            + a5 * T ** 4 / 5
            + a6 / T
    )


H_g = H_NASA(T, a1_g, a2_g, a3_g, a4_g, a5_g, a6_g)
H_l = H_NASA(T, a1_l, a2_l, a3_l, a4_l, a5_l, a6_l)

delta_H = H_g - H_l

print("дельта H_исп (333.1 K) = {:.3f} kJ/mol".format(delta_H / 1000))
