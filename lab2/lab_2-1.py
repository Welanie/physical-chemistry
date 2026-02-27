import numpy as np

R = 8.314472  # J/mol/K
T = 340.0

# Функции NASA

def cp(T, a):
    return R * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)

def h(T, a):
    return R*T*(a[0] + a[1]*T/2 + a[2]*T**2/3 +
                a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)

def s(T, a):
    return R*(a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 +
              a[3]*T**3/3 + a[4]*T**4/4 + a[6])

def g(T, a):
    return h(T,a) - T*s(T,a)

# C3H3N (CH2=CHCN) 200–1000 K

C3H3N = [
    3.04396646E+00,
    1.05333467E-02,
    1.96574996E-05,
    -3.42001077E-08,
    1.48155667E-11,
    2.06456740E+04,
    1.05816246E+01
]

# Стандартные NASA коэффициенты (200–1000 K)
# O2, CO2, H2O, N2

O2 = [
    3.78245636E+00,
    -2.99673416E-03,
    9.84730201E-06,
    -9.68129509E-09,
    3.24372837E-12,
    -1.06394356E+03,
    3.65767573E+00
]

CO2 = [
    2.35677352E+00,
    8.98459677E-03,
    -7.12356269E-06,
    2.45919022E-09,
    -1.43699548E-13,
    -4.83719697E+04,
    9.90105222E+00
]

H2O = [
    4.19864056E+00,
    -2.03643410E-03,
    6.52040211E-06,
    -5.48797062E-09,
    1.77197817E-12,
    -3.02937267E+04,
    -8.49032208E-01
]

N2 = [
    3.53100528E+00,
    -1.23660987E-04,
    -5.02999437E-07,
    2.43530612E-09,
    -1.40881235E-12,
    -1.04697628E+03,
    2.96747468E+00
]

# Расчёт свойств C3H3N при 340 K

Cp_C3H3N = cp(T, C3H3N)
S_C3H3N = s(T, C3H3N)
H_C3H3N = h(T, C3H3N)
G_C3H3N = g(T, C3H3N)

print("C3H3N at 340 K:")
print("Cp =", Cp_C3H3N, "J/mol/K")
print("S  =", S_C3H3N, "J/mol/K")
print("H  =", H_C3H3N/1000, "kJ/mol")
print("G  =", G_C3H3N/1000, "kJ/mol")

# Тепловой эффект и ΔG реакции

def reaction_property(T, species, nu):
    return sum(nu[i] * g(T, species[i]) for i in range(len(species)))

def reaction_enthalpy(T, species, nu):
    return sum(nu[i] * h(T, species[i]) for i in range(len(species)))

species = [CO2, H2O, N2, C3H3N, O2]

nu = [
    3.0,     # CO2
    1.5,     # H2O
    0.5,     # N2
    -1.0,    # C3H3N
    -3.75    # O2
]

dH = reaction_enthalpy(T, species, nu)
dG = reaction_property(T, species, nu)

print("\nCombustion at 340 K:")
print("delta H =", dH/1000, "kJ/mol")
print("delta G =", dG/1000, "kJ/mol")