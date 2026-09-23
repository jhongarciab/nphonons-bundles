# -*- coding: utf-8 -*-
"""
Tarea 8: verificacion algebraica (via matrices truncadas, QuTiP) de la
transformacion tipo "polaron"

    S = -(g_z/omega_m) * sigma_z * (a^dagger - a)

aplicada a

    H = (omega_q/2) sigma_z + omega_m a^dagger a + g_x sigma_x (a+a^dagger)
        + g_z sigma_z (a+a^dagger)

Se calcula e^S H e^{-S} mediante la expansion de Baker-Campbell-Hausdorff
hasta segundo orden:

    H' = H + [S,H] + (1/2!)[S,[S,H]] + ...

y se extrae el coeficiente del termino sigma_y (a^2 - a^dagger^2)
(equivalente a sigma_{+-} a^2, a^{dagger 2}), comparandolo con:

    Ec. (9)  esperada: coeficiente = -2i g            (sigma_y (a^dagger^2-a^2))
                        equiv.       -2g (sigma_+-sigma_-)(a^dagger^2-a^2)
    Ec. (11) usada:    coeficiente = +g  (sigma_+-sigma_-)(a^dagger^2-a^2)

con g = g_z g_x / omega_m (definicion del codigo).

Se verifica TANTO por derivacion de commutadores anidados (BCH truncada)
COMO por diagonalizacion/exponenciacion matricial exacta (para confirmar
que las correcciones de orden superior en g_z/omega_m son despreciables).

NO se modifica ningun script original.
"""

import numpy as np
from qutip import tensor, qeye, destroy, sigmaz, sigmax, sigmay, commutator

r = 0.1
gz_org, omega_m_org, kappa_org = 2 * np.pi * 6e6, 2 * np.pi * 100e6, 2 * np.pi * 100e3
gz_d = gz_org / kappa_org
om_m = omega_m_org / kappa_org
gx_dim = r * gz_d
g_theory = gz_d * gx_dim / om_m
omega_q_dim = 2 * om_m

print(f"g_z={gz_d}, g_x={gx_dim}, omega_m={om_m}, g=g_z*g_x/omega_m={g_theory}")
print(f"Parametro pequeno de la transformacion: g_z/omega_m = {gz_d/om_m:.4f}")

N = 30   # truncamiento de Fock, generoso para evitar artefactos de borde
Na = 2
sz = tensor(sigmaz(), qeye(N))
sx = tensor(sigmax(), qeye(N))
sy = tensor(sigmay(), qeye(N))
a = tensor(qeye(Na), destroy(N))
ad = a.dag()

H_qubit = 0.5 * omega_q_dim * sz
H_free = om_m * ad * a
H_gx = gx_dim * sx * (a + ad)
H_gz = gz_d * sz * (a + ad)
H = H_qubit + H_free + H_gx + H_gz

S = -(gz_d / om_m) * sz * (ad - a)

# ------------------------------------------------------------
# 1) BCH truncada a segundo orden (commutadores anidados exactos,
#    sin truncar Fock salvo por el tamano N elegido)
# ------------------------------------------------------------
C1 = commutator(S, H)                 # [S,H]
C2 = commutator(S, C1)                # [S,[S,H]]
H_bch2 = H + C1 + 0.5 * C2

# Termino teorico esperado en H' (Ec. 9): -2i*g*sigma_y*(ad^2 - a^2)
termino_teorico_Eq9 = -2j * g_theory * sy * (ad**2 - a**2)

# Para aislar el coeficiente medido, proyectamos sobre el "molde" de la
# forma sy*(ad^2-a^2): coeficiente = <molde, C1>/<molde,molde> usando el
# producto interno de Hilbert-Schmidt (traza), en el subespacio de bajos
# numeros de Fock para evitar efectos de borde del truncamiento.
molde = sy * (ad**2 - a**2)


def proyectar(operador, molde, Ncut=15, Ntot=None):
    """Coeficiente de minimos cuadrados de 'operador' sobre 'molde',
    restringido al subespacio de Fock < Ncut (para evitar bordes)."""
    if Ntot is None:
        Ntot = N
    from qutip import Qobj
    proj_osc = Qobj(np.diag([1.0 if n < Ncut else 0.0 for n in range(Ntot)]))
    proj = tensor(qeye(Na), proj_osc)
    op_c = proj * operador * proj
    mo_c = proj * molde * proj
    num = (mo_c.dag() * op_c).tr()
    den = (mo_c.dag() * mo_c).tr()
    return (num / den).real if abs((num/den).imag) < 1e-6 else num/den


coef_C1 = proyectar(C1, molde)
coef_bch2 = proyectar(H_bch2 - H, molde)   # parte inducida total (C1 + 0.5 C2)

print(f"\nCoeficiente de sigma_y*(a^dagger^2-a^2) en [S,H]        = {coef_C1:.6f}")
print(f"Coeficiente de sigma_y*(a^dagger^2-a^2) en [S,H]+0.5[S,[S,H]] = {coef_bch2:.6f}")
print(f"Esperado segun Ec.(9): -2g = {-2*g_theory:.6f}")

# Verificar que [S,[S,H]] NO aporta al termino sigma_y (a^2,a^dagger^2)
# (debe ser un simple corrimiento de energia, ver derivacion analitica)
coef_C2 = proyectar(C2, molde)
print(f"\nCoeficiente de sigma_y*(a^dagger^2-a^2) en [S,[S,H]] (deberia ser ~0) = {coef_C2:.3e}")
print(f"Norma de [S,[S,H]] menos su parte escalar (deberia ser chica): "
      f"{(C2 - (C2.tr()/C2.shape[0])*tensor(qeye(Na),qeye(N))).norm():.3e}  "
      f"vs norma de C2 = {C2.norm():.3e}")

# ------------------------------------------------------------
# 2) Verificacion con exponencial matricial EXACTA (no solo BCH truncada)
#    en un subespacio de Fock chico para que expm sea barato,
#    y comparando contra la prediccion analitica en ese mismo subespacio.
# ------------------------------------------------------------
Nsmall = 12
sz_s = tensor(sigmaz(), qeye(Nsmall))
sx_s = tensor(sigmax(), qeye(Nsmall))
sy_s = tensor(sigmay(), qeye(Nsmall))
a_s = tensor(qeye(Na), destroy(Nsmall))
ad_s = a_s.dag()
H_s = 0.5 * omega_q_dim * sz_s + om_m * ad_s * a_s + gx_dim * sx_s * (a_s + ad_s) + gz_d * sz_s * (a_s + ad_s)
S_s = -(gz_d / om_m) * sz_s * (ad_s - a_s)

Hp_exact = (S_s.expm()) * H_s * ((-S_s).expm())
molde_s = sy_s * (ad_s**2 - a_s**2)
coef_exact = proyectar(Hp_exact - H_s, molde_s, Ncut=Nsmall - 4, Ntot=Nsmall)
print(f"\n[Verificacion con exp matricial EXACTA, Nsmall={Nsmall}]")
print(f"Coeficiente exacto de sigma_y*(a^dagger^2-a^2) en (e^S H e^-S - H) = {coef_exact:.6f}")
print(f"Esperado (BCH a 2do orden, deberia coincidir si g_z/omega_m<<1)   = {coef_bch2:.6f}")
print(f"Diferencia relativa exacta vs BCH-2 = "
      f"{abs(coef_exact - coef_bch2)/abs(coef_bch2):.4%}  "
      f"(corrige por ordenes superiores en g_z/omega_m={gz_d/om_m:.4f})")

# ------------------------------------------------------------
# 3) Comparacion final con Ecs. (9) y (11)
# ------------------------------------------------------------
coef_Eq9 = -2 * g_theory
coef_Eq11 = g_theory   # tal como se establece en el enunciado (ya en forma sigma_+- ... )
# Nota: nuestro "molde" esta en la base sigma_y; convertimos Eq(11) (dada en sigma_+-)
# a la misma base sigma_y para comparar consistentemente:
#   sigma_y = -i(sigma_+ - sigma_-)  =>  (sigma_+-sigma_-) = i*sigma_y
#   Ec.(11): +g (sigma_+-sigma_-)(a^2-a^dagger^2)... con signo tal como en el enunciado:
#   +g(sigma_+-sigma_-)(a^dagger^2-a^2) = +g * i*sigma_y * (a^dagger^2-a^2)
#   = i*g * sigma_y*(a^dagger^2-a^2)  -> coeficiente en base sigma_y (complejo) = i*g
coef_Eq11_en_sy = 1j * g_theory

print(f"\n=== COMPARACION FINAL ===")
print(f"Coeficiente medido (BCH-2, base sigma_y)   = {coef_bch2:.6f}")
print(f"Coeficiente medido (exacto, base sigma_y)  = {coef_exact:.6f}")
print(f"Ec.(9)  predice (base sigma_y)  = -2i*g = {-2j*g_theory}")
print(f"Ec.(11) predice (base sigma_y)  =  i*g  = {coef_Eq11_en_sy}")

np.savez("tarea8_resultados.npz", g_theory=g_theory, coef_C1=coef_C1, coef_bch2=coef_bch2,
         coef_C2=coef_C2, coef_exact=coef_exact)
print("\nGuardado: tarea8_resultados.npz")
