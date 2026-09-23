# -*- coding: utf-8 -*-
"""
Tarea 16: es el "piso" en <n> (eps=0, qubit en base fisica, ronda 4) un
artefacto numerico o un efecto fisico real?

Se repite la Tarea 13 (caso qubit base, eps=0, tau_max=30, estroboscopico)
variando (atol,rtol) en {(1e-10,1e-8),(1e-12,1e-10),(1e-14,1e-12)} y
Nb en {16,24}. Se reporta <n> final y la pendiente de un ajuste lineal de
<n>(t) en la SEGUNDA MITAD de la ventana.

Si la pendiente cae con la tolerancia -> piso numerico.
Si no cambia -> piso fisico; en ese caso se aisla el termino causante
corriendo con g_x=0 (solo longitudinal) y g_z=0 (solo transversal).

Validacion (traza, hermiticidad, positividad, umbral -1e-8).
NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, basis, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace, expect)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
Na = 2
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m

tau_max = 30.0
K_total = int(round(tau_max / T_m))
stride = max(1, K_total // 100)
tlist = np.arange(0, K_total + 1, stride) * T_m
print(f"n_puntos={len(tlist)}  tau_max_real={tlist[-1]:.4f}")


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return tr, herm_err, min_eig, ok


def correr(Nb, atol, rtol, gx_val, gz_val, etiqueta):
    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb))
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))
    H0 = 0.5 * omega_q_dim * sz
    terms = [H0]
    if gx_val != 0:
        terms += [[gx_val * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
                  [gx_val * sx * bd, lambda t, _: np.exp(+1j * om_m * t)]]
    if gz_val != 0:
        terms += [[gz_val * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
                  [gz_val * sz * bd, lambda t, _: np.exp(+1j * om_m * t)]]
    H = terms
    diss = [kap * lindblad_dissipator(sm),
            (n_th + 1) * Gam_m * lindblad_dissipator(b),
            n_th * Gam_m * lindblad_dissipator(bd)]
    qubit_base = basis(Na, 1)
    rho0 = tensor(qubit_base * qubit_base.dag(), thermal_dm(Nb, 0))

    options = Options(nsteps=2_000_000, atol=atol, rtol=rtol, store_states=True)
    res = mesolve(H, rho0, tlist, diss, [], options=options)
    n_t = np.array([expect(bd * b, s) for s in res.states])

    # validacion en 3 instantes
    idxs = [0, len(tlist)//2, len(tlist)-1]
    oks = []
    for idx in idxs:
        rho_osc = ptrace(res.states[idx], 1)
        tr, he, me, ok = validar_estado(rho_osc)
        oks.append(ok)

    # ajuste lineal en la segunda mitad
    half = len(tlist) // 2
    coefs = np.polyfit(tlist[half:], n_t[half:], 1)
    pendiente = coefs[0]

    print(f"  [{etiqueta}] Nb={Nb} atol={atol:.0e} rtol={rtol:.0e}  "
          f"n_final={n_t[-1]:.5e}  pendiente(2da mitad)={pendiente:.4e}  "
          f"validacion={'OK' if all(oks) else 'FALLA'}")
    return n_t[-1], pendiente, all(oks)


print("\n=== Parte 1: escaneo de tolerancia y Nb (H completo, gx y gz ambos activos) ===")
resultados_tol = []
for Nb in [16, 24]:
    for atol, rtol in [(1e-10, 1e-8), (1e-12, 1e-10), (1e-14, 1e-12)]:
        n_final, pend, ok = correr(Nb, atol, rtol, gx_dim, gz_d, "completo")
        resultados_tol.append((Nb, atol, rtol, n_final, pend, ok))

print("\n=== TABLA (Parte 1) ===")
print(f"{'Nb':>4} {'atol':>8} {'rtol':>8} {'n_final':>12} {'pendiente':>12} {'ok':>5}")
for Nb, atol, rtol, n_final, pend, ok in resultados_tol:
    print(f"{Nb:>4} {atol:>8.0e} {rtol:>8.0e} {n_final:>12.5e} {pend:>12.4e} {str(ok):>5}")

pendientes = [row[4] for row in resultados_tol]
variacion_relativa = (max(pendientes) - min(pendientes)) / (abs(np.mean(pendientes)) + 1e-30)
print(f"\nVariacion relativa de la pendiente entre configuraciones = {variacion_relativa:.2%}")
es_numerico = variacion_relativa > 0.5   # criterio: cambia sustancialmente con tolerancia
print("Diagnostico:", "NUMERICO (la pendiente cambia con la tolerancia)" if es_numerico
      else "FISICO (la pendiente no cambia apreciablemente con la tolerancia/Nb)")

# ------------------------------------------------------------
# Parte 2: aislar el termino causante (siempre, para completitud,
# usando la tolerancia mas fina probada)
# ------------------------------------------------------------
print("\n=== Parte 2: aislar termino (g_x=0 vs g_z=0), atol=1e-14 rtol=1e-12, Nb=24 ===")
n_gz_solo, pend_gz_solo, ok1 = correr(24, 1e-14, 1e-12, 0.0, gz_d, "solo g_z (longitudinal)")
n_gx_solo, pend_gx_solo, ok2 = correr(24, 1e-14, 1e-12, gx_dim, 0.0, "solo g_x (transversal)")

print("\n=== TABLA (Parte 2) ===")
print(f"{'caso':>25} {'n_final':>12} {'pendiente':>12}")
print(f"{'solo g_z (longitudinal)':>25} {n_gz_solo:>12.5e} {pend_gz_solo:>12.4e}")
print(f"{'solo g_x (transversal)':>25} {n_gx_solo:>12.5e} {pend_gx_solo:>12.4e}")
n_completo_ref = resultados_tol[-1][3]  # Nb=24, tolerancia mas fina
print(f"{'completo (referencia)':>25} {n_completo_ref:>12.5e}")

np.savez("tarea16_resultados.npz",
         resultados_tol=np.array(resultados_tol, dtype=object),
         n_gz_solo=n_gz_solo, pend_gz_solo=pend_gz_solo,
         n_gx_solo=n_gx_solo, pend_gx_solo=pend_gx_solo)
print("\nGuardado: tarea16_resultados.npz")
