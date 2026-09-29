# -*- coding: utf-8 -*-
"""
Tarea 19: origen del calentamiento (eps=0, propagador de Floquet).

Para cada g_z/omega_m en {0.015, 0.03, 0.06, 0.12} (g_z*g_x constante,
= 360, elegido para que g_z/omega_m=0.06 coincida EXACTAMENTE con los
parametros del paper -g_z=60, g_x=6- usados en la Tarea 18(i)), sin
drive y sin retuneo (omega_q=2*omega_m siempre), se calcula el propagador
de Floquet de un periodo, se obtiene el punto fijo (<n>) y se identifica
el modo de relajacion dominante hacia ese punto fijo proyectando los
autovectores sobre el operador numero n=a^dagger a (analogo a como la
Tarea 18(ii) identifico el modo de paridad). Esa tasa (lambda, parte
real) se reporta como "tasa de calentamiento" para ese punto.

Se ajusta la tasa contra (g_z/omega_m)^2 y contra g_x^2 por separado
(g_x = 360/g_z, por lo que estos dos ajustes prueban escalamientos
opuestos) y se compara con gamma_phase-flip de la Tarea 18(ii)
(6.912e-4, en el punto del paper con drive).

Validacion (traza, hermiticidad, positividad, umbral -1e-8).
NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, Options, propagator,
                    vector_to_operator, Qobj, ptrace)

r_baseline = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m
Na = 2

# Producto constante, calibrado para reproducir EXACTAMENTE la Tarea 18(i)
# en g_z/omega_m=0.06 (g_z=60, g_x=6, g_z*g_x=360)
producto_gzgx = 360.0

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return tr, herm_err, min_eig, ok


def correr_caso(gz_over_wm, Nb=16, n_modos=6):
    gz = gz_over_wm * om_m
    gx = producto_gzgx / gz

    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb))
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))
    H0 = 0.5 * omega_q_dim * sz
    H = [H0,
         [gx * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gx * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [gz * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)]]
    c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

    print(f"\n=== g_z/omega_m={gz_over_wm} (g_z={gz:.3f}, g_x={gx:.3f}, "
          f"g_z*g_x={gz*gx:.3f}) ===")
    U = propagator(H, T_m, c_ops, options=opts)

    evals, evecs = U.eigenstates()
    orden = np.argsort(-np.abs(evals))
    evals = evals[orden]
    evecs = [evecs[i] for i in orden]

    rho_fixed_raw = vector_to_operator(evecs[0])
    rho_fixed = (rho_fixed_raw + rho_fixed_raw.dag()) / 2
    rho_fixed = rho_fixed / rho_fixed.tr()
    tr, herm_err, min_eig, ok = validar_estado(rho_fixed)
    print(f"  Punto fijo: tr={tr:.6f} herm={herm_err:.3e} mineig={min_eig:.3e}  "
          f"{'OK' if ok else 'FALLA'}")

    n_op_full = bd * b
    n_fixed = (n_op_full * rho_fixed).tr().real

    # Identificar el modo dominante de relajacion de <n>: proyectar
    # autovectores sobre n_op_full (extendido al espacio completo)
    n_op_ext = n_op_full  # ya esta en el espacio completo qubit+osc
    overlaps = []
    for k in range(1, n_modos):
        Xk = vector_to_operator(evecs[k])
        overlap = (n_op_ext.dag() * Xk).tr()
        overlaps.append(abs(overlap))
    idx = int(np.argmax(overlaps)) + 1
    mu_dom = evals[idx]
    lam_dom = -np.log(mu_dom) / T_m if abs(mu_dom) > 1e-14 else np.inf

    print(f"  <n>_fijo = {n_fixed:.6f}")
    print(f"  Modo dominante de calentamiento: k={idx}, mu={mu_dom:.6f}, "
          f"lambda={lam_dom.real:.6e} (+/-{lam_dom.imag:.3e}j)")

    return dict(gz_over_wm=gz_over_wm, gz=gz, gx=gx, n_fixed=n_fixed,
                lambda_heat=lam_dom.real, ok=ok)


resultados = []
for gz_over_wm in [0.015, 0.03, 0.06, 0.12]:
    r_ = correr_caso(gz_over_wm)
    resultados.append(r_)

print("\n=== TABLA RESUMEN (Tarea 19) ===")
print(f"{'gz/wm':>7} {'g_z':>8} {'g_x':>8} {'n_fijo':>9} {'lambda_heat':>12}")
for r_ in resultados:
    print(f"{r_['gz_over_wm']:>7.3f} {r_['gz']:>8.3f} {r_['gx']:>8.3f} "
          f"{r_['n_fixed']:>9.5f} {r_['lambda_heat']:>12.5e}")

# Ajustes: lambda_heat vs (gz/wm)^2 y vs gx^2
gz_wm = np.array([r_['gz_over_wm'] for r_ in resultados])
gx_arr = np.array([r_['gx'] for r_ in resultados])
lam = np.array([r_['lambda_heat'] for r_ in resultados])

def ajustar(xvar, y, nombre):
    A = np.vstack([xvar, np.ones_like(xvar)]).T
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"  Ajuste vs {nombre}: lambda_heat = {slope:.5e}*x + {intercept:.5e}   R^2={r2:.5f}")
    return slope, intercept, r2

print("\n--- Ajuste 1: lambda_heat vs (g_z/omega_m)^2 ---")
s1, i1, r2_1 = ajustar(gz_wm**2, lam, "(gz/wm)^2")

print("--- Ajuste 2: lambda_heat vs g_x^2 ---")
s2, i2, r2_2 = ajustar(gx_arr**2, lam, "gx^2")

mejor = "(g_z/omega_m)^2" if r2_1 > r2_2 else "g_x^2"
print(f"\nMejor descriptor: {mejor}  (R^2: (gz/wm)^2={r2_1:.5f} vs gx^2={r2_2:.5f})")

gamma_phase_flip_paper = 6.912e-4
lam_en_006 = [r_['lambda_heat'] for r_ in resultados if abs(r_['gz_over_wm'] - 0.06) < 1e-9][0]
print(f"\nComparacion con gamma_phase-flip de la Tarea 18(ii) (eps=1.44, con drive) "
      f"= {gamma_phase_flip_paper:.4e}")
print(f"lambda_heat en g_z/omega_m=0.06 (eps=0, sin drive) = {lam_en_006:.4e}")
print(f"Razon = {lam_en_006/gamma_phase_flip_paper:.4f}")

np.savez("tarea19_resultados.npz",
         gz_over_wm=gz_wm, gx=gx_arr, n_fixed=np.array([r_['n_fixed'] for r_ in resultados]),
         lambda_heat=lam, s1=s1, i1=i1, r2_1=r2_1, s2=s2, i2=i2, r2_2=r2_2)
print("\nGuardado: tarea19_resultados.npz")
