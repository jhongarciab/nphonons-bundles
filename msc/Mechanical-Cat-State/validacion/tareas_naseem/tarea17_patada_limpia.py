# -*- coding: utf-8 -*-
"""
Tarea 17: version limpia de la Tarea 15 (patada polaronica).

Mismo barrido: g_eff=0.36 fijo, eps/g_eff=2 fijo, g_z*g_x constante
(=g_eff*omega_m/2), g_z/omega_m en {0.015, 0.03, 0.06, 0.12}. Pero ahora:

(a) Se compensa el corrimiento de Lamb/dispersivo retonando
    omega_q = omega_d = 2*(omega_m + delta_1),
    delta_1 = g_x^2 [Im S_{1-} + Im S_{1+}]  (calculado con el g_x de
    cada caso), tanto en el modelo COMPLETO como en las formulas de
    detuning D2m, D2p usadas para el modelo EFECTIVO.
(b) El modelo efectivo incluye el termino delta_1 * a^dagger a en H_eff.
(c) Se mide la TASA de decaimiento de la paridad (phase-flip), no el
    valor final: se corre hasta tau=40/Gamma2, se ajusta linealmente la
    paridad en la ventana [15/Gamma2, 40/Gamma2], y se reporta la
    pendiente -dP/dt, junto con |<a^2>| promedio en esa ventana.

Se compara -dP/dt del completo con la del modelo efectivo (que solo
tiene decaimiento de paridad por gamma y Gamma_1), y se ajusta la
diferencia (completo - efectivo) contra (g_z/omega_m)^2 con intercepto
libre. Si el intercepto es ~0, el canal de "patada polaronica" queda
confirmado como el origen del exceso.

Qubit SIEMPRE en estado BASE fisico. Muestreo estroboscopico. Validacion
(traza, hermiticidad, positividad, umbral -1e-8). NO modifica scripts
originales.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, basis, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace, expect, Qobj)

r_baseline = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0

g_eff_fijo = 0.36
eps_fijo = 2.0 * g_eff_fijo
Gamma2_fijo = 4 * g_eff_fijo**2 / kap
tau_max = 40.0 / Gamma2_fijo
ventana_ini = 15.0 / Gamma2_fijo
ventana_fin = 40.0 / Gamma2_fijo
print(f"g_eff={g_eff_fijo}, eps={eps_fijo}, Gamma2/kappa={Gamma2_fijo:.5f}, "
      f"tau_max={tau_max:.4f}, ventana=[{ventana_ini:.3f},{ventana_fin:.3f}]")

x = kap / 2.0
Na, Nb = 2, 25
T_m = 2 * np.pi / om_m

K_total = int(round(tau_max / T_m))
stride = max(1, K_total // 130)
tlist = np.arange(0, K_total + 1, stride) * T_m
print(f"n_puntos={len(tlist)}  tau_max_real={tlist[-1]:.4f}")

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)
P_op = Qobj(np.diag((-1.0) ** np.arange(Nb)))


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return min_eig, ok


def pendiente_ventana(t, y):
    mask = (t >= ventana_ini) & (t <= ventana_fin)
    coefs = np.polyfit(t[mask], y[mask], 1)
    return coefs[0], y[mask]


def correr_caso(gz_over_wm):
    gz = gz_over_wm * om_m
    gx = g_eff_fijo * om_m / (2 * gz)

    D1m, D1p = om_m, 3 * om_m
    ImS1m = -D1m / (x**2 + D1m**2)
    ImS1p = -D1p / (x**2 + D1p**2)
    delta_1 = gx**2 * (ImS1m + ImS1p)

    omega_q_case = 2 * (om_m + delta_1)   # (a) compensacion del corrimiento
    omega_d_case = omega_q_case

    D2m = omega_q_case - 2 * om_m         # = 2*delta_1 por construccion
    D2p = omega_q_case + 2 * om_m
    ReS2m = x / (x**2 + D2m**2)
    ReS2p = x / (x**2 + D2p**2)
    ImS2m = -D2m / (x**2 + D2m**2)
    ImS2p = -D2p / (x**2 + D2p**2)
    G1m_gx = 2 * gx**2 * (x / (x**2 + D1m**2))
    G1p_gx = 2 * gx**2 * (x / (x**2 + D1p**2))

    # --- Modelo COMPLETO, qubit BASE, con omega_q/omega_d retonados ---
    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb))
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))
    H0 = 0.5 * omega_q_case * sz
    H = [H0,
         [gx * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gx * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [gz * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [eps_fijo * sx, lambda t, _: np.exp(+1j * omega_d_case * t)],
         [eps_fijo * sx, lambda t, _: np.exp(-1j * omega_d_case * t)]]
    diss_full = [kap * lindblad_dissipator(sm),
                 (n_th + 1) * Gam_m * lindblad_dissipator(b),
                 n_th * Gam_m * lindblad_dissipator(bd)]
    qubit_base = basis(Na, 1)
    rho0_full = tensor(qubit_base * qubit_base.dag(), thermal_dm(Nb, 0))

    print(f"\nCorriendo COMPLETO g_z/omega_m={gz_over_wm} (g_z={gz:.3f}, g_x={gx:.3f}, "
          f"delta_1={delta_1:.4e}, omega_q_case={omega_q_case:.4f})...")
    res_full = mesolve(H, rho0_full, tlist, diss_full, [], options=options)
    print("... listo.")

    states_red = [ptrace(rho, 1) for rho in res_full.states]
    paridad_full_t = np.array([expect(P_op, s) for s in states_red])
    a2_full_t = np.array([expect(destroy(Nb) * destroy(Nb), s) for s in states_red])

    idxs = [0, len(tlist)//2, len(tlist)-1]
    minigs = [validar_estado(states_red[i])[0] for i in idxs]
    oks = [validar_estado(states_red[i])[1] for i in idxs]
    print(f"  Validacion mineig [0,mitad,final]: {minigs}  oks={oks}")

    pend_full, par_ventana_full = pendiente_ventana(tlist, paridad_full_t)
    a2_ventana_mask = (tlist >= ventana_ini) & (tlist <= ventana_fin)
    a2_prom_full = np.mean(np.abs(a2_full_t[a2_ventana_mask]))

    # --- Modelo EFECTIVO, con delta_1*a^dagger a y detunings consistentes ---
    a = destroy(Nb)
    adag = a.dag()
    G2m = 2 * g_eff_fijo**2 * ReS2m
    G2p = 2 * g_eff_fijo**2 * ReS2p
    dk = g_eff_fijo**2 * (ImS2m + ImS2p)
    chi = -2j * eps_fijo * g_eff_fijo / kap
    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2 + delta_1 * (adag * a)
    diss_eff = [G1m_gx * lindblad_dissipator(a), G1p_gx * lindblad_dissipator(adag),
                G2m * lindblad_dissipator(a**2), G2p * lindblad_dissipator(adag**2)]
    rho0_eff = thermal_dm(Nb, 0)
    res_eff = mesolve(H_eff, rho0_eff, tlist, diss_eff, [], options=options)
    paridad_eff_t = np.array([expect(P_op, s) for s in res_eff.states])
    a2_eff_t = np.array([expect(a * a, s) for s in res_eff.states])
    pend_eff, par_ventana_eff = pendiente_ventana(tlist, paridad_eff_t)
    a2_prom_eff = np.mean(np.abs(a2_eff_t[a2_ventana_mask]))

    return dict(gz_over_wm=gz_over_wm, gz=gz, gx=gx, delta_1=delta_1,
                pend_full=pend_full, pend_eff=pend_eff,
                a2_prom_full=a2_prom_full, a2_prom_eff=a2_prom_eff,
                minigs=minigs, oks=oks)


resultados = []
for gz_over_wm in [0.015, 0.03, 0.06, 0.12]:
    r_ = correr_caso(gz_over_wm)
    print(f"  -dP/dt full={-r_['pend_full']:.5e}  -dP/dt eff={-r_['pend_eff']:.5e}  "
          f"diff={-r_['pend_full']-(-r_['pend_eff']):.5e}  "
          f"|<a^2>|_full={r_['a2_prom_full']:.4f}  |<a^2>|_eff={r_['a2_prom_eff']:.4f}")
    resultados.append(r_)

print("\n=== TABLA RESUMEN (Tarea 17) ===")
print(f"{'gz/wm':>7} {'delta_1':>10} {'-dP/dt full':>12} {'-dP/dt eff':>11} {'diff':>11} "
      f"{'|a^2|full':>10} {'|a^2|eff':>9}")
for r_ in resultados:
    diff = -r_['pend_full'] - (-r_['pend_eff'])
    print(f"{r_['gz_over_wm']:>7.3f} {r_['delta_1']:>10.4e} {-r_['pend_full']:>12.5e} "
          f"{-r_['pend_eff']:>11.5e} {diff:>11.5e} {r_['a2_prom_full']:>10.4f} "
          f"{r_['a2_prom_eff']:>9.4f}")

# Ajuste: diff vs (gz/wm)^2 con intercepto libre
x2 = np.array([(r_['gz_over_wm']) ** 2 for r_ in resultados])
diffs = np.array([-r_['pend_full'] - (-r_['pend_eff']) for r_ in resultados])
A = np.vstack([x2, np.ones_like(x2)]).T
(slope, intercept), residuals, rank, sv = np.linalg.lstsq(A, diffs, rcond=None)
pred = A @ np.array([slope, intercept])
ss_res = np.sum((diffs - pred) ** 2)
ss_tot = np.sum((diffs - diffs.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

# error estandar de los coeficientes (aprox, minimos cuadrados)
n_pts = len(diffs)
dof = max(1, n_pts - 2)
sigma2 = ss_res / dof
cov = sigma2 * np.linalg.inv(A.T @ A)
slope_err = np.sqrt(cov[0, 0])
intercept_err = np.sqrt(cov[1, 1])

print(f"\nAjuste: diff = {slope:.5e} (+/-{slope_err:.2e}) * (g_z/omega_m)^2 "
      f"+ {intercept:.5e} (+/-{intercept_err:.2e})   R^2={r2:.5f}")
print("Canal de patada polaronica confirmado (intercepto ~0):" ,
      "SI" if abs(intercept) < 2 * intercept_err else "NO / no concluyente")

np.savez("tarea17_resultados.npz",
         gz_over_wm=np.array([r_['gz_over_wm'] for r_ in resultados]),
         pend_full=np.array([r_['pend_full'] for r_ in resultados]),
         pend_eff=np.array([r_['pend_eff'] for r_ in resultados]),
         a2_prom_full=np.array([r_['a2_prom_full'] for r_ in resultados]),
         a2_prom_eff=np.array([r_['a2_prom_eff'] for r_ in resultados]),
         slope=slope, slope_err=slope_err, intercept=intercept, intercept_err=intercept_err, r2=r2)
print("\nGuardado: tarea17_resultados.npz")
