# -*- coding: utf-8 -*-
"""
Tarea 15: patada de polaron y paridad.

Qubit en estado BASE fisico. eps/g_eff=2 fijo, g_eff=0.36 FIJO
(Gamma2/kappa=4*g_eff^2=0.5184 fijo). Se varia g_z/omega_m en
{0.015, 0.03, 0.06, 0.12} manteniendo el producto g_z*g_x constante
(= g_eff*omega_m/2 = 180), ajustando g_x = g_eff*omega_m/(2*g_z).

En estado estacionario (tau_max=15/Gamma2, igual para todos los casos ya
que Gamma2 es fijo), se reporta paridad, poblacion impar, |<a^2>|, y
fidelidad con el gato par |C+>.

Hipotesis: 1 - paridad crece aproximadamente como (2*g_z/omega_m)^2.

Se reporta tambien la paridad del modelo efectivo (g_eff=2g fijo, sin
dependencia de g_z/omega_m) como referencia.

Muestreo estroboscopico. Validacion (traza, hermiticidad, positividad,
umbral -1e-8). NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, basis, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace, expect, coherent, Qobj)
from qutip.metrics import fidelity as state_fidelity

r_baseline = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0

g_eff_fijo = 0.36
eps_fijo = 2.0 * g_eff_fijo   # eps/g_eff=2
Gamma2_fijo = 4 * g_eff_fijo**2 / kap
tau_max = 15.0 / Gamma2_fijo
print(f"g_eff={g_eff_fijo}, eps={eps_fijo}, Gamma2/kappa={Gamma2_fijo:.5f}, tau_max={tau_max:.4f}")

x = kap / 2.0
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m
ImS1m = -D1m / (x**2 + D1m**2)
ImS1p = -D1p / (x**2 + D1p**2)

Na, Nb = 2, 25
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m

K_total = int(round(tau_max / T_m))
stride = max(1, K_total // 100)
tlist = np.arange(0, K_total + 1, stride) * T_m
print(f"n_puntos={len(tlist)}  tau_max_real={tlist[-1]:.4f}")

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)
P_op = Qobj(np.diag((-1.0) ** np.arange(Nb)))


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return tr, herm_err, min_eig, ok


# --- Modelo EFECTIVO de referencia (no depende de g_z/omega_m) ---
a = destroy(Nb)
adag = a.dag()
G2m = 2 * g_eff_fijo**2 * ReS2m
G2p = 2 * g_eff_fijo**2 * ReS2p
dk = g_eff_fijo**2 * (ImS2m + ImS2p)
chi = -2j * eps_fijo * g_eff_fijo / kap
H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
# tasas de un foton para el efectivo: usamos gx "de referencia" r=0.1 (irrelevante,
# G1 es minusculo de cualquier forma; se usa solo para no dejar el modelo sin canal 1-foton)
gx_ref = r_baseline * (gz_org / kappa_org)
G1m_ref = 2 * gx_ref**2 * (x / (x**2 + D1m**2))
G1p_ref = 2 * gx_ref**2 * (x / (x**2 + D1p**2))
diss_eff = [G1m_ref * lindblad_dissipator(a), G1p_ref * lindblad_dissipator(adag),
            G2m * lindblad_dissipator(a**2), G2p * lindblad_dissipator(adag**2)]
rho0_eff = thermal_dm(Nb, 0)
res_eff = mesolve(H_eff, rho0_eff, tlist, diss_eff, [adag * a], options=options)
rho_eff_ss = res_eff.states[-1]
paridad_eff = expect(P_op, rho_eff_ss)
a2_eff = expect(a * a, rho_eff_ss)
print(f"\nModelo EFECTIVO (referencia): paridad_ss={paridad_eff:.6f}  |<a^2>|={abs(a2_eff):.6f}")


def correr_full(gz_over_wm):
    gz = gz_over_wm * om_m
    gx = g_eff_fijo * om_m / (2 * gz)   # mantiene g_z*g_x = g_eff*omega_m/2 constante

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
         [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [eps_fijo * sx, lambda t, _: np.exp(+1j * omega_q_dim * t)],
         [eps_fijo * sx, lambda t, _: np.exp(-1j * omega_q_dim * t)]]
    diss = [kap * lindblad_dissipator(sm),
            (n_th + 1) * Gam_m * lindblad_dissipator(b),
            n_th * Gam_m * lindblad_dissipator(bd)]
    qubit_base = basis(Na, 1)
    rho0 = tensor(qubit_base * qubit_base.dag(), thermal_dm(Nb, 0))

    print(f"\nCorriendo g_z/omega_m={gz_over_wm}  (g_z={gz:.4f}, g_x={gx:.4f}, "
          f"g_z*g_x={gz*gx:.4f})...")
    res = mesolve(H, rho0, tlist, diss, [], options=options)
    print("... listo.")

    rho_final = res.states[-1]
    rho_osc_ss = ptrace(rho_final, 1)

    tr, herm_err, min_eig, ok = validar_estado(rho_osc_ss)
    print(f"  Validacion: tr={tr:.3e} herm={herm_err:.3e} mineig={min_eig:.3e}  "
          f"{'OK' if ok else 'FALLA umbral -1e-8'}")

    paridad = expect(P_op, rho_osc_ss)
    poblacion_impar = (1 - paridad) / 2
    a2 = expect(a * a, rho_osc_ss)

    alpha = np.sqrt(a2)
    cat_plus = (coherent(Nb, alpha) + coherent(Nb, -alpha)).unit()
    F_cat = state_fidelity(rho_osc_ss, cat_plus * cat_plus.dag())

    return dict(gz_over_wm=gz_over_wm, gz=gz, gx=gx, paridad=paridad,
                poblacion_impar=poblacion_impar, abs_a2=abs(a2), a2=a2, F_cat=F_cat,
                min_eig=min_eig, ok=ok)


resultados = []
for gz_over_wm in [0.015, 0.03, 0.06, 0.12]:
    r_ = correr_full(gz_over_wm)
    print(f"  paridad={r_['paridad']:.6f}  poblacion_impar={r_['poblacion_impar']:.6f}  "
          f"|<a^2>|={r_['abs_a2']:.6f}  F_cat={r_['F_cat']:.6f}")
    resultados.append(r_)

print("\n=== TABLA RESUMEN (Tarea 15) ===")
print(f"{'gz/wm':>7} {'(2gz/wm)^2':>11} {'paridad':>9} {'1-paridad':>10} {'pob_impar':>10} "
      f"{'|<a^2>|':>9} {'F_cat':>8}")
for r_ in resultados:
    pred = (2 * r_['gz_over_wm']) ** 2
    print(f"{r_['gz_over_wm']:>7.3f} {pred:>11.5f} {r_['paridad']:>9.5f} "
          f"{1-r_['paridad']:>10.5f} {r_['poblacion_impar']:>10.5f} {r_['abs_a2']:>9.5f} "
          f"{r_['F_cat']:>8.5f}")

print(f"\nParidad modelo EFECTIVO (referencia, sin dependencia de g_z/omega_m) = {paridad_eff:.6f}  "
      f"(1-paridad_eff = {1-paridad_eff:.3e})")

# Ajuste: 1-paridad vs (g_z/omega_m)^2
x2 = np.array([(r_['gz_over_wm']) ** 2 for r_ in resultados])
y = np.array([1 - r_['paridad'] for r_ in resultados])
A = np.vstack([x2, np.ones_like(x2)]).T
slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"\nAjuste lineal: 1-paridad = {slope:.5f} * (g_z/omega_m)^2 + {intercept:.5f}")
print(f"Prediccion de la hipotesis (pendiente=4, sin intercepto): 1-paridad = 4*(g_z/omega_m)^2")

np.savez("tarea15_resultados.npz",
         gz_over_wm=np.array([r_['gz_over_wm'] for r_ in resultados]),
         paridad=np.array([r_['paridad'] for r_ in resultados]),
         poblacion_impar=np.array([r_['poblacion_impar'] for r_ in resultados]),
         abs_a2=np.array([r_['abs_a2'] for r_ in resultados]),
         F_cat=np.array([r_['F_cat'] for r_ in resultados]),
         paridad_eff=paridad_eff, slope=slope, intercept=intercept)
print("\nGuardado: tarea15_resultados.npz")
