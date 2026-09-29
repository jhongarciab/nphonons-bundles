# -*- coding: utf-8 -*-
"""
Tarea 14: repite la Tarea 11 (dominio de validez) con el qubit en su
estado BASE fisico (basis(Na,1)), |alpha|^2=eps/g_eff=2 fijo, g_z escalado
por {0.125, 0.25, 0.5, 1} (g_x FIJO), g_eff=2g (Tareas 7-8). INCLUYE
g_z x0.125 (omitido en la Tarea 11 por costo) -- se usa Nb=20 y menos
puntos de salida para acotar tiempo, tal como indica el enunciado.

Se anade al modelo efectivo el termino de Lamb (Ec. 19 del paper):
    delta_1 * a^dagger a,   delta_1 = g_x^2 [Im S_{1-} + Im S_{1+}]
y se reportan resultados CON y SIN ese termino.

Metricas: F_min, F integrada en [0,10/Gamma2], t90(completo)/t90(efectivo),
Gamma2/kappa = 4 g_eff^2/kappa.

Muestreo estroboscopico. Validacion (traza, hermiticidad, positividad,
umbral -1e-8). NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, basis, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace)
from qutip.metrics import fidelity as state_fidelity

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_baseline = gz_org / kappa_org
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_fijo = r * gz_baseline

x = kap / 2.0
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m
ImS1m = -D1m / (x**2 + D1m**2)
ImS1p = -D1p / (x**2 + D1p**2)

Na = 2
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m
alpha2_target = 2.0

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)


def tlist_estrobo(tau_max, n_puntos):
    K_total = int(round(tau_max / T_m))
    stride = max(1, K_total // n_puntos)
    m = np.arange(0, K_total + 1, stride)
    return m * T_m


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return min_eig, ok


def correr_caso(gz_scale, Nb, n_puntos):
    gz = gz_scale * gz_baseline
    gx = gx_fijo
    g_eff = 2 * (gz * gx / om_m)
    eps = alpha2_target * g_eff
    Gamma2 = 4 * g_eff**2 / kap
    tau_max = 10.0 / Gamma2
    tlist = tlist_estrobo(tau_max, n_puntos)

    G1m_gx = 2 * gx**2 * (x / (x**2 + D1m**2))
    G1p_gx = 2 * gx**2 * (x / (x**2 + D1p**2))
    delta_1 = gx**2 * (ImS1m + ImS1p)

    # --- Modelo COMPLETO, qubit inicial BASE fisico ---
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
         [eps * sx, lambda t, _: np.exp(+1j * omega_q_dim * t)],
         [eps * sx, lambda t, _: np.exp(-1j * omega_q_dim * t)]]
    diss_full = [kap * lindblad_dissipator(sm),
                 (n_th + 1) * Gam_m * lindblad_dissipator(b),
                 n_th * Gam_m * lindblad_dissipator(bd)]
    qubit_base = basis(Na, 1)
    rho0_full = tensor(qubit_base * qubit_base.dag(), thermal_dm(Nb, 0))
    res_full = mesolve(H, rho0_full, tlist, diss_full, [bd * b], options=options)
    nb_full = res_full.expect[0]
    states_full_red = [ptrace(rho, 1) for rho in res_full.states]

    # --- Modelo efectivo, SIN Lamb ---
    a = destroy(Nb)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m
    G2p = 2 * g_eff**2 * ReS2p
    dk = g_eff**2 * (ImS2m + ImS2p)
    chi = -2j * eps * g_eff / kap
    diss_eff = [G1m_gx * lindblad_dissipator(a), G1p_gx * lindblad_dissipator(adag),
                G2m * lindblad_dissipator(a**2), G2p * lindblad_dissipator(adag**2)]
    rho0_eff = thermal_dm(Nb, 0)

    def correr_eff(con_lamb):
        H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
        if con_lamb:
            H_eff = H_eff + delta_1 * (adag * a)
        res = mesolve(H_eff, rho0_eff, tlist, diss_eff, [adag * a], options=options)
        return res.expect[0], res.states

    nb_eff_sin, states_eff_sin = correr_eff(False)
    nb_eff_con, states_eff_con = correr_eff(True)

    def metricas(states_eff, nb_eff):
        fidelity_t = np.array([state_fidelity(states_full_red[i], states_eff[i])
                                for i in range(len(tlist))])
        F_min = fidelity_t.min()
        F_int = np.trapz(fidelity_t, tlist) / (tlist[-1] - tlist[0])

        def t90(t, n):
            n_ss = np.mean(n[-6:])
            if n_ss < 1e-6:
                return 0.0
            umbral = 0.9 * n_ss
            idx = np.argmax(n >= umbral)
            if n[idx] < umbral or idx == 0:
                return np.nan
            t0, t1, n0, n1 = t[idx-1], t[idx], n[idx-1], n[idx]
            return t0 + (umbral - n0) * (t1 - t0) / (n1 - n0)

        return F_min, F_int, t90(tlist, nb_full), t90(tlist, nb_eff)

    Fmin_sin, Fint_sin, t90f, t90e_sin = metricas(states_eff_sin, nb_eff_sin)
    Fmin_con, Fint_con, _, t90e_con = metricas(states_eff_con, nb_eff_con)

    idxs = [0, len(tlist)//2, len(tlist)-1]
    minigs = [validar_estado(states_full_red[i])[0] for i in idxs]
    oks = [validar_estado(states_full_red[i])[1] for i in idxs]

    return dict(gz_scale=gz_scale, g_eff=g_eff, eps=eps, Gamma2=Gamma2, tau_max=tau_max,
                delta_1=delta_1, n_puntos=len(tlist),
                Fmin_sin=Fmin_sin, Fint_sin=Fint_sin, t90f=t90f, t90e_sin=t90e_sin,
                Fmin_con=Fmin_con, Fint_con=Fint_con, t90e_con=t90e_con,
                minigs=minigs, oks=oks)


casos = [
    (0.125, 20, 50),
    (0.25, 25, 80),
    (0.5, 25, 100),
    (1.0, 25, 120),
]

resultados = []
for gz_scale, Nb, npts in casos:
    print(f"\n=== g_z x{gz_scale} (Nb={Nb}, n_puntos_pedidos={npts}) ===")
    r_ = correr_caso(gz_scale, Nb, npts)
    print(f"  g_eff={r_['g_eff']:.5f}  eps={r_['eps']:.5f}  Gamma2={r_['Gamma2']:.5f}  "
          f"delta_1={r_['delta_1']:.3e}  tau_max={r_['tau_max']:.3f}  n_puntos={r_['n_puntos']}")
    print(f"  SIN Lamb: F_min={r_['Fmin_sin']:.5f}  F_int={r_['Fint_sin']:.5f}  "
          f"t90f/t90e={r_['t90f']/r_['t90e_sin'] if r_['t90e_sin'] else float('nan'):.4f}")
    print(f"  CON Lamb: F_min={r_['Fmin_con']:.5f}  F_int={r_['Fint_con']:.5f}  "
          f"t90f/t90e={r_['t90f']/r_['t90e_con'] if r_['t90e_con'] else float('nan'):.4f}")
    print(f"  Validacion mineig [0,mitad,final]: {r_['minigs']}  oks={r_['oks']}")
    resultados.append(r_)

print("\n=== TABLA RESUMEN (Tarea 14) ===")
hdr = (f"{'gz_x':>6} {'Gamma2/k':>9} | {'Fmin_sin':>9} {'Fint_sin':>9} {'t90rat_sin':>11} | "
       f"{'Fmin_con':>9} {'Fint_con':>9} {'t90rat_con':>11}")
print(hdr)
for r_ in resultados:
    t90rs = r_['t90f']/r_['t90e_sin'] if r_['t90e_sin'] else float('nan')
    t90rc = r_['t90f']/r_['t90e_con'] if r_['t90e_con'] else float('nan')
    print(f"{r_['gz_scale']:>6.3f} {r_['Gamma2']:>9.5f} | {r_['Fmin_sin']:>9.5f} {r_['Fint_sin']:>9.5f} "
          f"{t90rs:>11.4f} | {r_['Fmin_con']:>9.5f} {r_['Fint_con']:>9.5f} {t90rc:>11.4f}")

# Ajuste ley de potencia: 1-F_min vs Gamma2/kappa (con Lamb, la version mas completa)
G2 = np.array([r_['Gamma2'] for r_ in resultados])
one_minus_F = np.array([1 - r_['Fmin_con'] for r_ in resultados])
logG = np.log(G2)
logF = np.log(one_minus_F)
A = np.vstack([logG, np.ones_like(logG)]).T
slope, intercept = np.linalg.lstsq(A, logF, rcond=None)[0]
print(f"\nAjuste ley de potencia (con Lamb): 1-F_min = {np.exp(intercept):.5f} * (Gamma2/kappa)^{slope:.4f}")

cumple = [r_ for r_ in resultados if r_['Fmin_con'] > 0.99]
if cumple:
    m_ = min(cumple, key=lambda r_: r_['Gamma2'])
    print(f"F_min>0.99 se alcanza desde Gamma2/kappa <= {m_['Gamma2']:.5f}")
else:
    Gamma2_pred_099 = np.exp((np.log(0.01) - intercept) / slope)
    print(f"Ningun caso alcanza F_min>0.99. Extrapolacion de la ley de potencia: "
          f"Gamma2/kappa ~ {Gamma2_pred_099:.5f} para F_min=0.99")

np.savez("tarea14_resultados.npz",
         gz_scales=np.array([r_['gz_scale'] for r_ in resultados]),
         Gamma2s=G2,
         Fmin_sin=np.array([r_['Fmin_sin'] for r_ in resultados]),
         Fint_sin=np.array([r_['Fint_sin'] for r_ in resultados]),
         Fmin_con=np.array([r_['Fmin_con'] for r_ in resultados]),
         Fint_con=np.array([r_['Fint_con'] for r_ in resultados]),
         slope=slope, intercept=intercept)
print("\nGuardado: tarea14_resultados.npz")
