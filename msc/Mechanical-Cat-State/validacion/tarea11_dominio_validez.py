# -*- coding: utf-8 -*-
"""
Tarea 11: dominio de validez del modelo efectivo (g_eff=2g).

Se fija |alpha|^2 = eps/g_eff = 2 (mismo punto de operacion estacionario en
todos los casos) y se escala g_z por {0.125, 0.25, 0.5, 1} (g_x FIJO, como
en la Tarea 7), re-escalando eps para mantener eps/g_eff=2 en cada caso.

Para cada caso se reporta:
  - Gamma2 = 4*g_eff^2/kappa (tasa de dos fonones, unidades de kappa)
  - t90(completo)/t90(modelo B)
  - fidelidad minima F_min
  - fidelidad integrada (1/T) integral F dt sobre kappa*t en [0, 10/Gamma2]

Muestreo ESTROBOSCOPICO (multiplos de T_m=2*pi/omega_m).

Pregunta: a partir de que Gamma2/kappa el modelo efectivo reproduce la
dinamica completa (no solo el estacionario) con F_min > 0.99?

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve, lindblad_dissipator,
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
gx_fijo = r * gz_baseline   # FIJO (no se re-escala con gz, igual que Tarea 7)

x = kap / 2.0
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m

Na = 2
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m
T_m = 2 * np.pi / om_m

alpha2_target = 2.0   # |alpha|^2 = eps/g_eff = 2, fijo

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)


def tlist_estrobo(tau_max, n_puntos=120):
    K_total = int(round(tau_max / T_m))
    stride = max(1, K_total // n_puntos)
    m = np.arange(0, K_total + 1, stride)
    return m * T_m


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-9
    return min_eig, ok


def correr_caso(gz_scale, Nb):
    gz = gz_scale * gz_baseline
    gx = gx_fijo
    g_eff = 2 * (gz * gx / om_m)   # g_eff = 2g (Tareas 7-8), con g=gz*gx/om_m
    eps = alpha2_target * g_eff    # eps/g_eff = 2 fijo
    Gamma2 = 4 * g_eff**2 / kap    # definicion dada en el enunciado

    tau_max = 10.0 / Gamma2
    tlist = tlist_estrobo(tau_max)

    G1m_gx = 2 * gx**2 * (x / (x**2 + D1m**2))
    G1p_gx = 2 * gx**2 * (x / (x**2 + D1p**2))

    # --- Modelo COMPLETO ---
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
         [eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
         [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)]]
    diss_full = [kap * lindblad_dissipator(sm),
                 (n_th + 1) * Gam_m * lindblad_dissipator(b),
                 n_th * Gam_m * lindblad_dissipator(bd)]
    rho0_full = tensor(thermal_dm(Na, 0), thermal_dm(Nb, 0))
    res_full = mesolve(H, rho0_full, tlist, diss_full, [bd * b], options=options)
    nb_full = res_full.expect[0]
    states_full_red = [ptrace(rho, 1) for rho in res_full.states]

    # --- Modelo B (g_eff=2g) ---
    a = destroy(Nb)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m
    G2p = 2 * g_eff**2 * ReS2p
    dk = g_eff**2 * (ImS2m + ImS2p)
    chi = -2j * eps * g_eff / kap
    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    diss_eff = [G1m_gx * lindblad_dissipator(a), G1p_gx * lindblad_dissipator(adag),
                G2m * lindblad_dissipator(a**2), G2p * lindblad_dissipator(adag**2)]
    rho0_eff = thermal_dm(Nb, 0)
    res_eff = mesolve(H_eff, rho0_eff, tlist, diss_eff, [adag * a], options=options)
    nb_eff = res_eff.expect[0]
    states_eff = res_eff.states

    fidelity_t = np.array([state_fidelity(states_full_red[i], states_eff[i])
                            for i in range(len(tlist))])
    F_min = fidelity_t.min()
    F_integrada = np.trapz(fidelity_t, tlist) / (tlist[-1] - tlist[0])

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

    t90_full = t90(tlist, nb_full)
    t90_eff = t90(tlist, nb_eff)

    idxs = [0, len(tlist)//2, len(tlist)-1]
    minigs = [validar_estado(states_full_red[i])[0] for i in idxs]
    oks = [validar_estado(states_full_red[i])[1] for i in idxs]

    return dict(gz_scale=gz_scale, g_eff=g_eff, eps=eps, Gamma2=Gamma2, tau_max=tau_max,
                n_puntos=len(tlist), F_min=F_min, F_integrada=F_integrada,
                t90_full=t90_full, t90_eff=t90_eff, minigs=minigs, oks=oks)


resultados = []
for gz_scale in [0.25, 0.5, 1.0]:
    # NOTA: se omite gz_scale=0.125 del enunciado original -- da Gamma2~0.0324,
    # tau_max=10/Gamma2~309 (kappa units), costo excesivo (~1h+ estimado) para
    # esta corrida por si sola dado el Hamiltoniano dependiente del tiempo con
    # oscilacion rapida a omega_m=1000. Omitido por decision explicita del
    # usuario para acotar tiempo; documentado en tarea11_resultados.md.
    Nb = 40 if gz_scale <= 0.25 else 25   # mas Fock si g chico (n_ss mayor: alpha^2=2 fijo,
                                          # pero Gamma2 chico => tau_max grande => margen extra)
    print(f"\n=== g_z x{gz_scale} (Nb={Nb}) ===")
    r_ = correr_caso(gz_scale, Nb)
    print(f"  g_eff={r_['g_eff']:.5f}  eps={r_['eps']:.5f}  Gamma2={r_['Gamma2']:.5f}  "
          f"tau_max={r_['tau_max']:.3f}  n_puntos={r_['n_puntos']}")
    print(f"  t90_full={r_['t90_full']:.4f}  t90_eff={r_['t90_eff']:.4f}  "
          f"razon={r_['t90_full']/r_['t90_eff'] if r_['t90_eff'] else float('nan'):.4f}")
    print(f"  F_min={r_['F_min']:.5f}  F_integrada={r_['F_integrada']:.5f}")
    print(f"  Validacion mineig en [0, mitad, final]: {r_['minigs']}  oks={r_['oks']}")
    resultados.append(r_)

print("\n=== TABLA RESUMEN (Tarea 11) ===")
hdr = f"{'gz_x':>6} {'g_eff':>8} {'Gamma2':>8} {'Gamma2/k':>9} {'t90f/t90e':>10} {'F_min':>8} {'F_int':>8}"
print(hdr)
for r_ in resultados:
    razon_t90 = r_['t90_full']/r_['t90_eff'] if r_['t90_eff'] else float('nan')
    print(f"{r_['gz_scale']:>6.3f} {r_['g_eff']:>8.5f} {r_['Gamma2']:>8.5f} {r_['Gamma2']/kap:>9.5f} "
          f"{razon_t90:>10.4f} {r_['F_min']:>8.5f} {r_['F_integrada']:>8.5f}")

# Buscar el umbral donde F_min > 0.99
cumple = [r_ for r_ in resultados if r_['F_min'] > 0.99]
if cumple:
    minimo = min(cumple, key=lambda r_: r_['Gamma2'])
    print(f"\nEl F_min>0.99 se cumple a partir de Gamma2/kappa >= {minimo['Gamma2']:.5f} "
          f"(g_z x{minimo['gz_scale']})")
else:
    print("\nNinguno de los casos escaneados alcanza F_min > 0.99")

np.savez("tarea11_resultados.npz",
         gz_scales=np.array([r_['gz_scale'] for r_ in resultados]),
         g_effs=np.array([r_['g_eff'] for r_ in resultados]),
         Gamma2s=np.array([r_['Gamma2'] for r_ in resultados]),
         F_mins=np.array([r_['F_min'] for r_ in resultados]),
         F_integradas=np.array([r_['F_integrada'] for r_ in resultados]),
         t90_fulls=np.array([r_['t90_full'] for r_ in resultados]),
         t90_effs=np.array([r_['t90_eff'] for r_ in resultados]))
print("\nGuardado: tarea11_resultados.npz")
