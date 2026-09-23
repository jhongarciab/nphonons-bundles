# -*- coding: utf-8 -*-
"""
Tarea 22: sesgo de ruido (bit-flip vs phase-flip).

Para Gamma2/kappa en {0.13, 0.52, 2.07} (g_z x {0.25,0.5,1}, g_x fijo) se
barre |alpha|^2 en {1,2,3,4,5} (Nb=max(20, 4|alpha|^2+12)). Para el
modelo COMPLETO (Floquet) y el EFECTIVO (Liouvilliano, g_eff=2g, delta_1,
Y AHORA SIEMPRE con el amortiguamiento intrinseco sqrt((n_th+1)gamma)a,
sqrt(n_th gamma)a^dagger -- correccion del bug de la Tarea 21) se reporta:

  - gamma_phase-flip: tasa del modo con mayor overlap con P
  - gamma_bit-flip: tasa (y frecuencia, parte imaginaria) del modo con
    mayor overlap con a
  - brecha de confinamiento: tasa del modo con mayor overlap con n,
    EXCLUYENDO los modos ya asignados a P o a
  - eta = gamma_phase-flip / gamma_bit-flip

Se ajusta ln(gamma_bit-flip) vs |alpha|^2 (se espera pendiente ~ -2 si
hay supresion exponencial).

Se repite Gamma2/kappa=0.13 (g_z x0.25) compensando la desintonia:
  completo: omega_q=omega_d=2*(omega_m+delta_1)
  efectivo: SIN el termino delta_1*a^dagger a, con Delta_2-=0 (default)

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, Options, propagator,
                    vector_to_operator, Qobj, liouvillian)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_baseline = gz_org / kappa_org
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_fijo = r * gz_baseline
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m
Na = 2

x = kap / 2.0
D2m0, D2p0 = 0.0, 4 * om_m
ReS2m0 = x / (x**2 + D2m0**2)
ReS2p0 = x / (x**2 + D2p0**2)
ImS2m0 = -D2m0 / (x**2 + D2m0**2)
ImS2p0 = -D2p0 / (x**2 + D2p0**2)
D1m, D1p = om_m, 3 * om_m
ImS1m = -D1m / (x**2 + D1m**2)
ImS1p = -D1p / (x**2 + D1p**2)
G1m_gx = 2 * gx_fijo**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_fijo**2 * (x / (x**2 + D1p**2))
delta_1 = gx_fijo**2 * (ImS1m + ImS1p)

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def clasificar_y_extraer(evals, Xks, Nb, es_completo):
    P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb)))
    n_op = destroy(Nb).dag() * destroy(Nb)
    a_op = destroy(Nb)
    if es_completo:
        P_ref, n_ref, a_ref = tensor(qeye(Na), P_osc), tensor(qeye(Na), n_op), tensor(qeye(Na), a_op)
    else:
        P_ref, n_ref, a_ref = P_osc, n_op, a_op

    modos = []
    for k, (mu, Xk) in enumerate(zip(evals, Xks)):
        if k == 0:
            continue  # excluir el trivial
        if es_completo:
            lam = -np.log(mu) / T_m
        else:
            lam = -mu
        normXk = Xk.norm()
        ov_P = abs((P_ref.dag() * Xk).tr()) / normXk
        ov_n = abs((n_ref.dag() * Xk).tr()) / normXk
        ov_a = abs((a_ref.dag() * Xk).tr()) / normXk
        modos.append(dict(k=k, lam=lam, ov_P=ov_P, ov_n=ov_n, ov_a=ov_a))

    # phase-flip: menor lambda entre los modos donde P domina sobre n y a
    cand_P = [m for m in modos if m['ov_P'] > m['ov_n'] and m['ov_P'] > m['ov_a']]
    gamma_pf = min((m['lam'].real for m in cand_P), default=float('nan'))
    k_pf = min(cand_P, key=lambda m: m['lam'].real)['k'] if cand_P else None

    # bit-flip: menor lambda entre los modos donde a domina sobre P y n
    cand_a = [m for m in modos if m['ov_a'] > m['ov_P'] and m['ov_a'] > m['ov_n']]
    if cand_a:
        modo_bf = min(cand_a, key=lambda m: m['lam'].real)
        gamma_bf = modo_bf['lam'].real
        freq_bf = modo_bf['lam'].imag
        k_bf = modo_bf['k']
    else:
        gamma_bf, freq_bf, k_bf = float('nan'), float('nan'), None

    # confinamiento: menor lambda entre los modos donde n domina, EXCLUYENDO k_pf y k_bf
    cand_n = [m for m in modos if m['ov_n'] > m['ov_P'] and m['ov_n'] > m['ov_a']
              and m['k'] not in (k_pf, k_bf)]
    gamma_conf = min((m['lam'].real for m in cand_n), default=float('nan'))

    return dict(gamma_pf=gamma_pf, gamma_bf=gamma_bf, freq_bf=freq_bf, gamma_conf=gamma_conf)


def correr_full(gz_scale, alpha2, Nb, compensar=False):
    gz = gz_scale * gz_baseline
    gx = gx_fijo
    g_dim = gz * gx / om_m
    g_eff = 2 * g_dim
    eps = alpha2 * g_eff
    Gamma2 = 4 * g_eff**2 / kap

    if compensar:
        oq = 2 * (om_m + delta_1)
    else:
        oq = omega_q_dim
    od = oq

    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb))
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))
    H0 = 0.5 * oq * sz
    H = [H0,
         [gx * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gx * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [gz * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [eps * sx, lambda t, _: np.exp(+1j * od * t)],
         [eps * sx, lambda t, _: np.exp(-1j * od * t)]]
    c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

    U = propagator(H, T_m, c_ops, options=opts)
    evals, evecs = U.eigenstates()
    orden = np.argsort(-np.abs(evals))
    n_modos = min(12, len(evals))
    evals_top = evals[orden][:n_modos]
    Xks = [vector_to_operator(evecs[orden[i]]) for i in range(n_modos)]

    metrica = clasificar_y_extraer(evals_top, Xks, Nb, es_completo=True)
    return dict(Gamma2=Gamma2, g_eff=g_eff, eps=eps, **metrica)


def correr_efectivo(gz_scale, alpha2, Nb, compensar=False):
    gz = gz_scale * gz_baseline
    gx = gx_fijo
    g_dim = gz * gx / om_m
    g_eff = 2 * g_dim
    eps = alpha2 * g_eff
    Gamma2 = 4 * g_eff**2 / kap

    a = destroy(Nb)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m0
    G2p = 2 * g_eff**2 * ReS2p0
    dk = g_eff**2 * (ImS2m0 + ImS2p0)
    chi = -2j * eps * g_eff / kap
    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    if not compensar:
        H_eff = H_eff + delta_1 * (adag * a)
    # CORRECCION: amortiguamiento intrinseco SIEMPRE incluido
    c_ops = [np.sqrt(G1m_gx) * a, np.sqrt(G1p_gx) * adag,
             np.sqrt(G2m) * (a * a), np.sqrt(G2p) * (adag * adag),
             np.sqrt((n_th + 1) * Gam_m) * a, np.sqrt(n_th * Gam_m) * adag]

    L = liouvillian(H_eff, c_ops)
    evals, evecs = L.eigenstates(sparse=False)
    orden = np.argsort(-evals.real)
    n_modos = min(12, len(evals))
    evals_top = evals[orden][:n_modos]
    Xks = [vector_to_operator(evecs[orden[i]]) for i in range(n_modos)]

    metrica = clasificar_y_extraer(evals_top, Xks, Nb, es_completo=False)
    return dict(Gamma2=Gamma2, **metrica)


def escanear(gz_scale, etiqueta, compensar=False):
    print(f"\n{'='*20} {etiqueta} (g_z x{gz_scale}, compensar={compensar}) {'='*20}")
    filas = []
    for alpha2 in [1, 2, 3, 4, 5]:
        Nb = max(20, 4 * alpha2 + 12)
        print(f"\n--- |alpha|^2={alpha2} (Nb={Nb}) ---")
        full = correr_full(gz_scale, alpha2, Nb, compensar=compensar)
        eff = correr_efectivo(gz_scale, alpha2, Nb, compensar=compensar)
        eta_full = full['gamma_pf'] / full['gamma_bf'] if full['gamma_bf'] else float('nan')
        eta_eff = eff['gamma_pf'] / eff['gamma_bf'] if eff['gamma_bf'] else float('nan')
        print(f"  Gamma2/kappa={full['Gamma2']:.5f}")
        print(f"  FULL: pf={full['gamma_pf']:.4e}  bf={full['gamma_bf']:.4e} "
              f"(freq={full['freq_bf']:.3e})  conf={full['gamma_conf']:.4e}  eta={eta_full:.4f}")
        print(f"  EFF : pf={eff['gamma_pf']:.4e}  bf={eff['gamma_bf']:.4e} "
              f"(freq={eff['freq_bf']:.3e})  conf={eff['gamma_conf']:.4e}  eta={eta_eff:.4f}")
        filas.append((alpha2, full, eff, eta_full, eta_eff))
    return filas


resultados_totales = {}
for gz_scale, nombre in [(0.25, "Gamma2/k=0.13"), (0.5, "Gamma2/k=0.52"), (1.0, "Gamma2/k=2.07")]:
    resultados_totales[nombre] = escanear(gz_scale, nombre, compensar=False)

print("\n\n" + "=" * 70)
print("TABLA RESUMEN GENERAL (sin compensar)")
print("=" * 70)
for nombre, filas in resultados_totales.items():
    print(f"\n--- {nombre} ---")
    print(f"{'a^2':>4} {'pf_full':>10} {'bf_full':>10} {'conf_full':>10} {'eta_full':>9} | "
          f"{'pf_eff':>10} {'bf_eff':>10} {'conf_eff':>10} {'eta_eff':>9}")
    for alpha2, full, eff, eta_full, eta_eff in filas:
        print(f"{alpha2:>4} {full['gamma_pf']:>10.3e} {full['gamma_bf']:>10.3e} "
              f"{full['gamma_conf']:>10.3e} {eta_full:>9.3f} | {eff['gamma_pf']:>10.3e} "
              f"{eff['gamma_bf']:>10.3e} {eff['gamma_conf']:>10.3e} {eta_eff:>9.3f}")

    # Ajuste ln(gamma_bf) vs alpha^2
    alphas2 = np.array([f[0] for f in filas])
    ln_bf_full = np.log(np.array([f[1]['gamma_bf'] for f in filas]))
    ln_bf_eff = np.log(np.array([f[2]['gamma_bf'] for f in filas]))
    A = np.vstack([alphas2, np.ones_like(alphas2)]).T
    slope_full, ic_full = np.linalg.lstsq(A, ln_bf_full, rcond=None)[0]
    slope_eff, ic_eff = np.linalg.lstsq(A, ln_bf_eff, rcond=None)[0]
    print(f"  Ajuste ln(gamma_bf) vs |alpha|^2:  completo: pendiente={slope_full:.4f}  "
          f"efectivo: pendiente={slope_eff:.4f}   (esperado ~ -2)")

# ------------------------------------------------------------
# Repetir Gamma2/k=0.13 CON compensacion
# ------------------------------------------------------------
resultados_comp = escanear(0.25, "Gamma2/k=0.13 COMPENSADO", compensar=True)

print("\n\n=== COMPARACION: gamma_bf con vs sin compensacion (g_z x0.25) ===")
sin_comp = resultados_totales["Gamma2/k=0.13"]
print(f"{'a^2':>4} {'bf_full_sin':>12} {'bf_full_con':>12} {'razon':>8} | "
      f"{'bf_eff_sin':>11} {'bf_eff_con':>11} {'razon':>8}")
for (a2, f_sin, e_sin, _, _), (_, f_con, e_con, _, _) in zip(sin_comp, resultados_comp):
    r_full = f_con['gamma_bf'] / f_sin['gamma_bf']
    r_eff = e_con['gamma_bf'] / e_sin['gamma_bf']
    print(f"{a2:>4} {f_sin['gamma_bf']:>12.4e} {f_con['gamma_bf']:>12.4e} {r_full:>8.3f} | "
          f"{e_sin['gamma_bf']:>11.4e} {e_con['gamma_bf']:>11.4e} {r_eff:>8.3f}")

np.savez("tarea22_resultados.npz",
         **{f"{nombre}_full_bf": np.array([f[1]['gamma_bf'] for f in filas])
            for nombre, filas in resultados_totales.items()},
         **{f"{nombre}_eff_bf": np.array([f[2]['gamma_bf'] for f in filas])
            for nombre, filas in resultados_totales.items()})
print("\nGuardado: tarea22_resultados.npz")
