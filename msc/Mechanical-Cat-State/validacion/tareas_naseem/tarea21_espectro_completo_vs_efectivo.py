# -*- coding: utf-8 -*-
"""
Tarea 21: espectro completo (Floquet) frente al modelo efectivo
(g_eff=2g + delta_1, Liouvilliano independiente del tiempo).

Para el modelo efectivo se usa qutip.liouvillian(H_eff, c_ops) y se
diagonaliza con .eigenstates() (equivalente a "eigenenergies sparse=False"
pero incluyendo tambien los autovectores, necesarios para clasificar los
modos igual que en la Tarea 20).

Barrido: |alpha|^2=eps/g_eff=2 fijo, g_z escalado por {0.125,0.25,0.5,1}
(g_x FIJO, eps escalado). Se reporta: Gamma2/kappa, brecha de
confinamiento (completo y efectivo), gamma_phase-flip (completo,
efectivo, y formula 2|alpha|^2(Gamma_- + Gamma_+)), tasa del modo de
coherencia logica.

Bonus (si el costo lo permite): g_z x0.25 con |alpha|^2 en {1,2,3,4}.

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, Options, propagator,
                    vector_to_operator, Qobj, liouvillian, lindblad_dissipator)

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
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m
ImS1m = -D1m / (x**2 + D1m**2)
ImS1p = -D1p / (x**2 + D1p**2)
G1m_gx = 2 * gx_fijo**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_fijo**2 * (x / (x**2 + D1p**2))
delta_1 = gx_fijo**2 * (ImS1m + ImS1p)

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def clasificar_modos(evals, Xks, Nb, es_completo):
    """evals ya ordenados por |mu| desc (completo) o Re desc (efectivo, cercano a 0).
    Devuelve lista de dicts con k, lambda_real, clase, overlaps."""
    P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb)))
    n_op = destroy(Nb).dag() * destroy(Nb)
    a_op = destroy(Nb)
    a2_op = destroy(Nb) * destroy(Nb)
    if es_completo:
        P_ref = tensor(qeye(Na), P_osc)
        n_ref = tensor(qeye(Na), n_op)
        a_ref = tensor(qeye(Na), a_op)
        a2_ref = tensor(qeye(Na), a2_op)
    else:
        P_ref, n_ref, a_ref, a2_ref = P_osc, n_op, a_op, a2_op

    operadores = {"P": P_ref, "n": n_ref, "a": a_ref, "a2": a2_ref}

    resultados = []
    for k, (mu, Xk) in enumerate(zip(evals, Xks)):
        if es_completo:
            lam = 0.0 + 0.0j if abs(mu) > 1 - 1e-13 else -np.log(mu) / T_m
        else:
            lam = -mu  # generador continuo: autovalor = -tasa (mu ya complejo)

        normXk = Xk.norm()
        overlaps = {nom: abs((op.dag() * Xk).tr()) / normXk for nom, op in operadores.items()}
        clase = "trivial" if (k == 0) else max(overlaps, key=overlaps.get)
        resultados.append(dict(k=k, mu=mu, lam=lam, overlaps=overlaps, clase=clase))
    return resultados


def extraer_metricas(resultados):
    """De la lista clasificada, extrae: brecha de confinamiento (menor
    lambda entre los clasificados como 'n'), gamma_phase-flip (menor
    lambda entre 'P'), tasa de coherencia logica (menor lambda entre 'a'),
    excluyendo el modo trivial k=0."""
    def menor_lambda(clase_buscada):
        candidatos = [r_['lam'].real for r_ in resultados[1:] if r_['clase'] == clase_buscada]
        return min(candidatos) if candidatos else float('nan')

    return dict(confinamiento=menor_lambda('n'), phase_flip=menor_lambda('P'),
                coherencia_logica=menor_lambda('a'))


def correr_full(gz_scale, alpha2, Nb):
    gz = gz_scale * gz_baseline
    gx = gx_fijo
    g_dim = gz * gx / om_m
    g_eff = 2 * g_dim
    eps = alpha2 * g_eff
    Gamma2 = 4 * g_eff**2 / kap

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
    c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

    U = propagator(H, T_m, c_ops, options=opts)
    evals, evecs = U.eigenstates()
    orden = np.argsort(-np.abs(evals))
    evals = evals[orden][:10]
    Xks = [vector_to_operator(evecs[orden[i]]) for i in range(10)]

    resultados = clasificar_modos(evals, Xks, Nb, es_completo=True)
    metricas = extraer_metricas(resultados)
    return dict(Gamma2=Gamma2, g_eff=g_eff, eps=eps, **metricas)


def correr_efectivo(gz_scale, alpha2, Nb):
    gz = gz_scale * gz_baseline
    gx = gx_fijo
    g_dim = gz * gx / om_m
    g_eff = 2 * g_dim
    eps = alpha2 * g_eff
    Gamma2 = 4 * g_eff**2 / kap

    a = destroy(Nb)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m
    G2p = 2 * g_eff**2 * ReS2p
    dk = g_eff**2 * (ImS2m + ImS2p)
    chi = -2j * eps * g_eff / kap
    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2 + delta_1 * (adag * a)
    c_ops = [np.sqrt(G1m_gx) * a, np.sqrt(G1p_gx) * adag,
             np.sqrt(G2m) * (a * a), np.sqrt(G2p) * (adag * adag)]

    L = liouvillian(H_eff, c_ops)
    evals, evecs = L.eigenstates(sparse=False)
    orden = np.argsort(-evals.real)   # menos negativo primero (mas lento)
    evals = evals[orden][:10]
    Xks = [vector_to_operator(evecs[orden[i]]) for i in range(10)]

    resultados = clasificar_modos(evals, Xks, Nb, es_completo=False)
    metricas = extraer_metricas(resultados)

    formula_phase_flip = 2 * alpha2 * (G1m_gx + G1p_gx)
    return dict(Gamma2=Gamma2, **metricas, formula_phase_flip=formula_phase_flip)


print(f"delta_1 = {delta_1:.5e},  G1m_gx={G1m_gx:.5e},  G1p_gx={G1p_gx:.5e}")

print("\n=== Barrido principal: |alpha|^2=2, g_z x{0.125,0.25,0.5,1} ===")
resultados_principal = []
for gz_scale in [0.125, 0.25, 0.5, 1.0]:
    Nb = 20
    print(f"\n--- g_z x{gz_scale} ---")
    full = correr_full(gz_scale, 2.0, Nb)
    eff = correr_efectivo(gz_scale, 2.0, Nb)
    print(f"  Gamma2/kappa = {full['Gamma2']:.5f}")
    print(f"  Confinamiento: completo={full['confinamiento']:.5e}  efectivo={eff['confinamiento']:.5e}")
    print(f"  Phase-flip:    completo={full['phase_flip']:.5e}  efectivo={eff['phase_flip']:.5e}  "
          f"formula={eff['formula_phase_flip']:.5e}")
    print(f"  Coherencia logica: completo={full['coherencia_logica']:.5e}  "
          f"efectivo={eff['coherencia_logica']:.5e}")
    resultados_principal.append((gz_scale, full, eff))

print("\n=== TABLA RESUMEN (Tarea 21, barrido principal) ===")
hdr = (f"{'gz_x':>6} {'Gamma2/k':>9} | {'confin_full':>11} {'confin_eff':>10} | "
       f"{'pflip_full':>10} {'pflip_eff':>10} {'pflip_form':>10} | "
       f"{'coh_full':>9} {'coh_eff':>9}")
print(hdr)
for gz_scale, full, eff in resultados_principal:
    print(f"{gz_scale:>6.3f} {full['Gamma2']:>9.5f} | {full['confinamiento']:>11.4e} "
          f"{eff['confinamiento']:>10.4e} | {full['phase_flip']:>10.4e} "
          f"{eff['phase_flip']:>10.4e} {eff['formula_phase_flip']:>10.4e} | "
          f"{full['coherencia_logica']:>9.4e} {eff['coherencia_logica']:>9.4e}")

np.savez("tarea21_principal.npz",
         datos=np.array([(gz, f['Gamma2'], f['confinamiento'], e['confinamiento'],
                           f['phase_flip'], e['phase_flip'], e['formula_phase_flip'],
                           f['coherencia_logica'], e['coherencia_logica'])
                          for gz, f, e in resultados_principal]))
print("\nGuardado: tarea21_principal.npz")

# ------------------------------------------------------------
# Bonus: g_z x0.25, |alpha|^2 en {1,2,3,4}
# ------------------------------------------------------------
print("\n\n=== BONUS: g_z x0.25, |alpha|^2 en {1,2,3,4} ===")
resultados_bonus = []
for alpha2 in [1.0, 2.0, 3.0, 4.0]:
    Nb = max(20, int(4 * alpha2 + 10))
    print(f"\n--- |alpha|^2={alpha2} (Nb={Nb}) ---")
    full = correr_full(0.25, alpha2, Nb)
    eff = correr_efectivo(0.25, alpha2, Nb)
    print(f"  Gamma2/kappa = {full['Gamma2']:.5f}")
    print(f"  Confinamiento: completo={full['confinamiento']:.5e}  efectivo={eff['confinamiento']:.5e}")
    print(f"  Phase-flip:    completo={full['phase_flip']:.5e}  efectivo={eff['phase_flip']:.5e}  "
          f"formula={eff['formula_phase_flip']:.5e}")
    resultados_bonus.append((alpha2, full, eff))

print("\n=== TABLA RESUMEN (Bonus) ===")
print(f"{'alpha^2':>7} {'confin_full':>11} {'confin_eff':>10} | {'pflip_full':>10} "
      f"{'pflip_eff':>10} {'pflip_form':>10}")
for alpha2, full, eff in resultados_bonus:
    print(f"{alpha2:>7.1f} {full['confinamiento']:>11.4e} {eff['confinamiento']:>10.4e} | "
          f"{full['phase_flip']:>10.4e} {eff['phase_flip']:>10.4e} {eff['formula_phase_flip']:>10.4e}")

np.savez("tarea21_bonus.npz",
         datos=np.array([(a2, f['Gamma2'], f['confinamiento'], e['confinamiento'],
                           f['phase_flip'], e['phase_flip'], e['formula_phase_flip'])
                          for a2, f, e in resultados_bonus]))
print("\nGuardado: tarea21_bonus.npz")
