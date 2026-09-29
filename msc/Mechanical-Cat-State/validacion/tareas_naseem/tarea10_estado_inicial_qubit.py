# -*- coding: utf-8 -*-
"""
Tarea 10: repite fig2 (eps=1.44=Omega_full, drive real del codigo original)
iniciando el qubit en:
  (i)  estado BASE fisico:      basis(Na,1)  (sz=-1, energia -omega_q/2)
  (ii) estado EXCITADO fisico:  basis(Na,0)  (sz=+1, energia +omega_q/2)

NOTA: fig2_git_v1.py usa `thermal_dm(Na,0)` = basis(Na,0) como estado
inicial, etiquetado "ground" en los comentarios del codigo, pero verificado
en la Tarea 7 que basis(Na,0) es FISICAMENTE el estado EXCITADO (sz=+1).
Es decir, fig2 (y por herencia Tareas 1-2 de la ronda anterior) en
realidad parte con el qubit en el estado EXCITADO, no en el base.

Se compara F(t) y <n>(t) contra el modelo EFECTIVO B (g_eff=2g,
eps=Omega_full=1.44, la normalizacion correcta segun Tareas 7-8) para
ambas condiciones iniciales del qubit, con muestreo estroboscopico
(tlist en multiplos de T_m=2*pi/omega_m) para evitar el aliasing
identificado en la Tarea 9.

Pregunta: la caida de fidelidad en kappa*t~1 (vista en la Tarea 1,
F_min=0.73 con el qubit iniciado en lo que se penso era "ground" pero es
"excited") es un artefacto de partir con el qubit en el estado
EQUIVOCADO? Si desaparece iniciando en el estado base fisico, la respuesta
es si.

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, basis, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace)
from qutip.metrics import fidelity as state_fidelity

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
g_eff_2g = 2 * g_dim
eps = 4 * g_dim   # = Omega_full del codigo original (1.44)

x = kap / 2.0
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m
G1m_gx = 2 * gx_dim**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_dim**2 * (x / (x**2 + D1p**2))

Na, Nb_full = 2, 30
N_eff = 30   # debe coincidir con Nb_full para poder comparar fidelidad
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m
T_m = 2 * np.pi / om_m

print(f"eps = Omega_full = 4g = {eps:.6f},  g_eff=2g = {g_eff_2g:.6f}")

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)


def tlist_estrobo(tau_max, n_puntos=100):
    K_total = int(round(tau_max / T_m))
    stride = max(1, K_total // n_puntos)
    m = np.arange(0, K_total + 1, stride)
    return m * T_m


tau_max = 39.0
tlist = tlist_estrobo(tau_max)
print(f"n_puntos estroboscopicos = {len(tlist)}, tau_max_real={tlist[-1]:.3f}")

b = tensor(qeye(Na), destroy(Nb_full))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb_full))
sz = tensor(sigmaz(), qeye(Nb_full))
sx = tensor(sigmax(), qeye(Nb_full))
H0 = 0.5 * omega_q_dim * sz
H = [H0,
     [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [gz_d * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
     [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)]]
diss_full = [kap * lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th * Gam_m * lindblad_dissipator(bd)]

# ------------------------------------------------------------
# Modelo efectivo B (g_eff=2g), referencia comun
# ------------------------------------------------------------
a = destroy(N_eff)
adag = a.dag()
G2m = 2 * g_eff_2g**2 * ReS2m
G2p = 2 * g_eff_2g**2 * ReS2p
dk = g_eff_2g**2 * (ImS2m + ImS2p)
chi = -2j * eps * g_eff_2g / kap
H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
diss_eff = [G1m_gx * lindblad_dissipator(a), G1p_gx * lindblad_dissipator(adag),
            G2m * lindblad_dissipator(a**2), G2p * lindblad_dissipator(adag**2)]
rho0_eff = thermal_dm(N_eff, 0)
print("\nCorriendo modelo EFECTIVO B (g_eff=2g)...")
res_eff = mesolve(H_eff, rho0_eff, tlist, diss_eff, [adag * a], options=Options(
    nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True))
states_eff = res_eff.states
nb_eff = res_eff.expect[0]
print("... listo.")


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-9
    return tr, herm_err, min_eig, ok


def correr_full(qubit_inicial, etiqueta):
    rho0 = tensor(qubit_inicial * qubit_inicial.dag(), thermal_dm(Nb_full, 0))
    print(f"\nCorriendo modelo COMPLETO, qubit inicial = {etiqueta}...")
    res = mesolve(H, rho0, tlist, diss_full, [bd * b], options=options)
    print("... listo.")
    nb_full = res.expect[0]
    states_red = [ptrace(rho, 1) for rho in res.states]
    fidelity_t = np.array([state_fidelity(states_red[i], states_eff[i]) for i in range(len(tlist))])

    idxs = [0, len(tlist)//4, len(tlist)//2, 3*len(tlist)//4, len(tlist)-1]
    print(f"  Validacion (umbral -1e-9):")
    falla = False
    for idx in idxs:
        tro, heo, meo, oko = validar_estado(states_red[idx])
        rho_qb = ptrace(res.states[idx], 0)
        trq, heq, meq, okq = validar_estado(rho_qb)
        ok = oko and okq
        falla = falla or not ok
        print(f"    t={tlist[idx]:7.3f}  osc: mineig={meo:.3e}  qubit: mineig={meq:.3e}  "
              f"{'OK' if ok else 'FALLA'}")
    print("  Resultado global:", "FALLA" if falla else "OK (umbral -1e-9)")

    return nb_full, fidelity_t


qubit_base = basis(Na, 1)       # fisicamente el estado base (sz=-1)
qubit_excitado = basis(Na, 0)   # fisicamente el estado excitado (sz=+1) -- lo que usa fig2 original

nb_full_base, F_base = correr_full(qubit_base, "BASE fisico (basis(Na,1))")
nb_full_exc, F_exc = correr_full(qubit_excitado, "EXCITADO fisico (basis(Na,0), = fig2 original)")

print("\n=== TABLA COMPARATIVA ===")
print(f"{'kt':>7} {'n_eff':>8} | {'n_full(base)':>13} {'F(base)':>9} | "
      f"{'n_full(exc)':>12} {'F(exc)':>9}")
for i in range(0, len(tlist), max(1, len(tlist)//15)):
    print(f"{tlist[i]:>7.2f} {nb_eff[i]:>8.4f} | {nb_full_base[i]:>13.4f} {F_base[i]:>9.4f} | "
          f"{nb_full_exc[i]:>12.4f} {F_exc[i]:>9.4f}")

print(f"\nF_min (base)     = {F_base.min():.4f}  en kt={tlist[np.argmin(F_base)]:.3f}")
print(f"F_min (excitado) = {F_exc.min():.4f}  en kt={tlist[np.argmin(F_exc)]:.3f}")
print(f"F_final (base)     = {F_base[-1]:.4f}")
print(f"F_final (excitado) = {F_exc[-1]:.4f}")

respuesta = "SI desaparece" if F_base.min() > 0.95 else "NO desaparece (sigue habiendo dip)"
print(f"\nPregunta: la caida de F en kt~1 desaparece con qubit en estado base? -> {respuesta}")

np.savez("tarea10_resultados.npz", tlist=tlist, nb_eff=nb_eff,
         nb_full_base=nb_full_base, F_base=F_base,
         nb_full_exc=nb_full_exc, F_exc=F_exc)
print("\nGuardado: tarea10_resultados.npz")
