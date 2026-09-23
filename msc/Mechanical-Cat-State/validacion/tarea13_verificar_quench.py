# -*- coding: utf-8 -*-
"""
Tarea 13: repite el Paso A de la Tarea 9 (eps=0, tau_max=30, muestreo
estroboscopico) con el qubit iniciado en:
  (i)  BASE fisico:    basis(Na,1)  (sz=-1)
  (ii) EXCITADO fisico: basis(Na,0)  (sz=+1)  -- lo que TODAS las corridas
       previas salvo la Tarea 10 usaron por error (thermal_dm(Na,0)).

Hipotesis a verificar:
  (i)  qubit en base:     <n> estroboscopico final < 1e-4
  (ii) qubit excitado:    piso final ~ (2*g_z/omega_m)^2 = 0.0144
                          + contribucion del "par" (proceso de dos fonones
                          real, a diferencia del piso (i) que es puro
                          artefacto del quench inicial del qubit excitado).

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
Na, Nb_full = 2, 16
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m

piso_pred = (2 * gz_d / om_m) ** 2
print(f"Prediccion piso (qubit excitado) = (2*g_z/omega_m)^2 = {piso_pred:.6f}")

tau_max = 30.0
K_total = int(round(tau_max / T_m))
stride = max(1, K_total // 100)
tlist = np.arange(0, K_total + 1, stride) * T_m
print(f"T_m={T_m:.6e}  n_puntos={len(tlist)}  tau_max_real={tlist[-1]:.4f}")

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
     [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)]]   # eps=0
diss = [kap * lindblad_dissipator(sm),
        (n_th + 1) * Gam_m * lindblad_dissipator(b),
        n_th * Gam_m * lindblad_dissipator(bd)]

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return tr, herm_err, min_eig, ok


def correr(qubit_inicial, etiqueta):
    rho0 = tensor(qubit_inicial * qubit_inicial.dag(), thermal_dm(Nb_full, 0))
    print(f"\nCorriendo eps=0, qubit inicial = {etiqueta}...")
    res = mesolve(H, rho0, tlist, diss, [], options=options)
    print("... listo.")
    n_t = np.array([expect(bd * b, s) for s in res.states])
    sz_t = np.array([expect(sz, s) for s in res.states])
    pe_t = (1 + sz_t.real) / 2
    dndt = np.gradient(n_t, tlist)

    idxs = [0, len(tlist)//4, len(tlist)//2, 3*len(tlist)//4, len(tlist)-1]
    print("  Validacion (umbral -1e-8):")
    falla = False
    for idx in idxs:
        rho_osc = ptrace(res.states[idx], 1)
        rho_qb = ptrace(res.states[idx], 0)
        tro, heo, meo, oko = validar_estado(rho_osc)
        trq, heq, meq, okq = validar_estado(rho_qb)
        ok = oko and okq
        falla = falla or not ok
        print(f"    t={tlist[idx]:8.4f}  osc: mineig={meo:.3e}  qubit: mineig={meq:.3e}  "
              f"{'OK' if ok else 'FALLA'}")
    print("  Resultado global:", "FALLA" if falla else "OK (umbral -1e-8)")

    print(f"  <n> final = {n_t[-1]:.6e}   dn/dt final = {dndt[-1]:.3e}   p_e_ss = {np.mean(pe_t[-8:]):.3e}")
    return n_t, dndt, pe_t


n_base, dndt_base, pe_base = correr(basis(Na, 1), "BASE fisico")
n_exc, dndt_exc, pe_exc = correr(basis(Na, 0), "EXCITADO fisico")

print("\n=== TABLA RESUMEN (Tarea 13) ===")
print(f"{'':>20} {'<n> final':>12} {'dn/dt final':>14} {'p_e_ss':>10}")
print(f"{'BASE':>20} {n_base[-1]:>12.3e} {dndt_base[-1]:>14.3e} {pe_base[-8:].mean():>10.3e}")
print(f"{'EXCITADO':>20} {n_exc[-1]:>12.3e} {dndt_exc[-1]:>14.3e} {pe_exc[-8:].mean():>10.3e}")
print(f"\nPrediccion piso (excitado) = (2 g_z/omega_m)^2 = {piso_pred:.6f}")
print(f"<n>final(excitado)/prediccion = {n_exc[-1]/piso_pred:.4f}")
print(f"\nHipotesis BASE (<n><1e-4):", "CONFIRMADA" if n_base[-1] < 1e-4 else
      f"NO CONFIRMADA (n_final={n_base[-1]:.3e})")

np.savez("tarea13_resultados.npz", tlist=tlist, n_base=n_base, dndt_base=dndt_base,
         pe_base=pe_base, n_exc=n_exc, dndt_exc=dndt_exc, pe_exc=pe_exc, piso_pred=piso_pred)
print("\nGuardado: tarea13_resultados.npz")
