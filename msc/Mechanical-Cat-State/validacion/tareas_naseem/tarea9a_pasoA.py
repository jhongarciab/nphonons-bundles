# -*- coding: utf-8 -*-
"""
Tarea 9, Paso A (script liviano, separado del barrido completo por
limitaciones de memoria de la maquina): eps=0, muestreo estroboscopico,
verificar <n> ~ 1e-4 o menor. Una sola corrida con store_states=True para
obtener validacion Y expects en el mismo solve (evita duplicar computo).
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace, expect)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
Na, Nb_full = 2, 16   # reducido de 20 a 16 para bajar huella de memoria
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m
T_m = 2 * np.pi / om_m

tau_max = 30.0
K_total = int(round(tau_max / T_m))
stride = max(1, K_total // 100)   # ~100 puntos de salida
tlist = np.arange(0, K_total + 1, stride) * T_m
print(f"T_m={T_m:.6e}  K_total={K_total}  stride={stride}  n_puntos={len(tlist)}  "
      f"tau_max_real={tlist[-1]:.4f}")

b = tensor(qeye(Na), destroy(Nb_full))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb_full))
sz = tensor(sigmaz(), qeye(Nb_full))
sx = tensor(sigmax(), qeye(Nb_full))

H0 = 0.5 * omega_q_dim * sz
H = [
    H0,
    [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
    [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
    [gz_d * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
    [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
]  # eps=0: SIN terminos de drive
diss = [kap * lindblad_dissipator(sm),
        (n_th + 1) * Gam_m * lindblad_dissipator(b),
        n_th * Gam_m * lindblad_dissipator(bd)]
rho0 = tensor(thermal_dm(Na, 0), thermal_dm(Nb_full, 0))

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)
print("Corriendo (eps=0, estroboscopico)...")
res = mesolve(H, rho0, tlist, diss, [], options=options)
print("... listo.")

n_t = np.array([expect(bd * b, s) for s in res.states])
sz_t = np.array([expect(sz, s) for s in res.states])
pe_t = (1 + sz_t.real) / 2
dndt = np.gradient(n_t, tlist)

def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-9
    return tr, herm_err, min_eig, ok

idxs = [0, len(tlist)//4, len(tlist)//2, 3*len(tlist)//4, len(tlist)-1]
print("\nValidacion (umbral -1e-9):")
falla = False
for idx in idxs:
    rho_osc = ptrace(res.states[idx], 1)
    rho_qb = ptrace(res.states[idx], 0)
    tro, heo, meo, oko = validar_estado(rho_osc)
    trq, heq, meq, okq = validar_estado(rho_qb)
    ok = oko and okq
    falla = falla or not ok
    print(f"  t={tlist[idx]:8.4f}  osc: tr={tro:.3e} herm={heo:.3e} mineig={meo:.3e} | "
          f"qubit: tr={trq:.3e} herm={heq:.3e} mineig={meq:.3e}  {'OK' if ok else 'FALLA'}")
print("Resultado global:", "FALLA" if falla else "OK (umbral -1e-9)")

print(f"\n<n> estroboscopico: min={n_t.min():.3e} max={n_t.max():.3e} final={n_t[-1]:.3e}")
print(f"dn/dt final (estroboscopico) = {dndt[-1]:.3e}")
print(f"p_e_ss = {np.mean(pe_t[-8:]):.3e}")
print("Umbral esperado: <n> <= 1e-4")
print("Resultado:", "OK" if n_t.max() <= 1e-4 else f"NO cumple (max={n_t.max():.3e})")

np.savez("tarea9_pasoA.npz", tlist=tlist, n_t=n_t, dndt=dndt, pe_t=pe_t)
print("\nGuardado: tarea9_pasoA.npz")
