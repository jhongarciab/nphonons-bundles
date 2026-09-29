# -*- coding: utf-8 -*-
"""
Tarea 9: muestreo estroboscopico para eliminar el aliasing del
desplazamiento polaronico beta(t) = (g_z/omega_m) sz (e^{i omega_m t} - 1)
(en el marco de interaccion mecanico), que en la Tarea 6b, muestreado en
tiempos NO conmensurados con T_m=2*pi/omega_m, producia un "piso" espurio
en <n> (~0.02) y una deriva dn/dt~-0.01 que NO son fisicos.

tlist se define SOLO en multiplos enteros de T_m (donde beta(t)=0 exacto):
    tlist = m * T_m,  m = 0, 1*stride, 2*stride, ...

Paso A: con eps=0 (sin drive, con disipacion), verificar que <n>
estroboscopico se mantiene ~1e-4 o menor (confirma que el piso de 6b era
aliasing, no fisica real).

Paso B: repetir el barrido de la Tarea 6b (eps/kappa en
{0.02,0.05,0.1,0.2,0.4,0.72,1.44}), tau_max=60, muestreo estroboscopico,
comparando n_ss y t90 del completo contra el modelo B (g_eff=2g).

Validacion (traza, hermiticidad, positividad, umbral -1e-9) en instantes
seleccionados. NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace)

# ------------------------------------------------------------
# Parametros fisicos (identicos a rondas anteriores)
# ------------------------------------------------------------
r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m           # g del codigo
g_eff_2g = 2 * g_dim                    # g_eff correcto (Tareas 7-8)

x = kap / 2.0
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m
G1m_gx = 2 * gx_dim**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_dim**2 * (x / (x**2 + D1p**2))

Na, Nb_full = 2, 20
N_eff = 40
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m

T_m = 2 * np.pi / om_m
print(f"T_m = 2*pi/omega_m = {T_m:.6e}  (kappa units)")

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8)


def tlist_estrobo(tau_max, n_puntos_deseados=150):
    K_total = int(round(tau_max / T_m))
    stride = max(1, K_total // n_puntos_deseados)
    m = np.arange(0, K_total + 1, stride)
    return m * T_m


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-9
    return tr, herm_err, min_eig, ok


def correr_full(eps, tau_max, validar=False):
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
        [eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
        [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)],
    ]
    diss = [kap * lindblad_dissipator(sm),
            (n_th + 1) * Gam_m * lindblad_dissipator(b),
            n_th * Gam_m * lindblad_dissipator(bd)]
    rho0 = tensor(thermal_dm(Na, 0), thermal_dm(Nb_full, 0))

    tlist = tlist_estrobo(tau_max)
    res = mesolve(H, rho0, tlist, diss, [sz, bd * b], options=options)
    sz_t, n_t = res.expect[0], res.expect[1]
    pe_t = (1 + sz_t.real) / 2
    dndt = np.gradient(n_t, tlist)

    resultado = dict(tlist=tlist, n_t=n_t, pe_t=pe_t, dndt=dndt)

    if validar:
        opt_val = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)
        res2 = mesolve(H, rho0, tlist, diss, [], options=opt_val)
        idxs = [0, len(tlist)//4, len(tlist)//2, 3*len(tlist)//4, len(tlist)-1]
        print("  Validacion (estados reducidos, umbral -1e-9):")
        falla = False
        for idx in idxs:
            rho_full = res2.states[idx]
            rho_osc = ptrace(rho_full, 1)
            rho_qb = ptrace(rho_full, 0)
            tro, heo, meo, oko = validar_estado(rho_osc)
            trq, heq, meq, okq = validar_estado(rho_qb)
            ok = oko and okq
            falla = falla or not ok
            print(f"    t={tlist[idx]:8.4f}  osc: tr={tro:.3e} herm={heo:.3e} mineig={meo:.3e} | "
                  f"qubit: tr={trq:.3e} herm={heq:.3e} mineig={meq:.3e}  {'OK' if ok else 'FALLA'}")
        print("  Resultado global:", "FALLA" if falla else "OK (umbral -1e-9)")

    return resultado


def correr_eff(eps, g_eff, tau_max):
    a = destroy(N_eff)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m
    G2p = 2 * g_eff**2 * ReS2p
    dk = g_eff**2 * (ImS2m + ImS2p)
    chi = -2j * eps * g_eff / kap

    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    diss = [G1m_gx * lindblad_dissipator(a),
            G1p_gx * lindblad_dissipator(adag),
            G2m * lindblad_dissipator(a**2),
            G2p * lindblad_dissipator(adag**2)]
    rho0 = thermal_dm(N_eff, 0)

    tlist = tlist_estrobo(tau_max)
    res = mesolve(H_eff, rho0, tlist, diss, [adag * a], options=options)
    n_t = res.expect[0]
    return tlist, n_t


def tiempo_subida_90(t, n):
    n_ss = np.mean(n[-8:])
    if n_ss < 1e-6:
        return 0.0, n_ss
    umbral = 0.9 * n_ss
    idx = np.argmax(n >= umbral)
    if n[idx] < umbral:
        return np.nan, n_ss
    if idx == 0:
        return t[0], n_ss
    t0, t1, n0, n1 = t[idx - 1], t[idx], n[idx - 1], n[idx]
    return t0 + (umbral - n0) * (t1 - t0) / (n1 - n0), n_ss


# ------------------------------------------------------------
# PASO A: eps=0, verificar <n> estroboscopico ~ 1e-4 o menor
# ------------------------------------------------------------
print("\n=== PASO A: eps=0 (sin drive), verificacion del piso ===")
res0 = correr_full(0.0, tau_max=30.0, validar=True)
print(f"\n<n> estroboscopico: min={res0['n_t'].min():.3e}  max={res0['n_t'].max():.3e}  "
      f"final={res0['n_t'][-1]:.3e}")
print(f"dn/dt final (estroboscopico) = {res0['dndt'][-1]:.3e}")
print("Umbral esperado por el enunciado: <n> <= 1e-4")
print("Resultado:", "OK" if res0['n_t'].max() <= 1e-4 else
      f"NO cumple umbral (max={res0['n_t'].max():.3e})")

np.savez("tarea9_pasoA.npz", tlist=res0["tlist"], n_t=res0["n_t"], dndt=res0["dndt"])

# ------------------------------------------------------------
# PASO B: barrido en eps, comparando con modelo B (g_eff=2g)
# ------------------------------------------------------------
print("\n\n=== PASO B: barrido eps/kappa, estroboscopico, tau_max=60 ===")
eps_list = [0.02, 0.05, 0.1, 0.2, 0.4, 0.72, 1.44]
resultados = []
for eps in eps_list:
    print(f"\n--- eps/kappa = {eps} ---")
    full = correr_full(eps, tau_max=60.0, validar=False)
    tlist_e, n_eff_t = correr_eff(eps, g_eff_2g, tau_max=60.0)

    n_ss_full = np.mean(full["n_t"][-8:])
    n_ss_effB = np.mean(n_eff_t[-8:])
    t90_full, _ = tiempo_subida_90(full["tlist"], full["n_t"])
    t90_effB, _ = tiempo_subida_90(tlist_e, n_eff_t)
    dndt_final = full["dndt"][-1]
    pe_ss = np.mean(full["pe_t"][-8:])

    print(f"  FULL: n_ss={n_ss_full:.5f}  t90={t90_full:.3f}  dn/dt_final={dndt_final:.3e}  "
          f"p_e_ss={pe_ss:.5f}")
    print(f"  B(2g): n_ss={n_ss_effB:.5f}  t90={t90_effB:.3f}")
    print(f"  razon n_ss full/B = {n_ss_full/n_ss_effB if n_ss_effB>1e-9 else float('nan'):.4f}")

    resultados.append((eps, n_ss_full, t90_full, dndt_final, pe_ss, n_ss_effB, t90_effB))

print("\n=== TABLA RESUMEN (Tarea 9, Paso B) ===")
hdr = f"{'eps/k':>7} {'n_full':>9} {'t90_full':>9} {'dndt_f':>10} {'pe_ss':>8} | {'n_B':>9} {'t90_B':>7} | {'full/B':>8}"
print(hdr)
for row in resultados:
    eps, nf, t90f, dndtf, pef, nB, t90B = row
    razon = nf / nB if nB > 1e-9 else float('nan')
    print(f"{eps:>7.2f} {nf:>9.5f} {t90f:>9.3f} {dndtf:>10.2e} {pef:>8.5f} | "
          f"{nB:>9.5f} {t90B:>7.3f} | {razon:>8.4f}")

np.savez("tarea9_pasoB.npz", eps_list=np.array(eps_list), resultados=np.array(resultados))
print("\nGuardado: tarea9_pasoA.npz, tarea9_pasoB.npz")
