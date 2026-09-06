#!/usr/bin/env python3
"""
04_escritura_trayectoria.py — Escritura condicionada vía trayectorias cuánticas
================================================================================

Protocolo corregido respecto a 03_escritura_polaron.py.

El problema con partir de rho_ss es que el estado de 2 fonones tiene
ocupación ~1.5% en el promedio — no hay "material" sobre el que actuar.

Solución: usar mcsolve para detectar en cada trayectoria el instante t*
en que el sistema ha acumulado 2 fonones (⟨n̂⟩ ≈ 2 en esa trayectoria),
extraer el estado ψ(t*) y aplicar el pulso diferencial desde allí.

Protocolo en tres etapas (todo en una trayectoria):
  1. Generación: mcsolve con H_prod hasta t* (detectado cuando n_bar ~ 2)
  2. Escritura:  mesolve con H_w desde ρ(t*) durante t_π
  3. Verificación: F_w = ⟨Ψ_-,2̃|ρ(t*+t_π)|Ψ_-,2̃⟩

Parámetros: λ=0.22 (producción de pureza), Δ₂ = λ²/ω_b - 2ω_b - J
Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
from pathlib import Path
import qutip as qt

# =============================================================================
# PARÁMETROS
# =============================================================================
RERUN = False

omega_b   = 1.0
gamma     = 0.0002
gamma_phi = 0.0004
J         = 0.5
kappa     = 0.001
lam       = 0.22          # parámetros de producción de pureza
alpha     = lam / omega_b  # = 0.22

# Detuning resonancia Stokes n=2 (ec. res_II)
Delta_2   = lam**2 / omega_b - 2.0 * omega_b - J   # = -2.4516

# Bombeo de producción
Omega_prod = 0.2

# Pulso de escritura diferencial: Ω_w cumple 1/J << t_π << 1/κ
Omega_w   = 0.05
t_pi      = np.pi / (2.0 * np.sqrt(2.0) * Omega_w)   # ≈ 22.2/ω_b

# Trayectorias
ntraj     = 50
seed_base = 324680751

# Tiempo de generación: suficiente para ver varios eventos de 2 fonones
t_gen_max = 15000.0 / omega_b
Nt_gen    = 15001
tlist_gen = np.linspace(0.0, t_gen_max, Nt_gen)

# Tiempo del pulso de escritura
n_steps_w = 300
tlist_w   = np.linspace(0.0, 2.0 * t_pi, n_steps_w)

Ncut = 16

print(f"Δ₂/ω_b  = {Delta_2:.6f}")
print(f"t_π/ω_b = {t_pi:.4f}")
print(f"Condición: 1/J={1/J:.1f} << t_π={t_pi:.1f} << 1/κ={1/kappa:.1f}  ✓")
print(f"ntraj = {ntraj},  t_gen_max = {t_gen_max:.0f}/ω_b")

# =============================================================================
# OPERADORES
# =============================================================================
b   = qt.destroy(Ncut)
nb  = b.dag() * b
Ib  = qt.qeye(Ncut)
sm  = qt.sigmam()
Iq  = qt.qeye(2)

b_s  = qt.tensor(Iq, Iq, b)
nb_s = qt.tensor(Iq, Iq, nb)
sm1  = qt.tensor(sm, Iq, Ib);  sp1 = sm1.dag()
sm2  = qt.tensor(Iq, sm, Ib);  sp2 = sm2.dag()
pe1  = sp1 * sm1
pe2  = sp2 * sm2
ne   = pe1 + pe2

# Estados colectivos excitónicos
ket_cv     = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
ket_vc     = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
ket_dark   = (ket_cv - ket_vc) / np.sqrt(2.0)
ket_bright = (ket_cv + ket_vc) / np.sqrt(2.0)

proj_dark_exc   = ket_dark   * ket_dark.dag()
proj_bright_exc = ket_bright * ket_bright.dag()
proj_dark_sys   = qt.tensor(proj_dark_exc,   Ib)
proj_bright_sys = qt.tensor(proj_bright_exc, Ib)

# Base de polarón: |2̃⟩ = D(α)|2⟩
D_op          = qt.displace(Ncut, alpha)
fock2_polaron = D_op * qt.basis(Ncut, 2)
proj_objetivo = qt.tensor(proj_dark_exc,
                           fock2_polaron * fock2_polaron.dag())
proj_fuente   = qt.tensor(proj_bright_exc,
                           fock2_polaron * fock2_polaron.dag())

# Operadores de colapso
c_ops = [
    np.sqrt(kappa)     * b_s,
    np.sqrt(gamma)     * sm1,
    np.sqrt(gamma)     * sm2,
    np.sqrt(gamma_phi) * pe1,
    np.sqrt(gamma_phi) * pe2,
]

# Hamiltonianos
H_prod  = (omega_b * nb_s
           + Delta_2 * ne
           + lam * ne * (b_s + b_s.dag())
           + J * (sp1 * sm2 + sp2 * sm1)
           + Omega_prod * (sp1 + sm1 + sp2 + sm2))

H_pulso = (omega_b * nb_s
           + Delta_2 * ne
           + lam * ne * (b_s + b_s.dag())
           + J * (sp1 * sm2 + sp2 * sm1)
           + Omega_w * (sp1 - sp2 + sm1 - sm2))   # diferencial

# Estado inicial: |vv, 0⟩
psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(Ncut, 0))

# =============================================================================
# CÁLCULO O CARGA
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
CODES_DIR = BASE_DIR.parent.parent
RESULTS_DATA_DIR = CODES_DIR / "results" / "data"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
data_file = RESULTS_DATA_DIR / "04_escritura_trayectoria_data.npz"

if not RERUN:
    rng   = np.random.default_rng(seed_base)
    seeds = [int(rng.integers(1e9)) for _ in range(ntraj)]

    opts_mc = {
        "keep_runs_results": True,
        "store_states":      True,    # necesario para extraer ψ(t*)
        "nsteps":            200000,
        "improved_sampling": True,
        "progress_bar":      "text",
    }

    e_ops_gen = [nb_s, proj_dark_sys, proj_bright_sys]

    print(f"\nEtapa 1: mcsolve de generación ({ntraj} trayectorias)...")
    res = qt.mcsolve(H_prod, psi0, tlist_gen, c_ops,
                     e_ops=e_ops_gen, ntraj=ntraj,
                     options=opts_mc, seeds=seeds)

    # ── Detectar t* en cada trayectoria: primer instante con ⟨n̂⟩ ≥ 1.8 ────
    # Umbral 1.8 en lugar de 2.0 para robustez con resolución temporal
    umbral_n  = 1.8
    t_star_list  = []
    F_w_list     = []   # fidelidad de escritura en cada trayectoria
    F_src_list   = []   # población en estado fuente |Ψ+,2̃⟩ en t*
    n_star_list  = []   # ⟨n̂⟩ en t*
    traj_validas = 0

    print(f"\nEtapa 2: detectar t* y aplicar pulso de escritura...")

    for k in range(ntraj):
        # ⟨n̂⟩(t) para esta trayectoria
        n_traj = res.expect[k][0]   # índice 0 = nb_s

        # Buscar primer índice donde ⟨n̂⟩ ≥ umbral
        idx_star = np.argmax(n_traj >= umbral_n)

        if n_traj[idx_star] < umbral_n:
            # Esta trayectoria nunca alcanzó el umbral
            continue

        t_star = tlist_gen[idx_star]
        psi_star = res.runs_states[k][idx_star]   # estado ψ(t*)
        rho_star = psi_star * psi_star.dag()

        # Verificar que es un estado válido
        if abs(rho_star.tr() - 1.0) > 1e-6:
            continue

        n_star       = qt.expect(nb_s,          rho_star)
        F_src_star   = qt.expect(proj_fuente,    rho_star)
        P_bright_star = qt.expect(proj_bright_sys, rho_star)

        # Solo proceder si hay suficiente población en el sector brillante
        if P_bright_star < 0.05:
            continue

        # Etapa 2: pulso de escritura desde ρ(t*)
        e_ops_w = [proj_objetivo, proj_fuente, nb_s, proj_dark_sys]

        res_w = qt.mesolve(
            H_pulso, rho_star, tlist_w, c_ops, e_ops_w,
            options=qt.Options(nsteps=5000, atol=1e-9, rtol=1e-7)
        )

        F_w_t   = np.array(res_w.expect[0])   # ⟨Ψ_-,2̃|ρ(t)|Ψ_-,2̃⟩
        F_src_t = np.array(res_w.expect[1])   # población fuente

        # Fidelidad en t = t_π y máximo
        idx_tpi = np.argmin(np.abs(tlist_w - t_pi))
        F_at_tpi = F_w_t[idx_tpi]
        F_max    = np.max(F_w_t)

        t_star_list.append(t_star)
        F_w_list.append(F_max)
        F_src_list.append(F_src_star)
        n_star_list.append(n_star)
        traj_validas += 1

        if traj_validas <= 5:
            print(f"  traj {k:2d}: t*={t_star:.1f}  ⟨n̂⟩(t*)={n_star:.3f}  "
                  f"F_src={F_src_star:.4f}  F_w(max)={F_max:.4f}  "
                  f"F_w(t_π)={F_at_tpi:.4f}")

    print(f"\n  Trayectorias válidas: {traj_validas}/{ntraj}")

    t_star_arr  = np.array(t_star_list)
    F_w_arr     = np.array(F_w_list)
    F_src_arr   = np.array(F_src_list)
    n_star_arr  = np.array(n_star_list)

    if traj_validas > 0:
        print(f"\n  ⟨n̂⟩ en t*:          {np.mean(n_star_arr):.3f} ± {np.std(n_star_arr):.3f}")
        print(f"  F_src en t*:         {np.mean(F_src_arr):.4f} ± {np.std(F_src_arr):.4f}")
        print(f"  F_w(max):            {np.mean(F_w_arr):.4f} ± {np.std(F_w_arr):.4f}")
        print(f"  Cociente F_w/F_src:  {np.mean(F_w_arr/F_src_arr):.4f}")

    np.savez(data_file,
             t_star=t_star_arr,
             F_w=F_w_arr,
             F_src=F_src_arr,
             n_star=n_star_arr,
             traj_validas=np.array([traj_validas]),
             ntraj=np.array([ntraj]),
             t_pi=np.array([t_pi]),
             Omega_w=np.array([Omega_w]),
             Delta_2=np.array([Delta_2]),
             umbral_n=np.array([umbral_n]))
    print(f"\nDatos guardados en {data_file}")

else:
    data        = np.load(data_file)
    t_star_arr  = data['t_star']
    F_w_arr     = data['F_w']
    F_src_arr   = data['F_src']
    n_star_arr  = data['n_star']
    traj_validas = int(data['traj_validas'][0])
    t_pi        = float(data['t_pi'][0])
    Omega_w     = float(data['Omega_w'][0])
    umbral_n    = float(data['umbral_n'][0])
    print(f"Datos cargados desde {data_file}")

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "="*60)
print("RESUMEN — Protocolo de escritura condicionada (n=2)")
print("="*60)
print(f"  ntraj total       = {ntraj}")
print(f"  Trayectorias válidas = {traj_validas}")
print(f"  Umbral ⟨n̂⟩        = {umbral_n}")
print(f"  t_π/ω_b           = {t_pi:.4f}")
print()
if traj_validas > 0:
    print(f"  ⟨n̂⟩(t*) promedio   = {np.mean(n_star_arr):.3f} ± {np.std(n_star_arr):.3f}")
    print(f"  F_src promedio      = {np.mean(F_src_arr):.4f} ± {np.std(F_src_arr):.4f}")
    print(f"  F_w(max) promedio   = {np.mean(F_w_arr):.4f} ± {np.std(F_w_arr):.4f}")
    ratio = F_w_arr / np.where(F_src_arr > 1e-6, F_src_arr, np.nan)
    print(f"  F_w/F_src promedio  = {np.nanmean(ratio):.4f}")
    print()
    print("  DIAGNÓSTICO:")
    print(f"  Si F_w/F_src ~ 1 -> pulso transfiere eficientemente |Ψ+,2̃⟩→|Ψ-,2̃⟩")
    print(f"  Si F_w/F_src < 1 -> pérdidas por disipación o mezcla fonónica")
    print(f"  Referencia T_mem(n=2) ~ 1/κ = {1/kappa:.1f}/ω_b >> t_π={t_pi:.1f}/ω_b  ✓")
else:
    print("  ADVERTENCIA: ninguna trayectoria alcanzó el umbral.")
    print("  Sugerencias: aumentar t_gen_max, reducir umbral_n, o aumentar Omega_prod.")
print("="*60)
