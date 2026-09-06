# -*- coding: utf-8 -*-
"""
concurrencia_herald_2qds.py
============================================================
Entrelazamiento excitónico heraldado por emisión de fonones
en la molécula excitónica (2QD + Förster).

Calcula:
  (A) Mapa C_herald(λ, κ) sobre la misma grilla de los mapas
      de pureza, con optimización local de Δ en torno a la
      resonancia de Stokes n = 3 (Régimen III + Lamb + Förster).
      C_herald = concurrencia del estado reducido de los QDs
      inmediatamente después de un salto fonónico:
            ρ_j = b ρ_ss b† / Tr[b ρ_ss b†]
  (B) Heraldo de bundle vía Monte Carlo (mcsolve): concurrencia
      condicionada al instante posterior al n-ésimo salto de κD[b]
      dentro de un clúster (bundle), en el mejor punto del mapa.
  (C) Control negativo J = 0 sobre un corte 1D en κ.

Patrón RERUN:
  RERUN = False -> computa y exporta NPZ
  RERUN = True  -> carga NPZ y regenera figuras

Régimen de validez: γ ≪ κ en toda la grilla (verificado en runtime).
============================================================
"""
import numpy as np
from pathlib import Path
import qutip as qt
from math import factorial

RERUN = False

BASE_DIR = Path(__file__).resolve().parent
CODES_DIR = BASE_DIR.parent.parent
RESULTS_DATA_DIR = CODES_DIR / "results" / "data"
RESULTS_OFICIAL_DIR = CODES_DIR / "results" / "oficial"
RESULTS_PGF_DIR = RESULTS_OFICIAL_DIR / "pgf"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OFICIAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PGF_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = RESULTS_DATA_DIR / "concurrencia_herald_2qd_data.npz"

# -----------------------------------------------------------------------------
# PARÁMETROS DE PRODUCCIÓN (unidades de ω_b)
# -----------------------------------------------------------------------------
omega_b   = 1.0
J         = 0.5
gamma     = 0.0002
gamma_phi = 0.0004
Omega     = 0.2
n_bundle  = 3
Ncut      = 10          # convergencia verificada vs Ncut=14 en el mejor punto

# Grilla λ × κ (misma estructura que pureza_2qds.py)
n_lam, n_kappa = 25, 25
lambda_arr = np.linspace(0.05, 0.30, n_lam)
kappa_arr  = np.logspace(-3.3, -1.7, n_kappa)   # κ ∈ [5e-4, 2e-2]

# Optimización local de Δ
n_Delta_opt  = 9
Delta_window = 0.20

# Monte Carlo (heraldo de bundle)
ntraj_mc   = 500
t_max_mc   = 1.0 / gamma            # ventana suficiente para varios bundles
nt_mc      = 4000
# Ventana de clúster: dentro de un bundle los n saltos se separan ~1/κ
# (cascada |n⟩→|n-1⟩→...→|0⟩ a tasa κ), mientras que bundles consecutivos
# se separan ~1/Ω_eff ≫ 1/κ. Se define dinámicamente como 5/κ.


# -----------------------------------------------------------------------------
# OPERADORES
# -----------------------------------------------------------------------------
def build_ops(Nc):
    b  = qt.tensor(qt.destroy(Nc), qt.qeye(2), qt.qeye(2))
    s1 = qt.tensor(qt.qeye(Nc), qt.destroy(2), qt.qeye(2))
    s2 = qt.tensor(qt.qeye(Nc), qt.qeye(2), qt.destroy(2))
    return b, s1, s2

def build_H(b, s1, s2, lam, Delta, Jc):
    Ne = s1.dag()*s1 + s2.dag()*s2
    return (omega_b*b.dag()*b + Delta*Ne + lam*Ne*(b + b.dag())
            + Omega*(s1 + s1.dag() + s2 + s2.dag())
            + Jc*(s1.dag()*s2 + s2.dag()*s1))

def build_cops(b, s1, s2, kappa):
    return [np.sqrt(kappa)*b, np.sqrt(gamma)*s1, np.sqrt(gamma)*s2,
            np.sqrt(gamma_phi)*s1.dag()*s1, np.sqrt(gamma_phi)*s2.dag()*s2]

def validate_rho(rho, tol=1e-7):
    """Traza unitaria, hermiticidad y positividad (autovalor mínimo)."""
    if abs(rho.tr() - 1.0) > tol:        return False
    if not rho.isherm:                   return False
    if np.min(rho.eigenenergies()) < -tol: return False
    return True

def resonance_estimate(lam, Jc):
    """Δ_3 ≈ -√(9ω_b² - 8Ω²) + λ²/ω_b - J  (Régimen III 2QD)."""
    arg = (n_bundle*omega_b)**2 - 8.0*Omega**2
    if arg < 0:
        return -n_bundle*omega_b - Jc
    return -np.sqrt(arg) + lam**2/omega_b - Jc

def concurrences(rho):
    """(C incondicional, C condicional 1-exc, P_1exc) del reducido 2QD."""
    rq = rho.ptrace([1, 2]); rq = rq / rq.tr()
    C  = qt.concurrence(rq)
    e, g = qt.basis(2, 1), qt.basis(2, 0)
    pp = (qt.tensor(e, g) + qt.tensor(g, e)).unit()
    mm = (qt.tensor(e, g) - qt.tensor(g, e)).unit()
    P1 = qt.ket2dm(pp) + qt.ket2dm(mm)
    r1 = P1 * rq * P1
    p1 = np.real(r1.tr())
    Cc = qt.concurrence(r1 / p1) if p1 > 1e-12 else 0.0
    return C, Cc, p1

# -----------------------------------------------------------------------------
# (A) MAPA C_herald(λ, κ)
# -----------------------------------------------------------------------------
def compute_point(b, s1, s2, lam, kappa, Jc):
    """
    Optimiza Δ en torno a la resonancia (maximizando ⟨n̂⟩, proxy de
    operación en resonancia de Stokes) y devuelve las concurrencias
    estacionaria y post-salto en el Δ óptimo.
    """
    cops = build_cops(b, s1, s2, kappa)
    D0   = resonance_estimate(lam, Jc)
    best = dict(nb=-1.0)
    for D in np.linspace(D0 - Delta_window, D0 + Delta_window, n_Delta_opt):
        H = build_H(b, s1, s2, lam, D, Jc)
        try:
            rho = qt.steadystate(H, cops, method='direct', use_rcm=True)
        except Exception:
            continue
        if not validate_rho(rho):
            continue
        nb = np.real(qt.expect(b.dag()*b, rho))
        if nb > best['nb']:
            best = dict(nb=nb, rho=rho, Delta=D)
    if best['nb'] < 0:
        return [np.nan]*6
    rho = best['rho']
    C_ss, Cc_ss, _ = concurrences(rho)
    rho_j = b * rho * b.dag()
    tr_j  = np.real(rho_j.tr())
    if tr_j < 1e-14:
        return [C_ss, Cc_ss, np.nan, np.nan, best['nb'], best['Delta']]
    rho_j = rho_j / tr_j
    C_h, Cc_h, _ = concurrences(rho_j)
    return [C_ss, Cc_ss, C_h, Cc_h, best['nb'], best['Delta']]

if not RERUN:
    assert gamma < kappa_arr.min(), "Violación del régimen γ ≪ κ en la grilla"

    b, s1, s2 = build_ops(Ncut)
    maps = {k: np.full((n_lam, n_kappa), np.nan)
            for k in ('C_ss', 'Cc_ss', 'C_h', 'Cc_h', 'nb', 'Delta')}

    print(f"Mapa C_herald: {n_lam}x{n_kappa}, Ncut={Ncut}, J={J}")
    for i, lam in enumerate(lambda_arr):
        for k, kap in enumerate(kappa_arr):
            vals = compute_point(b, s1, s2, lam, kap, J)
            for key, v in zip(maps, vals):
                maps[key][i, k] = v
        print(f"  λ = {lam:.3f} listo "
              f"(C_h max fila = {np.nanmax(maps['C_h'][i]):.3f})")

    # -------------------------------------------------------------------------
    # (C) CONTROL NEGATIVO J = 0 — corte 1D en λ = 0.22
    # -------------------------------------------------------------------------
    lam_best = 0.22
    Ch_J0  = np.full(n_kappa, np.nan)
    Ch_J   = np.full(n_kappa, np.nan)
    for k, kap in enumerate(kappa_arr):
        Ch_J[k]  = compute_point(b, s1, s2, lam_best, kap, J)[2]
        Ch_J0[k] = compute_point(b, s1, s2, lam_best, kap, 0.0)[2]
    print(f"Control J=0: C_h max = {np.nanmax(Ch_J0):.3f} "
          f"(vs {np.nanmax(Ch_J):.3f} con J={J})")

    # -------------------------------------------------------------------------
    # (B) HERALDO DE BUNDLE — Monte Carlo en el mejor punto del mapa
    # -------------------------------------------------------------------------
    idx = np.unravel_index(np.nanargmax(maps['C_h']), maps['C_h'].shape)
    lam_mc, kap_mc = lambda_arr[idx[0]], kappa_arr[idx[1]]
    D_mc = maps['Delta'][idx]
    print(f"MC en λ={lam_mc:.3f}, κ={kap_mc:.2e}, Δ={D_mc:.4f}")

    H_mc    = build_H(b, s1, s2, lam_mc, D_mc, J)
    cops_mc = build_cops(b, s1, s2, kap_mc)
    psi0    = qt.tensor(qt.basis(Ncut, 0), qt.basis(2, 0), qt.basis(2, 0))
    tlist   = np.linspace(0, t_max_mc, nt_mc)

    res = qt.mcsolve(H_mc, psi0, tlist, cops_mc, ntraj=ntraj_mc,
                     options={'store_states': True,
                              'keep_runs_results': True,
                              'progress_bar': False})

    # Concurrencia condicionada a saltos fonónicos dentro de clústeres.
    # Heraldo físico: el PRIMER fonón del clúster (consistente con el
    # estado post-salto determinista b·ρ_ss·b†). Se registra también el
    # n-ésimo salto para contraste (estado al final de la cascada).
    # Intra-bundle: separación ~1/κ; ventana de clúster = 2/κ.
    C_first, C_nth = [], []
    bwin = 2.0 / kap_mc
    for itr in range(ntraj_mc):
        jt = np.asarray(res.col_times[itr])
        jw = np.asarray(res.col_which[itr])
        tb = jt[jw == 0]                       # tiempos de saltos fonónicos
        if len(tb) == 0:
            continue
        splits = np.where(np.diff(tb) > bwin)[0] + 1
        for cl in np.split(tb, splits):
            for tlabel, lst, need in ((cl[0], C_first, 1),
                                      (cl[n_bundle-1] if len(cl) >= n_bundle
                                       else None, C_nth, n_bundle)):
                if tlabel is None or len(cl) < need:
                    continue
                j = np.searchsorted(tlist, tlabel)
                if j >= nt_mc:
                    continue
                psi = res.runs_states[itr][j]
                rho = qt.ket2dm(psi) if psi.isket else psi
                lst.append(concurrences(rho)[0])
    C_first, C_nth = np.array(C_first), np.array(C_nth)
    def _rep(name, a):
        if len(a):
            print(f"  {name}: N={len(a)}, ⟨C⟩={a.mean():.3f}±{a.std():.3f}, "
                  f"P(C>0.5)={(a > 0.5).mean():.2f}")
        else:
            print(f"  {name}: sin eventos")
    _rep("Heraldo 1er fonón ", C_first)
    _rep(f"Heraldo {n_bundle}º fonón", C_nth)

    np.savez(DATA_FILE,
             lambda_arr=lambda_arr, kappa_arr=kappa_arr,
             **maps,
             Ch_J=Ch_J, Ch_J0=Ch_J0, lam_best=lam_best,
             C_first=C_first, C_nth=C_nth, lam_mc=lam_mc, kap_mc=kap_mc, D_mc=D_mc,
             params=np.array([omega_b, J, gamma, gamma_phi, Omega,
                              n_bundle, Ncut]))
    print(f"Datos exportados a {DATA_FILE}")

# -----------------------------------------------------------------------------
# FIGURAS (RERUN = True)
# -----------------------------------------------------------------------------
else:
    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex", "font.family": "serif",
        "text.usetex": True, "pgf.rcfonts": False})

    d = np.load(DATA_FILE)
    lam, kap = d['lambda_arr'], d['kappa_arr']

    fig, axes = plt.subplots(1, 2, figsize=(6.496, 2.9))
    plt.subplots_adjust(left=0.17, right=0.94, top=0.94,
                        bottom=0.10, wspace=0.10)

    # (a) Mapa C_herald(λ, κ)
    ax = axes[0]
    pc = ax.pcolormesh(kap, lam, d['C_h'], cmap='magma',
                       vmin=0, vmax=1, shading='auto')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\kappa/\omega_b$', fontsize=12)
    ax.set_ylabel(r'$\lambda/\omega_b$', fontsize=12)
    ax.text(0.05, 0.92, r'(a)', transform=ax.transAxes, fontsize=10,
            color='w')
    fig.colorbar(pc, ax=ax, label=r'$\mathcal{C}_{\rm herald}$')

    # (b) Control J = 0 + histograma de bundle
    ax = axes[1]
    ax.semilogx(kap, d['Ch_J'],  lw=0.9, label=r'$J/\omega_b=0.5$')
    ax.semilogx(kap, d['Ch_J0'], lw=0.9, ls='--', label=r'$J=0$')
    ax.axhline(d['C_first'].mean(), color='gray', lw=0.9, ls=':',
               label=r'$\langle\mathcal{C}\rangle_{\rm herald}$ (MC)')
    ax.set_xlabel(r'$\kappa/\omega_b$', fontsize=12)
    ax.set_ylabel(r'$\mathcal{C}_{\rm herald}$', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.text(0.05, 0.92, r'(b)', transform=ax.transAxes, fontsize=10)

    pgf_out = RESULTS_PGF_DIR / 'concurrencia_herald_2qd.pgf'
    pdf_out = RESULTS_OFICIAL_DIR / 'concurrencia_herald_2qd.pdf'
    fig.savefig(pgf_out)
    fig.savefig(pdf_out)
    print(f"Figuras generadas en: {pdf_out} y {pgf_out}")
