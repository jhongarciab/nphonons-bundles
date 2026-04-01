"""
Oscilaciones Super-Rabi — 2QD con Förster

Tres regímenes de oscilaciones tipo Rabi para el modelo 2QD.
Línea sólida: solución exacta del Hamiltoniano efectivo proyectado
  H_eff = Ω_eff^(n) (|e⟩⟨g| + h.c.)  en subespacio {|0,vv⟩, |n,Ψ+⟩}

Parámetros consistentes con los resultados del proyecto:
  Panel (a) Régimen I:   λ=0.08, Ω=0.01  (forster_2qds.py)
  Panel (b) Régimen II:  λ=0.20, Ω=0.01  (λ dentro de pureza_2qds.py)
  Panel (c) Régimen III: λ=0.08, Ω=0.50  (n²ω_b²-8Ω²=2>0 ✓)

J = 0.5 en todos los paneles.

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import matplotlib
matplotlib.use("pgf")

import numpy as np
import math
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
RERUN = True  # False = recalcular; True = cargar datos guardados

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 12,
})

omegab = 1.0
J      = 0.5

# -----------------------------------------------------------------------------
# Helpers visuales
# -----------------------------------------------------------------------------
def fix_main_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1e0, 7e4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_locator(FixedLocator([1e1, 1e4]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(labelsize=11)

def fix_inset_axis(axins):
    axins.set_xscale("log")
    axins.set_xlim(1e2, 1e6)
    axins.set_ylim(0, 1)
    axins.set_yticks([0, 1])
    axins.xaxis.set_major_locator(FixedLocator([1e3, 1e5]))
    axins.xaxis.set_major_formatter(LogFormatterMathtext())
    axins.tick_params(labelsize=10)

# -----------------------------------------------------------------------------
# Hamiltoniano efectivo 2×2
# -----------------------------------------------------------------------------
def sesolve_eff(Omega_eff, tlist):
    """Evolución exacta en subespacio {|g⟩,|e⟩} con H_eff=Ω_eff(σ_x)."""
    g    = qt.basis(2, 0)
    e    = qt.basis(2, 1)
    Heff = Omega_eff * (e * g.dag() + g * e.dag())
    res  = qt.sesolve(Heff, g, tlist, e_ops=[e * e.dag()])
    P_e  = np.real(res.expect[0])
    return P_e, 1.0 - P_e

# -----------------------------------------------------------------------------
# Fórmulas de Ω_eff por régimen
# -----------------------------------------------------------------------------
def omega_eff_regI(lam, Omega, n):
    """Régimen I: perturbativo puro, sin factor Franck-Condon."""
    return np.sqrt(2) * Omega / math.sqrt(math.factorial(n)) * (lam/omegab)**n

def omega_eff_regII(lam, Omega, n):
    """Régimen II: incluye factor Franck-Condon e^{-λ²/2ω_b²}."""
    return (np.sqrt(2) * Omega / math.sqrt(math.factorial(n))
            * (lam/omegab)**n * np.exp(-lam**2 / (2*omegab**2)))

def omega_eff_regIII(lam, Omega, n):
    """Régimen III: dressed states, válido para n²ω_b²>8Ω²."""
    Delta_n = -np.sqrt((n*omegab)**2 - 8*Omega**2) - J
    denom   = Delta_n**2 + 8*Omega**2 - Delta_n*np.sqrt(Delta_n**2 + 8*Omega**2)
    c_minus = np.sqrt(4*Omega**2 / denom)
    prod = 1.0
    for k in range(1, n):
        prod *= (n * c_minus**2 - k)
    return abs((-1)**n * np.sqrt(2) * Omega * (lam/omegab)**n
               * prod / (math.factorial(n-1) * math.sqrt(math.factorial(n))))

# -----------------------------------------------------------------------------
# Mallas temporales — adaptadas a cada Ω_eff
# Los ejes se mantienen log hasta 1.8e6 / 7e7 para comparabilidad
# -----------------------------------------------------------------------------
def make_tlist(Oeff, t_end, n_pts=9000):
    """Malla log desde t=1 hasta t_end, garantizando al menos un período."""
    T_osc = np.pi / Oeff
    t_max = max(t_end, 3 * T_osc)
    return np.logspace(0, np.log10(t_max), n_pts)

# -----------------------------------------------------------------------------
# Cálculo
# -----------------------------------------------------------------------------
if not RERUN:

    # ── Panel (a) — Régimen I: λ=0.08, Ω=0.01 ──────────────────────────────
    lam_a, Om_a = 0.08, 0.01
    print("Panel (a) n=2...", flush=True)
    Oeff_a2 = omega_eff_regI(lam_a, Om_a, 2)
    t_a = make_tlist(Oeff_a2, 1.8e6)
    a_Pnc, a_P0v = sesolve_eff(Oeff_a2, t_a)
    a_Pnc_an = np.sin(Oeff_a2 * t_a)**2
    a_P0v_an = 1.0 - a_Pnc_an
    print(f"  Ω_eff={Oeff_a2:.3e}  T_osc={np.pi/Oeff_a2:.3e}  ✓", flush=True)

    print("Panel (a) n=3 (inset)...", flush=True)
    Oeff_a3 = omega_eff_regI(lam_a, Om_a, 3)
    t_a_ins = make_tlist(Oeff_a3, 7e7)
    a_Pnc_ins, a_P0v_ins = sesolve_eff(Oeff_a3, t_a_ins)
    a_Pnc_ins_an = np.sin(Oeff_a3 * t_a_ins)**2
    a_P0v_ins_an = 1.0 - a_Pnc_ins_an
    print(f"  Ω_eff={Oeff_a3:.3e}  T_osc={np.pi/Oeff_a3:.3e}  ✓", flush=True)

    # ── Panel (b) — Régimen II: λ=0.20, Ω=0.01 ─────────────────────────────
    lam_b, Om_b = 0.20, 0.01
    print("Panel (b) n=2...", flush=True)
    Oeff_b2 = omega_eff_regII(lam_b, Om_b, 2)
    t_b = make_tlist(Oeff_b2, 1.8e6)
    b_Pnc, b_P0v = sesolve_eff(Oeff_b2, t_b)
    b_Pnc_an = np.sin(Oeff_b2 * t_b)**2
    b_P0v_an = 1.0 - b_Pnc_an
    print(f"  Ω_eff={Oeff_b2:.3e}  T_osc={np.pi/Oeff_b2:.3e}  ✓", flush=True)

    print("Panel (b) n=3 (inset)...", flush=True)
    Oeff_b3 = omega_eff_regII(lam_b, Om_b, 3)
    t_b_ins = make_tlist(Oeff_b3, 7e7)
    b_Pnc_ins, b_P0v_ins = sesolve_eff(Oeff_b3, t_b_ins)
    b_Pnc_ins_an = np.sin(Oeff_b3 * t_b_ins)**2
    b_P0v_ins_an = 1.0 - b_Pnc_ins_an
    print(f"  Ω_eff={Oeff_b3:.3e}  T_osc={np.pi/Oeff_b3:.3e}  ✓", flush=True)

    # ── Panel (c) — Régimen III: λ=0.08, Ω=0.50 ────────────────────────────
    lam_c, Om_c = 0.08, 0.50
    # Verificar condición de validez
    check = (2*omegab)**2 - 8*Om_c**2
    print(f"Panel (c) condición n²ω_b²-8Ω²={check:.3f} ({'✓' if check>0 else '✗'})", flush=True)

    print("Panel (c) n=2...", flush=True)
    Oeff_c2 = omega_eff_regIII(lam_c, Om_c, 2)
    t_c = make_tlist(Oeff_c2, 1.8e6)
    c_Pnc, c_P0v = sesolve_eff(Oeff_c2, t_c)
    c_Pnc_an = np.sin(Oeff_c2 * t_c)**2
    c_P0v_an = 1.0 - c_Pnc_an
    print(f"  Ω_eff={Oeff_c2:.3e}  T_osc={np.pi/Oeff_c2:.3e}  ✓", flush=True)

    print("Panel (c) n=3 (inset)...", flush=True)
    Oeff_c3 = omega_eff_regIII(lam_c, Om_c, 3)
    t_c_ins = make_tlist(Oeff_c3, 7e7)
    c_Pnc_ins, c_P0v_ins = sesolve_eff(Oeff_c3, t_c_ins)
    c_Pnc_ins_an = np.sin(Oeff_c3 * t_c_ins)**2
    c_P0v_ins_an = 1.0 - c_Pnc_ins_an
    print(f"  Ω_eff={Oeff_c3:.3e}  T_osc={np.pi/Oeff_c3:.3e}  ✓", flush=True)

    np.savez(
        "results/data/rabi_allreg_data.npz",
        t_a=t_a,         t_a_ins=t_a_ins,
        t_b=t_b,         t_b_ins=t_b_ins,
        t_c=t_c,         t_c_ins=t_c_ins,
        a_Pnc=a_Pnc,     a_P0v=a_P0v,
        a_Pnc_an=a_Pnc_an, a_P0v_an=a_P0v_an,
        a_Pnc_ins=a_Pnc_ins, a_P0v_ins=a_P0v_ins,
        a_Pnc_ins_an=a_Pnc_ins_an, a_P0v_ins_an=a_P0v_ins_an,
        b_Pnc=b_Pnc,     b_P0v=b_P0v,
        b_Pnc_an=b_Pnc_an, b_P0v_an=b_P0v_an,
        b_Pnc_ins=b_Pnc_ins, b_P0v_ins=b_P0v_ins,
        b_Pnc_ins_an=b_Pnc_ins_an, b_P0v_ins_an=b_P0v_ins_an,
        c_Pnc=c_Pnc,     c_P0v=c_P0v,
        c_Pnc_an=c_Pnc_an, c_P0v_an=c_P0v_an,
        c_Pnc_ins=c_Pnc_ins, c_P0v_ins=c_P0v_ins,
        c_Pnc_ins_an=c_Pnc_ins_an, c_P0v_ins_an=c_P0v_ins_an,
    )
    print("✓ Datos guardados")

else:
    data = np.load("results/data/rabi_allreg_data.npz")
    t_a=data["t_a"];         t_a_ins=data["t_a_ins"]
    t_b=data["t_b"];         t_b_ins=data["t_b_ins"]
    t_c=data["t_c"];         t_c_ins=data["t_c_ins"]
    a_Pnc=data["a_Pnc"];     a_P0v=data["a_P0v"]
    a_Pnc_an=data["a_Pnc_an"]; a_P0v_an=data["a_P0v_an"]
    a_Pnc_ins=data["a_Pnc_ins"]; a_P0v_ins=data["a_P0v_ins"]
    a_Pnc_ins_an=data["a_Pnc_ins_an"]; a_P0v_ins_an=data["a_P0v_ins_an"]
    b_Pnc=data["b_Pnc"];     b_P0v=data["b_P0v"]
    b_Pnc_an=data["b_Pnc_an"]; b_P0v_an=data["b_P0v_an"]
    b_Pnc_ins=data["b_Pnc_ins"]; b_P0v_ins=data["b_P0v_ins"]
    b_Pnc_ins_an=data["b_Pnc_ins_an"]; b_P0v_ins_an=data["b_P0v_ins_an"]
    c_Pnc=data["c_Pnc"];     c_P0v=data["c_P0v"]
    c_Pnc_an=data["c_Pnc_an"]; c_P0v_an=data["c_P0v_an"]
    c_Pnc_ins=data["c_Pnc_ins"]; c_P0v_ins=data["c_P0v_ins"]
    c_Pnc_ins_an=data["c_Pnc_ins_an"]; c_P0v_ins_an=data["c_P0v_ins_an"]

# -----------------------------------------------------------------------------
# FIGURA
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6.00, 3.90), sharex=True)
kw = dict(lw=0.9)
skip = 200

# ── Panel (a) ────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(omegab * t_a, a_P0v, color="black", **kw)
ax.plot(omegab * t_a, a_Pnc, color="blue",  **kw)
ax.plot((omegab * t_a)[::skip], a_P0v_an[::skip], 'o', color="black", ms=1, alpha=0.6)
ax.plot((omegab * t_a)[::skip], a_Pnc_an[::skip], 'o', color="blue", ms=1, alpha=0.6)
fix_main_axis(ax)
ax.tick_params(labelbottom=False)
ax.text(0.63, 0.78, r"$P_{0vv}$",     color="black", fontsize=11, transform=ax.transAxes)
ax.text(0.63, 0.15, r"$P_{2\Psi_+}$", color="blue",  fontsize=11, transform=ax.transAxes)

axins = inset_axes(ax, width="33%", height="95%",
                   bbox_to_anchor=(0.19, 0.30, 0.80, 0.70),
                   bbox_transform=ax.transAxes, loc="upper left")
axins.plot(omegab * t_a_ins, a_P0v_ins, color="black", **kw)
axins.plot(omegab * t_a_ins, a_Pnc_ins, color="green", **kw)
axins.plot((omegab * t_a_ins)[::skip], a_P0v_ins_an[::skip], 'o', color="black", ms=0.5, alpha=0.6)
axins.plot((omegab * t_a_ins)[::skip], a_Pnc_ins_an[::skip], 'o', color="green", ms=0.5, alpha=0.6)
fix_inset_axis(axins)
axins.text(0.50, 0.78, r"$P_{0vv}$",     color="black", fontsize=9, transform=axins.transAxes)
axins.text(0.50, 0.15, r"$P_{3\Psi_+}$", color="green", fontsize=9, transform=axins.transAxes)

# ── Panel (b) ────────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(omegab * t_b, b_P0v, color="black", **kw)
ax.plot(omegab * t_b, b_Pnc, color="blue",  **kw)
ax.plot((omegab * t_b)[::skip], b_P0v_an[::skip], 'o', color="black", ms=1, alpha=0.6)
ax.plot((omegab * t_b)[::skip], b_Pnc_an[::skip], 'o', color="blue", ms=1, alpha=0.6)
fix_main_axis(ax)
ax.tick_params(labelbottom=False)
ax.set_ylabel("Poblaciones de los estados del sistema", fontsize=11, labelpad=8)
ax.text(0.53, 0.78, r"$P_{\bar{0}vv}$",     color="black", fontsize=11, transform=ax.transAxes)
ax.text(0.53, 0.15, r"$P_{\bar{2}\Psi_+}$", color="blue",  fontsize=11, transform=ax.transAxes)

axins = inset_axes(ax, width="33%", height="95%",
                   bbox_to_anchor=(0.135, 0.30, 0.80, 0.70),
                   bbox_transform=ax.transAxes, loc="upper left")
axins.plot(omegab * t_b_ins, b_P0v_ins, color="black", **kw)
axins.plot(omegab * t_b_ins, b_Pnc_ins, color="green", **kw)
axins.plot((omegab * t_b_ins)[::skip], b_P0v_ins_an[::skip], 'o', color="black", ms=0.5, alpha=0.6)
axins.plot((omegab * t_b_ins)[::skip], b_Pnc_ins_an[::skip], 'o', color="green", ms=0.5, alpha=0.6)
fix_inset_axis(axins)
axins.text(0.22, 0.78, r"$P_{\bar{0}vv}$",     color="black", fontsize=9, transform=axins.transAxes)
axins.text(0.22, 0.15, r"$P_{\bar{3}\Psi_+}$", color="green", fontsize=9, transform=axins.transAxes)

# ── Panel (c) ────────────────────────────────────────────────────────────────
ax = axes[2]
ax.plot(omegab * t_c, c_P0v, color="black", **kw)
ax.plot(omegab * t_c, c_Pnc, color="blue",  **kw)
ax.plot((omegab * t_c)[::skip], c_P0v_an[::skip], 'o', color="black", ms=1, alpha=0.6)
ax.plot((omegab * t_c)[::skip], c_Pnc_an[::skip], 'o', color="blue", ms=1, alpha=0.6)
fix_main_axis(ax)
ax.set_xlabel(r"$\omega_b\,t$", fontsize=12)
ax.text(0.49, 0.78, r"$P_{0+}$", color="black", fontsize=11, transform=ax.transAxes)
ax.text(0.49, 0.15, r"$P_{2-}$", color="blue",  fontsize=11, transform=ax.transAxes)

axins = inset_axes(ax, width="33%", height="95%",
                   bbox_to_anchor=(0.08, 0.30, 0.80, 0.70),
                   bbox_transform=ax.transAxes, loc="upper left")
axins.plot(omegab * t_c_ins, c_P0v_ins, color="black", **kw)
axins.plot(omegab * t_c_ins, c_Pnc_ins, color="green", **kw)
axins.plot((omegab * t_c_ins)[::skip], c_P0v_ins_an[::skip], 'o', color="black", ms=0.5, alpha=0.6)
axins.plot((omegab * t_c_ins)[::skip], c_Pnc_ins_an[::skip], 'o', color="green", ms=0.5, alpha=0.6)
fix_inset_axis(axins)
axins.text(0.15, 0.78, r"$P_{0+}$", color="black", fontsize=9, transform=axins.transAxes)
axins.text(0.15, 0.15, r"$P_{3-}$", color="green", fontsize=9, transform=axins.transAxes)

# ── Etiquetas (a),(b),(c) ─────────────────────────────────────────────────────
for idx, ax in enumerate(axes):
    ax.text(0.004, 0.95, f'$({chr(97+idx)})$',
            transform=ax.transAxes, ha='left', va='top', fontsize=12)

# ── Salida ────────────────────────────────────────────────────────────────────
plt.tight_layout()
fig.subplots_adjust(hspace=0.16)
plt.savefig("results/oficial/rabi_allreg.pdf", bbox_inches="tight")
plt.savefig("results/oficial/pgf/rabi_allreg.pgf")
plt.close()
print("✓ Figuras guardadas")