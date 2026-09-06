#!/usr/bin/env python3
"""
fig_concurrencia.py — Figura de concurrencia heraldada
=======================================================
Tres paneles:
  (a) Mapa C_herald(λ, κ)
  (b) Corte 1D J=0.5 vs J=0
  (c) Histograma MC en escala de 1-C (log) — datos concentrados en C~0.99
Convenciones: figsize 6.496 in, lw=0.9, fontsize 12/10, pgf backend.
Autor: Jhon S. García B.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts":   False,
    "font.family":   "serif",
    "font.size":     12,
})

# =============================================================================
# RUTAS Y DATOS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
CODES_DIR = BASE_DIR.parent.parent
DATA_FILE = CODES_DIR / 'results' / 'data' / 'concurrencia_herald_2qd_data.npz'
FIGS_DIR = CODES_DIR / 'results' / 'oficial'
PGF_DIR = FIGS_DIR / 'pgf'

data = np.load(DATA_FILE, allow_pickle=True)

lambda_arr = data['lambda_arr']
kappa_arr  = data['kappa_arr']
C_h        = data['C_h']
Ch_J       = data['Ch_J']
Ch_J0      = data['Ch_J0']
C_first    = data['C_first']
C_nth      = data['C_nth']
lam_mc     = float(data['lam_mc'])
kap_mc     = float(data['kap_mc'])

# =============================================================================
# FIGURA
# =============================================================================
fig = plt.figure(figsize=(6.496, 4.80))

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.10, right=0.97,
                       top=0.94, bottom=0.12,
                       hspace=0.42, wspace=0.38)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# ─── (a) Mapa C_herald(λ, κ) ─────────────────────────────────────────────────
LAM, KAP = np.meshgrid(lambda_arr, kappa_arr, indexing='ij')
im = ax_a.pcolormesh(LAM, KAP, C_h,
                     cmap='viridis', vmin=0.0, vmax=0.70,
                     shading='auto')
ax_a.set_yscale('log')

# Máximo de C_herald
idx_max  = np.unravel_index(np.argmax(C_h), C_h.shape)
ax_a.plot(lambda_arr[idx_max[0]], kappa_arr[idx_max[1]],
          'w*', ms=7, zorder=5, label=r'$C_{\rm max}$')

# Punto óptimo de pureza de la tesis
ax_a.plot(0.22, 0.001, 'w^', ms=5, zorder=5,
          markerfacecolor='none', markeredgewidth=0.9,
          label=r'$\lambda^*_{\Pi_n}$')

ax_a.set_xlabel(r'$\lambda/\omega_b$', fontsize=12)
ax_a.set_ylabel(r'$\kappa/\omega_b$', fontsize=12)
ax_a.text(0.05, 0.95, r'(a)', transform=ax_a.transAxes,
          fontsize=10, ha='left', va='top', color='white')
ax_a.legend(fontsize=8, loc='lower right',
            facecolor='none', edgecolor='none',
            labelcolor='white')

cbar = fig.colorbar(im, ax=ax_a, pad=0.03)
cbar.set_label(r'$C_{\rm herald}$', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# ─── (b) Corte 1D J=0.5 vs J=0 ───────────────────────────────────────────────
ax_b.plot(lambda_arr, Ch_J,  lw=0.9, color='steelblue',
          label=r'$J/\omega_b=0.5$')
ax_b.plot(lambda_arr, Ch_J0, lw=0.9, color='steelblue', ls='--',
          label=r'$J=0$')
ax_b.fill_between(lambda_arr, Ch_J0, Ch_J,
                  alpha=0.20, color='steelblue')

# Línea vertical en λ óptimo de pureza
ax_b.axvline(0.22, color='gray', lw=0.7, ls=':', alpha=0.8)
ax_b.text(0.223, 0.44, r'$\lambda^*_{\Pi_n}$',
          fontsize=9, color='gray', va='bottom')

# Anotar brecha Förster en el máximo
i_max = np.argmax(Ch_J)
dC    = Ch_J[i_max] - Ch_J0[i_max]
ymid  = (Ch_J[i_max] + Ch_J0[i_max]) / 2.0
ax_b.annotate('', xy=(lambda_arr[i_max], Ch_J[i_max]),
              xytext=(lambda_arr[i_max], Ch_J0[i_max]),
              arrowprops=dict(arrowstyle='<->', color='steelblue', lw=0.8))
ax_b.text(lambda_arr[i_max] + 0.006, ymid,
          rf'$\Delta C\approx{dC:.3f}$',
          fontsize=8, color='steelblue', va='center')

ax_b.set_xlabel(r'$\lambda/\omega_b$', fontsize=12)
ax_b.set_ylabel(r'$C_{\rm herald}$', fontsize=12)
ax_b.set_ylim(0.20, 0.78)
ax_b.legend(fontsize=9, loc='lower right')
ax_b.text(0.05, 0.95, r'(b)', transform=ax_b.transAxes,
          fontsize=10, ha='left', va='top')

# ─── (c) Distribución MC en escala 1-C ───────────────────────────────────────
# Datos concentrados en C~0.99 → graficar 1-C en escala log
eps       = 1e-4   # evitar log(0)
one_minus_C1 = np.clip(1.0 - C_first, eps, 1.0)
one_minus_Cn = np.clip(1.0 - C_nth,   eps, 1.0)

bins_log = np.logspace(np.log10(eps), 0, 30)

ax_c.hist(one_minus_C1, bins=bins_log, density=True,
          alpha=0.55, color='seagreen', edgecolor='seagreen', lw=0.5,
          label=(rf'1$^\circ$ fonón: '
                 rf'$\langle C\rangle={C_first.mean():.3f}\pm{C_first.std():.3f}$,'
                 rf'\ $P(C>0.5)={np.mean(C_first>0.5):.2f}$'
                 rf'\ ($N={len(C_first)}$)'))
ax_c.hist(one_minus_Cn, bins=bins_log, density=True,
          alpha=0.55, color='darkorange', edgecolor='darkorange', lw=0.5,
          label=(rf'$n$-ésimo fonón: '
                 rf'$\langle C\rangle={C_nth.mean():.3f}\pm{C_nth.std():.3f}$,'
                 rf'\ $P(C>0.5)={np.mean(C_nth>0.5):.2f}$'
                 rf'\ ($N={len(C_nth)}$)'))

ax_c.set_xscale('log')
ax_c.axvline(0.5, color='gray', lw=0.7, ls='--', alpha=0.8)
ax_c.text(0.55, ax_c.get_ylim()[1] if ax_c.get_ylim()[1] > 0 else 1.0,
          r'$C=0.5$', fontsize=9, color='gray', va='bottom')

ax_c.set_xlabel(r'$1 - C$', fontsize=12)
ax_c.set_ylabel(r'Densidad', fontsize=12)
ax_c.legend(fontsize=8, loc='upper left')
ax_c.text(0.01, 0.95, r'(c)', transform=ax_c.transAxes,
          fontsize=10, ha='left', va='top')
ax_c.set_title(rf'$\lambda/\omega_b={lam_mc:.4f},\;\kappa/\omega_b={kap_mc:.2e}$',
               fontsize=9, pad=3)

# =============================================================================
# GUARDAR
# =============================================================================
FIGS_DIR.mkdir(parents=True, exist_ok=True)
PGF_DIR.mkdir(parents=True, exist_ok=True)

pdf_out = FIGS_DIR / 'concurrencia_herald.pdf'
pgf_out = PGF_DIR / 'concurrencia_herald.pgf'

plt.savefig(pdf_out, bbox_inches='tight')
plt.savefig(pgf_out)
print("Guardado:")
print(f"  {pdf_out}")
print(f"  {pgf_out}")
