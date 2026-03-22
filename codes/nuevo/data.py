#!/usr/bin/env python3
"""
Sweet spot 1QD vs 2QD — Visualización final
=============================================

Usa los datos de sweet_spot_data.npz.

Panel (a): Purezas Π₂ y Π₃ vs κ/ω_b (1QD y 2QD superpuestas)
Panel (b): Tasa R₃ vs κ/ω_b (1QD y 2QD — enhancement superradiante)

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CARGAR DATOS
# =============================================================================
data = np.load('data.npz')
kappa = data['kappa_arr']

pi1_n2 = data['pi_1qd_n2']
pi2_n2 = data['pi_2qd_n2']
pi1_n3 = data['pi_1qd_n3']
pi2_n3 = data['pi_2qd_n3']

R1_n3 = data['R_1qd_n3']
R2_n3 = data['R_2qd_n3']

# =============================================================================
# FIGURA
# =============================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Panel (a): Purezas ──────────────────────────────────────────
ax_a.semilogx(kappa, pi1_n2, 'b-',  lw=1.5, label=r'1QD $\Pi_2$')
ax_a.semilogx(kappa, pi2_n2, 'r--', lw=1.5, label=r'2QD $\Pi_2$')
ax_a.semilogx(kappa, pi1_n3, 'b-',  lw=1.5, alpha=0.5, label=r'1QD $\Pi_3$')
ax_a.semilogx(kappa, pi2_n3, 'r--', lw=1.5, alpha=0.5, label=r'2QD $\Pi_3$')

ax_a.set_xlabel(r'$\kappa/\omega_b$', fontsize=13)
ax_a.set_ylabel(r'$\Pi_n$', fontsize=13)
ax_a.set_ylim(0, 1.05)
ax_a.set_xlim(kappa[0], kappa[-1])
ax_a.legend(fontsize=10, loc='lower left', ncol=2)
ax_a.tick_params(labelsize=10)
ax_a.text(0.03, 0.95, '(a)', transform=ax_a.transAxes,
          fontsize=14, va='top', fontweight='bold')

# ── Panel (b): Tasa R₃ ─────────────────────────────────────────
ax_b.loglog(kappa, R1_n3, 'b-',  lw=1.8, label=r'1QD $R_3$')
ax_b.loglog(kappa, R2_n3, 'r-',  lw=1.8, label=r'2QD $R_3$')

ax_b.set_xlabel(r'$\kappa/\omega_b$', fontsize=13)
ax_b.set_ylabel(r'$R_3 = \kappa \, n_a^{(3)}$', fontsize=13)
ax_b.set_xlim(kappa[0], kappa[-1])
ax_b.legend(fontsize=11, loc='upper right')
ax_b.tick_params(labelsize=10)
ax_b.text(0.03, 0.95, '(b)', transform=ax_b.transAxes,
          fontsize=14, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig("sweet_spot_final.pdf", bbox_inches='tight')
plt.savefig("sweet_spot_final.png", dpi=200, bbox_inches='tight')
print("✓ Figuras guardadas: sweet_spot_final.{pdf,png}")
plt.show()