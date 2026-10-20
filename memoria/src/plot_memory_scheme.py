"""Raman-level schematic for the three-dot phonon memory (not simulation data)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from memstyle import BLACK, BLUE, GOLD, GREEN, RED, save


fig, ax = plt.subplots(figsize=(3.35, 2.35))
ax.set(xlim=(0, 1.1), ylim=(0, 1))
ax.axis("off")

# Bare levels are schematic. Each Raman branch is brought into two-photon
# resonance by a different laser frequency; the drawing is not one rotating
# frame in which both dark states are resonant at once.
levels = [
    (0.08, 0.28, 0.23, BLACK, r"$|G,1\rangle$"),
    (0.57, 0.85, 0.36, RED, r"$|B,1\rangle$"),
    (0.57, 0.55, 0.25, BLUE, r"$|D_1,0\rangle$"),
    (0.57, 0.29, 0.36, GREEN, r"$|D_2,0\rangle$"),
]
for x, y, width, color, name in levels:
    ax.plot([x, x + width], [y, y], lw=1.7, color=color)
    label_y = y - 0.025 if "D_1" in name else y + 0.045
    ax.text(x + width / 2, label_y, name, color=color,
            ha="center", va="top" if "D_1" in name else "bottom", fontsize=8)

ax.add_patch(FancyArrowPatch((0.28, 0.30), (0.58, 0.83),
                             arrowstyle="<->", mutation_scale=9,
                             lw=1.15, color=GOLD))
ax.text(0.31, 0.64, r"$\Omega_s(t),\ \omega_{L,s}$", color=GOLD,
        ha="center", fontsize=8)

for x, y, color, label, tx, ty in [
    (0.66, 0.55, BLUE, r"$g_1$", 0.58, 0.68),
    (0.89, 0.29, GREEN, r"$g_2$", 0.95, 0.56),
]:
    ax.add_patch(FancyArrowPatch((x, 0.83), (x, y + 0.015),
                                 arrowstyle="<->", mutation_scale=9,
                                 lw=1.05, color=color))
    ax.text(tx, ty, label, color=color, fontsize=8)

fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.16)
save(fig, "figM0_memory_scheme")
