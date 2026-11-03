"""Kit de figuras en estilo de revista (PRA / AQT) para el grupo.
Calibrado contra figuras de arXiv:2609.18547 (PRA 114, 033734), 2606.10319 (AQT), 2610.26126 (PRA 114, 043701).
Rasgos reproducidos: fuentes tipo Computer Modern (mathtext cm), ejes con marcas hacia dentro en los cuatro lados,
etiquetas de panel (a),(b)... blancas sobre cuadro negro, lineas de color, linea vertical discontinua en el optimo con su valor anotado,
mapas 2D con barra de color y curva analitica discontinua superpuesta, esquema dibujado con llaves de cavidad y atomo con degradado,
diagramas de niveles con flechas de transicion. Anchos APS: 1 columna 3.4 in, 2 columnas 7.0 in.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap

# paleta (inspirada en las referencias: magenta, azul, verde, purpura, naranja)
C = dict(mag='#e0149a', blue='#1010e0', green='#5bc41a', purp='#a64ee0', orange='#f28c1c', red='#d62728', cyan='#17a2c4', gray='#666666', k='k')
COL = [C['mag'], C['blue'], C['green'], C['purp'], C['orange'], C['cyan'], C['red']]
CM_MAP = 'viridis'
# mapa divergente para log10 g2 (azul=antibunching fuerte, blanco=1, rojo=bunching)
CM_G2 = LinearSegmentedColormap.from_list('g2', ['#08306b', '#2171b5', '#9ecae1', '#f7f7f7', '#fdae6b', '#d94801', '#7f2704'])

def setup(fs=8.5):
    plt.rcParams.update({
        'font.family': 'cmr10', 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
        'font.size': fs, 'axes.labelsize': fs + 1, 'xtick.labelsize': fs, 'ytick.labelsize': fs, 'legend.fontsize': fs - 1,
        'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'lines.linewidth': 1.3, 'lines.markersize': 3.6, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 400,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'axes.grid': False, 'pdf.fonttype': 42,
    })
setup()

def panel(ax, label, loc='tl', fs=8):
    """etiqueta (a) en cuadro negro con texto blanco, como en las referencias"""
    x, y, ha, va = {'tl': (0.02, 0.97, 'left', 'top'), 'tr': (0.98, 0.97, 'right', 'top'), 'bl': (0.02, 0.03, 'left', 'bottom')}[loc]
    ax.text(x, y, f'({label})', transform=ax.transAxes, ha=ha, va=va, color='w', fontsize=fs, fontweight='bold', zorder=20,
            bbox=dict(boxstyle='square,pad=0.18', fc='k', ec='k'))

def vline(ax, x, text=None, ypos=0.08, **kw):
    ax.axvline(x, color='k', ls='--', lw=0.9, zorder=3)
    if text: ax.text(x, ypos, text, transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=8, bbox=dict(fc='w', ec='none', pad=0.5))

def cbar(fig, im, ax, label, size='3.5%', pad=0.06, **kw):
    """barra de color pegada al eje (sin solapar con paneles vecinos)"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes('right', size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax, **kw); cb.set_label(label); cb.ax.tick_params(direction='in', length=2.5)
    return cb

# ---------------- dibujo de esquemas ----------------
def bracket_cavity(ax, x, y, w=0.9, h=1.6, color=C['blue'], lw=2.4):
    """cavidad dibujada como par de llaves [ ] como en las referencias"""
    t = 0.12
    for sgn, xx in ((1, x), (-1, x + w)):
        ax.plot([xx + sgn * t, xx, xx, xx + sgn * t], [y + h / 2, y + h / 2, y - h / 2, y - h / 2], color=color, lw=lw, solid_capstyle='butt')
        ax.plot([xx, xx], [y - h / 2, y + h / 2], color=color, lw=lw)

def sphere(ax, x, y, r=0.7, c0=(0.55, 0.75, 1.0), c1=(0.1, 0.3, 0.8)):
    """esfera con degradado (atomo / punto cuantico)"""
    n = 120; yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]; rr = np.hypot(xx, yy)
    f = np.clip(np.hypot(xx + 0.35, yy - 0.35) / 1.6, 0, 1)
    img = np.zeros((n, n, 4))
    for k in range(3): img[..., k] = c0[k] * (1 - f) + c1[k] * f
    img[..., 3] = (rr <= 1).astype(float)
    ax.imshow(img, extent=(x - r, x + r, y - r, y + r), origin='lower', zorder=5, interpolation='bilinear')

def wavy(ax, p0, p1, amp=0.07, n=9, color=C['orange'], lw=1.5):
    p0, p1 = np.array(p0, float), np.array(p1, float); d = p1 - p0; L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
    s = np.linspace(0, 1, 400); off = amp * np.sin(2 * np.pi * n * s) * np.sin(np.pi * s) ** 0.5
    pts = p0[None] + np.outer(s, d) + np.outer(off, nrm); ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=4)

def straight(ax, p0, p1, color=C['orange'], lw=1.3, ls='-'):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls, zorder=4)

def arrow(ax, p0, p1, color='k', lw=1.4, style='-|>', ms=8, ls='-', both=False, rad=0.0):
    st = '<|-|>' if both else style
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=st, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=6, connectionstyle=f'arc3,rad={rad}'))

def levels(ax, energies, x0=0.0, w=0.8, labels=None, colors=None, lw=1.8, fs=8):
    """niveles como barras horizontales; devuelve centros x,y para dibujar flechas"""
    pos = []
    for i, e in enumerate(energies):
        col = colors[i] if colors else 'k'
        ax.plot([x0, x0 + w], [e, e], color=col, lw=lw)
        if labels: ax.text(x0 - 0.06, e, labels[i], ha='right', va='center', fontsize=fs)
        pos.append((x0 + w / 2, e))
    return pos

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')


def finish(fig):
    """quita las etiquetas menores de ejes logaritmicos (evita 2x10^-2, 3x10^-2 ... apiladas)"""
    from matplotlib.ticker import NullFormatter
    for ax in fig.axes:
        if ax.get_xscale() == 'log': ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale() == 'log': ax.yaxis.set_minor_formatter(NullFormatter())
