"""Estilo común de las figuras (PRA): serif 8–9 pt, paleta Okabe–Ito (apta para daltonismo)."""
import matplotlib as mpl

COL1, COL2 = 3.375, 7.0          # ancho de columna PRA (in)
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']


def aplicar():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'cm', 'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def etiqueta(ax, txt):
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, va='top', ha='left', fontsize=9)
