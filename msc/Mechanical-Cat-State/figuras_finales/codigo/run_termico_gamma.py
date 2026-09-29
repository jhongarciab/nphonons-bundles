"""Figura térmica, panel (b) nuevo: rejilla (γ/κ, x) en la isla (κ₂/κ = 0.25, g_z/κ = 14), filtro y plano, N = 22."""
import sys
import numpy as np
from multiprocessing import Pool
import calc_termico as CT
import run_termico as RT

GAMS = np.geomspace(1e-6, 1e-3, 13)


def tarea(t):
    x, gam, fil = t
    return float(CT.punto(x, 0.25, 14.0, fil, gam, 22)['eta'])


if __name__ == '__main__':
    tareas = [(x, g, f) for f in (1, 0) for g in GAMS for x in RT.XS]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        p.map(tarea, tareas, chunksize=4)
    print(len(tareas), 'puntos')
