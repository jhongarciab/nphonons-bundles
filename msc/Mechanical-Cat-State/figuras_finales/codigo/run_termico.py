"""Figura térmica: rejillas (x, parámetro) con el modelo efectivo (calc_termico.punto). Paralelo con multiprocessing.
Rejillas: x ∈ [1, 25] (40 valores, log) × {g_z/κ ∈ [10, 40] con κ₂/κ = 0.25} y × {κ₂/κ ∈ [0.02, 0.4] con g_z/κ = 14};
baño filtrado (κ_f/ω = 0.05) y plano; γ/κ ∈ {2e-5, 2e-4}; N = 22. Curva base en la isla (κ₂/κ = 0.25, g_z/κ = 14)
con N = 22 y N = 24 (convergencia). Uso: python run_termico.py [NPROC]"""
import sys, os
import numpy as np
from multiprocessing import Pool
import calc_termico as CT

XS = np.geomspace(1.0, 25.0, 40)
GZS = np.geomspace(10, 40, 14)
K2S = np.geomspace(0.02, 0.4, 14)


def tarea(t):
    x, k2k, gzk, fil, gam, N = t
    try:
        r = CT.punto(x, k2k, gzk, fil, gam, N)
        return (t, float(r['eta']))
    except Exception as e:
        return (t, repr(e))


if __name__ == '__main__':
    tareas = []
    for fil in (1, 0):
        for gam in (2e-5, 2e-4):
            for x in XS:
                for N in (22, 24):
                    tareas.append((x, 0.25, 14.0, fil, gam, N))          # curva base + convergencia
                for g in GZS:
                    tareas.append((x, 0.25, g, fil, gam, 22))
                for k in K2S:
                    tareas.append((x, k, 14.0, fil, gam, 22))
    tareas = list(dict.fromkeys(tareas))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(tarea, tareas, chunksize=4)
    malos = [r for r in res if isinstance(r[1], str)]
    print(f"{len(res)} puntos, {len(malos)} fallidos")
    for m in malos[:10]:
        print(m)
