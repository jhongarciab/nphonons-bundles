"""Utilidades comunes de los scripts de datos: puntos exactos en paralelo (max 2 procesos) y cache npz."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from multiprocessing import Pool
from bundles_core import *

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
BASE = dict(lam=0.14, Om=0.1, kap=0.003, gam=2e-4, gph=4e-4)

def _pt(a):
    n, lam, Om, kap, gam, gph, jump, kind, dL = a
    r = exact_point(n, lam, Om, kap, gam, gph, jump=jump, kind=kind, dL=dL, K=2 * n + 6)
    return r['P'], r['flux'], r['dist'], r['T']

def points(args, procs=2):
    """args: lista de tuplas (n,lam,Om,kap,gam,gph,jump,kind,dL|None). Devuelve P, flux, dist, T (arrays)."""
    with Pool(procs) as p:
        out = p.map(_pt, args, chunksize=8)
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]), np.array([o[3] for o in out]))

def save(name, meta, **arrs):
    np.savez(os.path.join(DATA, name), meta=json.dumps(meta), **arrs)
    print('guardado', name, {k: np.shape(v) for k, v in arrs.items()}, flush=True)

def cached(name):
    return os.path.exists(os.path.join(DATA, name)) and '--force' not in sys.argv
