"""F14: spatially resolved Markov dephasing for three stacked dots.

Use F13 pulses and golden-rule transition rates. Run on pcjhon.
K_ij is dimensionless and has unit diagonal; Lindblad operators are the
eigenmodes of gamma_phi*K, not a manually chosen common-noise fraction.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from f14_spatial_noise_model import run
from f13_reoptimize import rates

PULSE = (0.1975, 170.0, -0.02, 0.29, 150.0, -0.025)
G1, G2 = 0.35, 0.60
FB = 100
GC, GC3, GS = rates(FB, G1, G2)
RATIOS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float('inf'))


def matrix(kind, ell_over_d):
    if ell_over_d == 0:
        return np.eye(3)
    if np.isinf(ell_over_d):
        return np.ones((3, 3))
    ii = np.arange(3)
    dist = np.abs(ii[:, None] - ii[None, :]) / ell_over_d
    if kind == 'gaussian': return np.exp(-dist**2 / 2)
    if kind == 'exponential': return np.exp(-dist)
    raise ValueError(kind)


def job(a):
    kind, ratio, wait = a
    cov = matrix(kind, ratio)
    r = run(PULSE, wait, kappa=0.001, gph=4e-4,
            gc=GC, gc3=GC3, gs=GS, gaps=(-G1, -G2), cov=cov)
    return dict(kernel=kind, ell_over_d=ratio, wait=wait,
                covariance=cov.tolist(), eigenvalues=np.linalg.eigvalsh(cov).tolist(),
                F=r['F'], success=r['s'], F_herald=r['F_h'],
                passive_F=r['pas_F'], passive_success=r['pas_s'])


if __name__ == '__main__':
    jobs = [(kind, ratio, wait) for kind in ('gaussian', 'exponential')
            for ratio in RATIOS for wait in (2000, 8000)]
    with Pool(8) as pool:
        rows = pool.map(job, jobs)
    dest = Path(__file__).resolve().parents[1] / 'data' / 'f14_spatial_covariance.json'
    dest.write_text(json.dumps(dict(parameters=dict(g1=G1, g2=G2, fb=FB,
                         gamma_phi=4e-4, rates=[GC, GC3, GS]), rows=rows), indent=2))
    for r in rows:
        print(r['kernel'], r['ell_over_d'], r['wait'],
              f"F={r['F']:.3f} s={r['success']:.3f} Fh={r['F_herald']:.3f}",
              flush=True)
