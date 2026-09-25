# -*- coding: utf-8 -*-
"""Tarea 29-bis (original): A (qubit) vs B (buffer armonico Nb=6). Uso: worker <A|B> <alpha2> <kappa/g> <outfile>"""
import sys, numpy as np, modelo_buffer as mb
model, a2, kap, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
L, Nc, dq = mb.build(model, a2, kap)
r = mb.analizar(L, Nc, dq)
np.savez(out, model=model, alpha2=a2, kappa=kap, **{k: v for k, v in r.items()})
print(f"OK {model} a2={a2} k/g={kap:.4f} gap={r['gap']:.4e} Im={r['gap_im']:+.2e} pf={r['gamma_pf']:.4e} bf={r['gamma_bf']:.4e} ok={r['ok']}")
