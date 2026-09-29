# -*- coding: utf-8 -*-
"""Tarea 27: brecha robusta = Re(lambda) del 5o autovalor por Re(lambda) ascendente.
Uso: python tarea27_worker.py <Gamma2/kappa> <alpha2> <outfile>   (delta_m,Delta_q)=(0.048,0.144)"""
import sys, numpy as np
import modelo_comun as mc

G2, alpha2, outfile = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
gzs = mc.gz_scale_from_Gamma2(G2)
DQ = 0.144
U, p = mc.full_propagator(gzs, alpha2, 0.048, DQ)
Nb = p['Nb']
ev, evec = U.eigenstates()
mf = mc.modos_ordenados(ev, evec, Nb, True, mc.T_r)
L, _ = mc.effective_liouvillian(gzs, alpha2, DQ)
ee, eve = L.eigenstates(sparse=False)
me = mc.modos_ordenados(ee, eve, Nb, False, None)

def resumen(m, etiqueta):
    lg = m[1:4]
    kpf = max(lg, key=lambda x: x['ov_P'])       # phase-flip = el de mayor overlap con paridad
    ok = (kpf['ov_P'] > kpf['ov_a'] and kpf['ov_P'] > kpf['ov_n']
          and all(x['ov_a'] > x['ov_P'] for x in lg if x is not kpf))
    print(f"[{etiqueta}] G2={p['Gamma2']:.3f} overlaps k=0..5 (P,n,a | Re lam, Im lam):")
    for x in m[:6]:
        print(f"   k={x['k']} P={x['ov_P']:.3f} n={x['ov_n']:.3f} a={x['ov_a']:.3f} | {x['lam'].real:.4e} {x['lam'].imag:+.2e}")
    print(f"   logicos(k=1..3)={'OK' if ok else 'NO CUMPLEN: P=1,a=2'}  gap(k=4)={m[4]['lam'].real:.4e}")
    return dict(gap=m[4]['lam'].real, im_gap=m[4]['lam'].imag, gamma_pf=kpf['lam'].real,
                logicos_ok=ok, lam=np.array([x['lam'] for x in m]),
                ov=np.array([[x['ov_P'], x['ov_n'], x['ov_a']] for x in m]))
rf, re_ = resumen(mf, "COMPLETO"), resumen(me, "EFECTIVO")
np.savez(outfile, Gamma2=p['Gamma2'], alpha2=alpha2, **{f"{k}_full": v for k, v in rf.items()},
         **{f"{k}_eff": v for k, v in re_.items()})
