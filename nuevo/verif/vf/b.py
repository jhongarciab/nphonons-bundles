import sys; sys.path.insert(0,'/home/user/nphonons-bundles/nuevo/bundles/src')
import numpy as np, bundles_core as bc, itertools
from multiprocessing import Pool
n=2
def Jbest(a):
    l,g,nk,nO=a
    kaps=np.geomspace(1e-4,0.1,nk);Oms=np.geomspace(0.01,0.3,nO);bj=np.nan
    for k in kaps:
        for O in Oms:
            if bc.omega_eff(n,l,O)/bc.gamma_c(n,0,k,g,2*g)>0.5: continue
            if bc.P_closed(l,O,k,g,2*g,n)<0.9: continue
            J=bc.flux_closed(l,O,k,g,2*g,n)
            if np.isnan(bj) or J>bj: bj=J
    return bj
if __name__=='__main__':
    pts=[(0.0124,1e-5),(0.0124,1.5e-4),(0.0168,6.7e-4),(0.0308,3.1e-3),(0.0308,6e-3),(0.0565,4.5e-3),(0.1,6.6e-3),(0.1,1e-2),(0.14,6.6e-3),(0.0091,1e-5),(0.0067,1e-5),(0.005,1e-5)]
    with Pool(2) as p:
        a=p.map(Jbest,[q+(22,12) for q in pts]); b=p.map(Jbest,[q+(80,40) for q in pts])
    for q,x,y in zip(pts,a,b): print(q,x,y)
