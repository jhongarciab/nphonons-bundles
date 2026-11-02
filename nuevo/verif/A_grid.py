import os; os.environ['OMP_NUM_THREADS']='2'
import numpy as np, itertools, json, sys
from lib import *
gam,gph=2e-4,4e-4
pts=[(n,Om,k,l) for n in (2,3) for Om in (0.07,0.13) for k in (0.0015,0.004) for l in (0.12,0.17)]
pts+=[(2,0.2,0.01,0.2),(3,0.2,0.01,0.2),(2,0.05,0.001,0.1),(3,0.05,0.001,0.1),(2,0.1,0.003,0.14),(3,0.1,0.003,0.14)]
rows=[]
print("n Om kap lam | dpeak_loc dpeak_dr  pred(l2-n+eps) | err_loc err_dr (x1e-4) | J_loc J_dr  J_H2 | rel_loc rel_dr | 2Om2/n")
for (n,Om,k,l) in pts:
    Nb=n+9
    ml=Model(l,Om,k,gam,gph,Nb+2,'lab'); md=Model(l,Om,k,gam,gph,Nb+2,'pol_dressed')
    e,_,_=eps_n(n,l,Om); pred=l*l-n+e
    pl,Jl=peak(ml,pred,0.05,41); pd,Jd=peak(md,pred,0.05,41)
    d=np.exp(-l*l/2)*l**n/np.sqrt(factorial(n)); G0=4*Om**2*d**2/(n*k+gam+gph)
    JH=(n+l*l)/(1/G0+1/gam)
    r=dict(n=n,Om=Om,kap=k,lam=l,pk_lab=pl,pk_dr=pd,pred=pred,eps=e,J_lab=Jl,J_dr=Jd,J_H2=JH,pred0=l*l-n+2*Om**2/n)
    rows.append(r)
    print(f"{n} {Om} {k} {l} | {pl:.5f} {pd:.5f} {pred:.5f} | {1e4*(pl-pred):.1f} {1e4*(pd-pred):.1f} | {Jl:.3e} {Jd:.3e} {JH:.3e} | {Jl/JH-1:+.3f} {Jd/JH-1:+.3f} | err0={1e4*(pl-r['pred0']):.1f}",flush=True)
json.dump(rows,open('A_grid.json','w'))
