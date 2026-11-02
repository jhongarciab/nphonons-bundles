import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from C_micro import *
lam,Om,n=0.14,0.1,2
dL=lam**2-n+eps_n(n,lam,Om)[0]
print("convergencia (kap=0.003,kc=0.05):")
for Nb,Nc in ((8,3),(9,3),(11,3),(9,4)):
    P,Jc,Jr=micro(lam,Om,0.003,0.05,dL,Nb,Nc); print(Nb,Nc,"mean=%.5f P2=%.5f P0=%.5f"%(stats(P),P[2],P[0]),flush=True)
def run(kap,kcs):
    Pd,md,_=cyc(lam,Om,kap,gam,gph,n,variant='pol_dressed',Nb=12,M=32)
    Pl,ml,_=cyc(lam,Om,kap,gam,gph,n,variant='lab',Nb=11,M=32)
    ms=[];p2=[]
    for kc in kcs:
        P,_,_=micro(lam,Om,kap,kc,dL,9,3); ms.append(stats(P)); p2.append(P[2])
    kc=np.array(kcs); A=np.vstack([np.ones_like(kc),kc**2]).T
    m0=np.linalg.lstsq(A,ms,rcond=None)[0]; q0=np.linalg.lstsq(A,p2,rcond=None)[0]
    print(f"kap={kap}: kcs={kcs} means={np.round(ms,4)} P2={np.round(p2,4)}\n   extrap(kc->0): mean={m0[0]:.4f} (coef kc^2={m0[1]:.2f}) P2={q0[0]:.4f} | vestida mean={stats(Pd):.4f} P2={Pd[2]:.4f} | local mean={stats(Pl):.4f} P2={Pl[2]:.4f}",flush=True)
run(0.003,[0.03,0.05,0.1])
run(0.01,[0.05,0.1,0.15])
run(0.03,[0.1,0.15,0.2])
