import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, sys, json
from lib import *
from B_purity import cyc
gam,gph=2e-4,4e-4

def micro(lam,Om,kap,kc,dL,Nb,Nc,gam=gam,gph=gph,M=32):
    g=np.sqrt(kap*kc)/2            # regla de oro: kap = 4 g^2/kc
    sg=qt.tensor(qt.destroy(2),qt.qeye(Nb),qt.qeye(Nc)); b=qt.tensor(qt.qeye(2),qt.destroy(Nb),qt.qeye(Nc))
    c=qt.tensor(qt.qeye(2),qt.qeye(Nb),qt.destroy(Nc)); n=sg.dag()*sg
    H=dL*n+b.dag()*b+lam*n*(b+b.dag())+Om*(sg+sg.dag())+c.dag()*c+g*(b+b.dag())*(c+c.dag())
    cops=[np.sqrt(kc)*c,np.sqrt(gam)*sg,np.sqrt(gph)*n]
    dimH=2*Nb*Nc
    P=cycle_dist(H,cops,0,1,dimH,M=M)
    # tasa de saltos c y de saltos radiativos en estacionario
    L=sp_(qt.liouvillian(H,cops)).tolil(); L[0,:]=0; L=L.tocsc(); L=L.tolil()
    L[0,::dimH+1]=1; L=L.tocsc(); r=np.zeros(dimH**2,complex); r[0]=1
    v=spla.spsolve(L,r); rho=v.reshape((dimH,dimH),order='F')
    Jc=np.real(np.trace(rho@(cops[0].dag()*cops[0]).full())); Jr=gam*np.real(np.trace(rho@(sg.dag()*sg).full()))
    return P,Jc,Jr

def stats(P): 
    k=np.arange(len(P)); return (k*P).sum()

if __name__=="__main__":
    lam,Om,n=0.14,0.1,2
    dL=lam**2-n+eps_n(n,lam,Om)[0]
    print("dL=",dL)
    cases=[(0.003,0.05),(0.003,0.1),(0.003,0.03),(0.01,0.1),(0.01,0.2),(0.03,0.3)]
    for kap,kc in cases:
        Pl,ml,_=cyc(lam,Om,kap,gam,gph,n,variant='lab',Nb=11,M=32)
        Pd,md,_=cyc(lam,Om,kap,gam,gph,n,variant='pol_dressed',Nb=12,M=32)
        Pm,Jc,Jr=micro(lam,Om,kap,kc,dL,9,3)
        print(f"\nkap={kap} kc={kc}: g={np.sqrt(kap*kc)/2:.3e}  kap_extra_local(lam2*kap/gam)={lam**2*kap/gam:.3f}")
        for name,P in (("local",Pl),("vestida",Pd),("micro",Pm)):
            print(f"  {name:8s} mean={stats(P):.4f} P0..5={np.round(P[:6],4)}")
        print(f"  micro: Jc={Jc:.3e} Jrad={Jr:.3e} Jc/Jrad={Jc/Jr:.4f}   Jlocal={ml.flux(dL):.3e}  Jdress={md.flux(dL):.3e}",flush=True)
