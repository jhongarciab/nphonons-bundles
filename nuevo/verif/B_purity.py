import os; os.environ['OMP_NUM_THREADS']='2'
import numpy as np, itertools, json
from lib import *
import scipy.linalg as sl
gam0,gph0=2e-4,4e-4

def cyc(lam,Om,kap,gam,gph,n,variant='pol_dressed',Nb=12,M=40):
    m=Model(lam,Om,kap,gam,gph,Nb,variant); e,_,_=eps_n(n,lam,Om); dL=lam**2-n+e
    H=m.Hfull(dL); return cycle_dist(H,m.cops,0,1,m.dim,M=M), m, dL

def episodes(m,dL,tau,kmax=40):
    """clustering exacto: episodio = saltos de fonon consecutivos con huecos < tau. Devuelve P(tamano=k) por episodio."""
    N=m.dim; H=m.Hfull(dL); L=qt.liouvillian(H,m.cops).full()
    Jb=qt.sprepost(m.Lphon,m.Lphon.dag()).full(); L0=L-Jb
    E=sl.expm(L0*tau); I=np.eye(N*N); inv=np.linalg.inv(L0)
    Kc=Jb@inv@(E-I); Kg=-Jb@inv@E
    tr=tracevec(N); rho=np.zeros(N*N,complex); rho[0]=1
    for it in range(30):
        a=rho; S=np.zeros(N*N,complex); P=np.zeros(kmax)
        for k in range(1,kmax):
            e=Kg@a; P[k]=np.real(tr@e); S+=e; a=Kc@a
        new=S/(tr@S)
        if np.linalg.norm(new-rho)<1e-12: rho=new; break
        rho=new
    P/=P.sum(); return P

if __name__=="__main__":
    out={}
    print("== Ciclos radiativos (base): P(N) para n=2,3, estados base")
    for n in (2,3):
        P,m,dL=cyc(0.14,0.1,0.003,gam0,gph0,n)
        ks=np.arange(len(P)); print(n,"dL=",round(dL,5),"P0..5=",np.round(P[:6],4),"mean=",(ks*P).sum(),"e^-l2=",np.exp(-0.14**2),"suma",P.sum())
        print("   flujo/Jrad check: J=",m.flux(dL))
    print("== Barrido max P_n (ciclos) sobre kap,Om,gam; lam=0.14, y lam=0.1,0.2 ; n=2")
    best={}
    for n in (2,3):
      for lam in (0.1,0.14,0.2):
        res=[]
        for kap in (3e-4,1e-3,3e-3,1e-2,3e-2):
          for Om in (0.03,0.05,0.1,0.2,0.3):
            for gam in (2e-5,2e-4,2e-3):
                P,_,_=cyc(lam,Om,kap,gam,4e-4,n,M=24); res.append((P[n],P[n]+0,kap,Om,gam))
        res.sort(reverse=True); b=res[0]
        print(f"n={n} lam={lam}: max P_n(ciclo)={b[0]:.5f} en kap={b[2]},Om={b[3]},gam={b[4]}; e^-l2={np.exp(-lam**2):.5f}; (ratio {b[0]/np.exp(-lam**2):.4f})",flush=True)
        out[f"cyc_{n}_{lam}"]=b
    print("== Episodios (clustering de huecos tau): base, n=2")
    for kap in (0.003,):
      m=Model(0.14,0.1,kap,gam0,gph0,12,'pol_dressed'); dL=0.14**2-2+eps_n(2,0.14,0.1)[0]
      for tf in (0.5,1,2,3,5,8,12):
        P=episodes(m,dL,tf/kap)
        print(f"tau={tf}/kap: P1={P[1]:.4f} P2={P[2]:.4f} P3={P[3]:.4f} P4={P[4]:.4f} | frac fonones en episodios n=2: {2*P[2]/sum(k*P[k] for k in range(len(P))):.4f}  e^-l2={np.exp(-0.14**2):.4f}",flush=True)
