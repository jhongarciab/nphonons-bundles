import numpy as np
from D import *
G=1.0; km=np.sqrt(2); N=7; a,sm=sp_ops(N); tc=5/km; x=km*tc
Hef=G*(sm.T@a@a+sm@a.T@a.T); psi0=np.zeros(2*N,complex); psi0[N]=1
rng=np.random.default_rng(11); kq,W=1.4,0.05; B=1000;T=1500.;t0=30.
ev=run(Hef,[np.sqrt(km)*a,np.sqrt(kq)*sm,np.sqrt(W)*sm.T],psi0,B,T,0.01,rng); sz=clusters(ev,B,tc,t0,T)
n=len(sz);p2=np.mean(sz==2);R=n/(B*(T-t0));print(f"kq={kq} W={W}: clusters={n} P2={p2:.4f}+-{np.sqrt(p2*(1-p2)/n):.4f} q_impl={1-p2/(1-2*np.exp(-x)):.4f} R*tc={R*tc:.3f}")
