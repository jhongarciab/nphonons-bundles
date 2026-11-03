import numpy as np, time
from D import *
G=1.0; km=np.sqrt(2); N=6; a,sm=sp_ops(N); tc=5/km; x=km*tc
Hef=G*(sm.T@a@a+sm@a.T@a.T); psi0=np.zeros(2*N,complex); psi0[N]=1
rng=np.random.default_rng(23); eta=lambda kq:1/(1+(kq/4)*(km+kq/2+2/km))
for kq in (0.05,1.4):
    for Om in (0.05,0.15):
        W=4*Om**2/(kq+4*G*G/km); B=1000; T=min(30000.,2.6e4/(B*eta(kq)*W)*1.0*B/B); T=2.4e4/(B*eta(kq)*W)*1.0; T=max(T,300.); t0=30.; tt=time.time()
        ev=run(Hef,[np.sqrt(km)*a,np.sqrt(kq)*sm,np.sqrt(W)*sm.T],psi0,B,T,0.02,rng); sz=clusters(ev,B,tc,t0,T)
        n=len(sz);p2=np.mean(sz==2);R=n/(B*(T-t0))
        print(f"kq={kq} Om={Om} W=4Om^2/(kq+4G^2/km)={W:.5f}: clusters={n} P2={p2:.4f}+-{np.sqrt(p2*(1-p2)/n):.4f} P4+={np.mean(sz>=4):.4f} q_impl={1-p2/(1-2*np.exp(-x)):.4f} R*tc={R*tc:.4f} R/(eta W)={R/(eta(kq)*W):.3f} ({time.time()-tt:.0f}s)",flush=True)
