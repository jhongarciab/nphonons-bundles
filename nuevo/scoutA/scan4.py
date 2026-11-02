import numpy as np,sys
from qdm import *
# mapa (Om, wL-S/2) para gam dado; guarda npz
gam=float(sys.argv[1]); kap=float(sys.argv[2]); eps=float(sys.argv[3]); S=float(sys.argv[4]) if len(sys.argv)>4 else 1.0
J=np.sqrt(S**2-eps**2)/2
Oms=np.array([0.002,0.004,0.007,0.01,0.015,0.02,0.03]); ds=np.linspace(-0.2,0.2,21)
G=np.zeros((len(Oms),len(ds))); N=G.copy(); Q=G.copy(); P1=G.copy(); P2=G.copy()
for i,Om in enumerate(Oms):
    for j,d in enumerate(ds):
        r,_,_=ss(Nb=16,eps=eps,J=J,wL=S/2+d,gam=gam,Om=Om,kap=kap)
        G[i,j],N[i,j],Q[i,j],P1[i,j],P2[i,j]=r['g2'],r['n'],r['Q'],r['P'][1],r['P'][2]
np.savez(f'scan4_g{gam}_k{kap}_e{eps}_S{S}.npz',Oms=Oms,ds=ds,G=G,N=N,Q=Q,P1=P1,P2=P2)
k=np.nanargmin(np.where(N>0.02,G,9)); i,j=np.unravel_index(k,G.shape)
print('gam',gam,'kap',kap,'eps',eps,'S',S,'| min g2 (n>0.02): Om=%.2f d=%.3f g2=%.3f n=%.3f Q=%.3f'%(Oms[i],ds[j],G[i,j],N[i,j],Q[i,j]))
k=np.nanargmin(np.where(N>0.02,Q,9)); i,j=np.unravel_index(k,G.shape)
print('   min Q: Om=%.2f d=%.3f g2=%.3f n=%.3f Q=%.3f P1=%.3f P2=%.3f'%(Oms[i],ds[j],G[i,j],N[i,j],Q[i,j],P1[i,j],P2[i,j]))
k=np.argmax(N); i,j=np.unravel_index(k,G.shape); print('   max n: Om=%.2f d=%.3f n=%.3f g2=%.3f Q=%.3f'%(Oms[i],ds[j],N[i,j],G[i,j],Q[i,j]))
