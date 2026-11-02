# Mapa (kappa_m/G, Omega/G) de flujo de clicks y pureza de ventana P2, modelo efectivo (validado contra el completo). omega=1, eps=gx/w=0.006, lambda=0.06.
import numpy as np, sys, time
from purity import *
gx,gz=0.006,0.06; G=2*gx*gz
kq=float(sys.argv[1]) if len(sys.argv)>1 else 0.05; x=float(sys.argv[2]) if len(sys.argv)>2 else 5.0
KM=np.geomspace(0.3,30,22); OM=np.geomspace(0.01,2.0,22)
out=np.zeros((len(OM),len(KM),4))
t=time.time()
for j,km in enumerate(KM):
  for i,om in enumerate(OM):
    H,c=eff_model(gx,gz,N=6,Om=om*G,kap_m=km*G,kap_q=kq*G)
    r=cluster_stats(H,c,0,x/(km*G),mmax=5)
    out[i,j]=[r['rate']/G,r['P2'],r['meank'],r['Pk'][0]]
np.savez(f'mapa_kq{kq}_x{x}.npz',KM=KM,OM=OM,out=out,kq=kq,x=x,gx=gx,gz=gz)
print('hecho',time.time()-t)
