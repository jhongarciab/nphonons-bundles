"""Mapas estaticos desde |f,0> ideal (T1 escalera): P2(km, off) y N=3."""
import numpy as np, time
from simP import *
KM=np.geomspace(0.5,20,13); OFF=np.arange(-12,12.1,1.0)
out={}
for name,T1 in [('T1_1.14',1.14),('T1_10',10.0)]:
    P5=np.zeros((len(KM),len(OFF),4)); P30=np.zeros_like(P5)
    for i,km in enumerate(KM):
        for j,of in enumerate(OFF):
            r=run_cycle(Cfg(T1=T1,km=km*TP,off=of*TP),None,Tmax=3.0,dT=0.1); P5[i,j]=r['P'][5]; P30[i,j]=r['P'][30]
    out[name+'_P_Tw0.5']=P5; out[name+'_P_Tw3']=P30; print(name,'best Tw0.5',P5[...,2].max(),np.unravel_index(P5[...,2].argmax(),P5[...,2].shape),'Tw3',P30[...,2].max(),flush=True)
np.savez('../datos/mapa_estatico_N2.npz',KM=KM,OFF=OFF,**out)
# N=3
KM3=np.geomspace(0.3,10,10); OFF3=np.arange(-15,15.1,2.5); o3={}
c3=Cfg(order=3,nq=5,N=7,cap=5,T1=1.14,km=2*TP)
sy=Sys(c3); print('N=3 Dq0/2pi',sy.Dq0/TP,'gap/2pi',sy.gap/TP,'V3 teorico 6g^3/a^2/2pi',6*(15.2)**3/138**2,flush=True)
for name,T1 in [('T1_1.14',1.14),('T1_10',10.0),('T1_inf',np.inf)]:
    P=np.zeros((len(KM3),len(OFF3),6)); 
    for i,km in enumerate(KM3):
        for j,of in enumerate(OFF3):
            r=run_cycle(replace(c3,T1=T1,km=km*TP,off=of*TP),None,Tmax=3.0,dT=0.1); pass
            P[i,j]=r['Praw'][30,:6]
    o3[name]=P
np.savez('../datos/mapa_estatico_N3.npz',KM3=KM3,OFF3=OFF3,**o3)
