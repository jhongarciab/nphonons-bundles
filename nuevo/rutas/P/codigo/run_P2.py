"""Modelo completo con pulsos: calibracion, barridos en km y desintonia, duracion de pulso/DRAG."""
import numpy as np, time, pickle
from simP import *
PUB=dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05,kint=0.37*TP)
IMP=dict(T1=10.0,Tphi=3.0,nth=0.005,pe=0.01,kint=0.05*TP)
base=Cfg(km=8.3*TP); sy0=Sys(base); cal={}
for gate in (False,True):
    cal[gate],cst=calibrate(sy0,Pulse(tp=0.024,gate=gate),nit=300); print('cal',gate,cst,np.round(cal[gate].x[:2],4),np.round(np.array(cal[gate].x[2:])/TP,2),flush=True)
pickle.dump(cal,open('../datos/calib_tp24.pkl','wb'))
KM=np.geomspace(0.5,20,13); OFF=np.arange(-12,12.1,2.0); out={}
for sname,sc in [('pub',PUB),('imp',IMP)]:
    for gate in (False,True):
        key=f'{sname}_{"B" if not gate else "A"}'; P=np.zeros((len(KM),31,4))
        for i,km in enumerate(KM):
            P[i]=run_cycle(replace(base,km=km*TP,**sc),cal[gate],Tmax=3.0,dT=0.1)['P']
        out[key+'_km']=P; ib=P[:,5,2].argmax(); kbest=KM[ib]
        Po=np.zeros((len(OFF),31,4))
        for j,of in enumerate(OFF):
            Po[j]=run_cycle(replace(base,km=kbest*TP,off=of*TP,**sc),cal[gate],Tmax=3.0,dT=0.1)['P']
        out[key+'_off']=Po; out[key+'_kbest']=kbest
        print(key,'kbest',kbest,'P2(Tw=.5) max km',P[ib,5,2],'off max',Po[:,5,2].max(),'@',OFF[Po[:,5,2].argmax()],flush=True)
np.savez('../datos/barrido_completo.npz',KM=KM,OFF=OFF,**out)
# duracion de pulso y DRAG
res=[]
for tp in (0.010,0.016,0.024,0.040):
    for gate in (False,True):
        for dr in (0.0,1.0):
            p,cst=calibrate(sy0,Pulse(tp=tp,gate=gate,drag=dr),nit=250)
            r=run_cycle(replace(base,**PUB),p,Tmax=2.0,dT=0.1)
            res.append((tp*1e3,int(gate),dr,cst,*r['prep']['pk'],r['P'][5,2],r['P'][5,0],r['P'][5,1])); print('tp',res[-1],flush=True)
np.save('../datos/pulsos_tp_drag.npy',np.array(res))
