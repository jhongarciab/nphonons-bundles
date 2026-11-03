"""Extras: km>20 MHz, cascada asintotica (T_w=3us) vs T_w=0.5, promedio cuasi-estatico, n_th(T), figuras npz."""
import numpy as np, pickle
from simP import *
cal=pickle.load(open('../datos/calib_tp24.pkl','rb'))
PUB=dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05,kint=0.37*TP); IMP=dict(T1=10.0,Tphi=3.0,nth=0.005,pe=0.01,kint=0.05*TP)
IDE=dict(T1=np.inf,Tphi=np.inf,nth=0.0,pe=0.0,kint=0.0)
def run(kw,km=10.815,off=-2.0,gate=False,pulses=True,Tmax=3.0):
    return run_cycle(Cfg(km=km*TP,off=off*TP,**{**IDE,**kw}),cal[gate] if pulses else None,Tmax=Tmax)['P']
out={}
# 1. km mas alto
out['km_ext']=[(sn,km,*run(sc,km,-2.0)[5],*run(sc,km,-2.0)[10]) for sn,sc in (('pub',PUB),('imp',IMP)) for km in (20,30,50,80)]
# 2. cascada a T_w=3 us (asintotica) y a 0.5
steps=[('ideal',IDE,False),('+pulsos',IDE,True),('+T1 1.14',dict(T1=1.14),True),('+Tphi 0.30',dict(T1=1.14,Tphi=0.30),True),('+termico',dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05),True),('+kint',PUB,True)]
out['cascada_asint']=[(n,*run(k,pulses=pu)[5][:3],*run(k,pulses=pu)[10][:3],*run(k,pulses=pu)[30][:3]) for n,k,pu in steps]
# 2b. cascada IMP
steps=[('ideal',IDE,False),('+pulsos',IDE,True),('+T1 10',dict(T1=10.0),True),('+Tphi 3',dict(T1=10.0,Tphi=3.0),True),('+termico',dict(T1=10.0,Tphi=3.0,nth=0.005,pe=0.01),True),('+kint',IMP,True)]
out['cascada_imp']=[(n,*run(k,pulses=pu)[5][:3],*run(k,pulses=pu)[10][:3],*run(k,pulses=pu)[30][:3]) for n,k,pu in steps]
# 3. tolerancia a Delta (cuasi-estatico) para PUB B/IMP B: off scan fino
OFF=np.arange(-12,12.1,2.0); out['OFF']=OFF
for sn,sc in (('pub',PUB),('imp',IMP)): out['off_'+sn]=np.array([run(sc,10.815,of)[5] for of in OFF])
# 4. T1 de |f> separado: escalera con T1 de e distinto (T1_e=1.14 vs f 2x) ya cubierto. Vector T1 sweep con rest ideal en T_w=1
out['T1_sweep_idealrest_Tw1']=[(t,*run(dict(T1=t))[10]) for t in (1.14,5,10,50)]
out['T1_sweep_PUBrest_Tw1']=[(t,*run({**PUB,'T1':t})[10]) for t in (1.14,5,10,50)]
pickle.dump(out,open('../datos/extras.pkl','wb'))
for k,v in out.items():
    if k=='OFF' or k.startswith('off_'): continue
    print('==',k); [print('  ',[round(float(x),4) if not isinstance(x,str) else x for x in row]) for row in v]
print('off_pub',np.round(out['off_pub'][:,2],3)); print('off_imp',np.round(out['off_imp'][:,2],3))
# n_th(T) a 2.257 GHz
h=6.62607015e-34; kB=1.380649e-23
for f in (2.257e9,):
    for T in (7,10,15,20,30,40,50,70): print('T',T,'mK n_th',1/np.expm1(h*f/(kB*T*1e-3)))
