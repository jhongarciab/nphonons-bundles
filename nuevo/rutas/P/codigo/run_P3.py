"""Presupuesto de error, barridos en T1, n_th, Tphi, p_e, escalera, flujo. Punto de operacion: caso B (qubit estacionado en pulsos), km=10.8 MHz, off=-2 MHz."""
import numpy as np, pickle, json
from simP import *
cal=pickle.load(open('../datos/calib_tp24.pkl','rb'))
PUB=dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05,kint=0.37*TP)
IMP=dict(T1=10.0,Tphi=3.0,nth=0.005,pe=0.01,kint=0.05*TP)
IDEAL=dict(T1=np.inf,Tphi=np.inf,nth=0.0,pe=0.0,kint=0.0)
KM0=10.815; OFF0=-2.0; IW=5   # indice T_w=0.5 us con dT=0.1
def P2(kw,gate=False,pulses=True,km=KM0,off=OFF0,Tmax=1.0,full=False):
    c=Cfg(km=km*TP,off=off*TP,**{**IDEAL,**kw}); r=run_cycle(c,cal[gate] if pulses else None,Tmax=Tmax,dT=0.1)
    return r if full else r['P'][IW]
R={}
# 1. cascada acumulada
for gate,nm_ in [(False,'B'),(True,'A')]:
    km=KM0 if not gate else 5.848; off=OFF0 if not gate else 0.0
    steps=[('ideal |f,0>, solo km (fuga virtual |e,1>)',dict(),False),('+ pulsos reales (g inf)',dict(),True),('+ T1 escalera',dict(T1=1.14),True),
           ('+ desfase Tphi=0.30us',dict(T1=1.14,Tphi=0.30),True),('+ termico (n_th=0.03, p_e=5%)',dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05),True),
           ('+ perdida intrinseca k_int=0.37 MHz',PUB,True)]
    R['cascada_'+nm_]=[(n,*P2(k,gate,pu,km,off)) for n,k,pu in steps]
    full=PUB; rem=[('sin T1 (T1=inf)',{**full,'T1':np.inf}),('sin desfase',{**full,'Tphi':np.inf}),('sin termico (nth=pe=0)',{**full,'nth':0.0,'pe':0.0}),('sin pe',{**full,'pe':0.0}),('sin k_int',{**full,'kint':0.0})]
    R['quitar_'+nm_]=[('completo',*P2(full,gate,True,km,off))]+[(n,*P2(k,gate,True,km,off)) for n,k in rem]+[('sin errores de pulso (|f,0> ideal)',*P2(full,gate,False,km,off))]
# 2. T1
T1s=[1.14,5,10,50]
R['T1_full']=[(t,*P2({**PUB,'T1':t})) for t in T1s]
R['T1_IMPrest']=[(t,*P2({**IMP,'T1':t})) for t in T1s]
R['T1_only']=[(t,*P2({'T1':t})) for t in T1s]
R['T1_onlyidealprep']=[(t,*P2({'T1':t},pulses=False)) for t in T1s]
R['T1_phi_inf_rest_PUB']=[(t,*P2({**PUB,'T1':t,'Tphi':np.inf})) for t in T1s]
# 3. termico
nths=[0,0.01,0.03,0.1,0.3]
R['nth_cold']=[(n,*P2({**PUB,'nth':n})) for n in nths]
R['nth_hot']=[(n,*P2({**PUB,'nth':n,'ng':n})) for n in nths]
R['nth_cold_T1_10']=[(n,*P2({**IMP,'nth':n,'kint':0.37*TP})) for n in nths]
R['nth_hot_T1_10']=[(n,*P2({**IMP,'nth':n,'ng':n,'kint':0.37*TP})) for n in nths]
R['pe']=[(p,*P2({**PUB,'pe':p})) for p in (0,0.01,0.03,0.05,0.10)]
R['Tphi']=[(t,*P2({'T1':1.14,'Tphi':t})) for t in (0.1,0.3,1,3,10,np.inf)]
R['Tphi_T1_10']=[(t,*P2({'T1':10.0,'Tphi':t})) for t in (0.1,0.3,1,3,10,np.inf)]
# 5. escalera
R['escalera']=[('escalera (f->e a 2 g_q)',*P2({'T1':1.14})),('sin escalera (f->e a g_q)',*P2({'T1':1.14,'r_fe':1.0})),
               ('escalera + directo f->g (g_q)',*P2({'T1':1.14,'g20':1.0})),('solo f->e nula (r=0) , sin T1 en e',*P2({'T1':1.14,'r_fe':0.0}))]
# 6. flujo: T_w(0.5) y T_w optimo
def flux(P2v,T1,Tw,reset): return P2v/(0.048+Tw+(5*T1 if reset=='pasivo' else 0.1))
R['flujo']=[]
for nm_,kw,gate,km,off in [('ideal (prep ideal, T1 inf)',IDEAL,False,KM0,OFF0),('PUB B',PUB,False,KM0,OFF0),('PUB B km=20',PUB,False,20.0,-4.0),('PUB A',PUB,True,5.848,0.0),('IMP B',IMP,False,KM0,OFF0),('IMP B T1=50',{**IMP,'T1':50.0,'Tphi':10.0},False,KM0,OFF0)]:
    r=P2(kw,gate,nm_!='ideal (prep ideal, T1 inf)',km,off,Tmax=2.0,full=True); Pc=r['P']
    for iw in (2,5,10):
        Tw=r['T'][iw]; T1=kw['T1'] if np.isfinite(kw['T1']) else 1e9
        R['flujo'].append((nm_,Tw,*Pc[iw],flux(Pc[iw,2],kw['T1'],Tw,'pasivo') if np.isfinite(kw['T1']) else np.nan,flux(Pc[iw,2],kw['T1'],Tw,'activo')))
pickle.dump(R,open('../datos/presupuesto.pkl','wb'))
for k,v in R.items():
    print('==',k)
    for row in v: print('  ',[ (round(x,4) if isinstance(x,(float,np.floating)) else x) for x in row])
