"""Validacion de simP.py contra QuTiP mesolve (modelo completo 4x6, sin truncamiento por cap) con pulsos dependientes del tiempo."""
import numpy as np, qutip as qt, time, json
from simP import *
cfg=Cfg(T1=1.14,Tphi=0.30,nth=0.03,ng=0.03,pe=0.05,kint=0.37*TP,km=8.3*TP)
res={}
for gate in (False,True):
    sy=Sys(replace(cfg,cap=99)); p,_=calibrate(sy,Pulse(tp=0.024,gate=gate),nit=150)
    Hd=pulse_funcs(sy,p); dims=[[sy.cfg.nq,sy.cfg.N],[sy.cfg.nq,sy.cfg.N]]
    Q=lambda m: qt.Qobj(m,dims=dims)
    H=qt.QobjEvo(lambda t: Q(sy.H0+Hd(t)))
    cops=[Q(c) for c in sy.counted+sy.unc]; rho0=Q(sy.rho_init())
    tl=np.linspace(0,2*p.tp+0.5,2*int(round((2*p.tp+0.5)/0.001))//2+1)
    t0=time.time(); r=qt.mesolve(H,rho0,tl,cops,e_ops=[Q(sy.nm), Q(sy.nqo)],options={'atol':1e-9,'rtol':1e-7,'nsteps':100000})
    nm=r.expect[0]; kd=sy.kd; 
    i1=np.argmin(abs(tl-2*p.tp)); mean_clicks_win=kd*np.trapezoid(nm[i1:],tl[i1:])
    # propio
    out=run_cycle(replace(cfg),p,Tmax=0.5,dT=0.1,sy=Sys(replace(cfg,cap=99)),gate_clicks=True); P=out['P'][-1]
    out4=run_cycle(replace(cfg),p,Tmax=0.5,dT=0.1,gate_clicks=True)
    mine_mean=(out['P'][-1]*np.array([0,1,2,3])).sum()   # >=3 aproximado como 3
    # poblacion qubit al fin de pulsos (mesolve) vs propio
    res[str(gate)]=dict(qutip_mean_clicks=float(mean_clicks_win),mine_full_P=out['P'][-1].tolist(),mine_cap4_P=out4['P'][-1].tolist(),mine_mean=float(mine_mean),
                        qutip_mean_nq_end=float(r.expect[1][-1]),mine_pk=out['prep']['pk'].tolist(),qutip_secs=time.time()-t0)
    # media de clics exacta de mine: sum m P(m) con m hasta 11
    import simP
    sy2=out['sy']; Rm=pulse_phase(sy2,p,sy2.rho_init()).sum(0)[None]; T,Pf,_=static_counts(sy2,Rm,0.5,0.1); res[str(gate)]['mine_mean_exact']=float((Pf[-1]*np.arange(12)).sum())
    print(gate,res[str(gate)],flush=True)
json.dump(res,open('../datos/val_qutip.json','w'),indent=1)
