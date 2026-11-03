"""Serie temporal: QuTiP mesolve (ensemble) y mcsolve (trayectorias con clics) del escenario PUB caso B, km=10.8 MHz."""
import numpy as np, qutip as qt, pickle, time
from simP import *
cal=pickle.load(open('../datos/calib_tp24.pkl','rb')); p=cal[False]
PUB=dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05,kint=0.37*TP)
sy=Sys(Cfg(km=10.815*TP,off=-2*TP,cap=99,**PUB)); c=sy.cfg; dims=[[c.nq,c.N],[c.nq,c.N]]; Q=lambda m: qt.Qobj(m,dims=dims)
gf,cfs=pulse_coefs(sy,p)
H=[Q(sy.H0),[Q(sy.Hg),lambda t: gf(t)]]
for cf in cfs:
    H.append([Q(sy.s),lambda t,cf=cf: cf(t)]); H.append([Q(sy.sd),lambda t,cf=cf: np.conj(cf(t))])
cops=[Q(x) for x in sy.counted+sy.unc]; rho0=Q(sy.rho_init()); Tend=2*p.tp+0.8; tl=np.arange(0,Tend,0.002)
projs=[Q(np.kron(np.diag(np.eye(c.nq)[k]),np.eye(c.N))) for k in range(c.nq)]; eo=projs+[Q(sy.nm)]
t0=time.time(); r=qt.mesolve(H,rho0,tl,cops,e_ops=eo,options={'atol':1e-9,'rtol':1e-7,'nsteps':100000,'max_step':0.002}); print('mesolve',time.time()-t0,flush=True)
ens=np.array(r.expect); kd=sy.kd; clicks=np.concatenate([[0],np.cumsum(0.5*(ens[-1][1:]+ens[-1][:-1])*np.diff(tl))])*kd
psi0=None
# trayectorias: estado inicial mixto -> muestreo de kets (mcsolve admite dm? usamos estado inicial puro |g,0> y mezcla aparte)
tr=[]; t0=time.time(); ntr=0
rng=np.random.default_rng(1)
pe,nth=c.pe,c.nth; pm=(nth/(1+nth))**np.arange(c.N); pm/=pm.sum()
res_all=[]
for it in range(400):
    q0=1 if rng.random()<pe else 0; n0=rng.choice(c.N,p=pm)
    ket=qt.tensor(qt.basis(c.nq,q0),qt.basis(c.N,n0))
    rr=qt.mcsolve(qt.QobjEvo(H) if False else H,ket,tl,cops,e_ops=eo,ntraj=4,options={'keep_runs_results':True,'progress_bar':False,'map':'serial','atol':1e-8,'rtol':1e-6,'max_step':0.002,'nsteps':100000,'store_states':False})
    for k in range(4):
        ct=np.array(rr.col_times[k]); cw=np.array(rr.col_which[k]); res_all.append((ct,cw,np.array([rr.runs_expect[j][k] for j in range(len(eo))])))
    if it==0: print('t/4traj',time.time()-t0,flush=True)
pickle.dump(dict(tl=tl,ens=ens,clicks_cum=clicks,tp=p.tp,traj=res_all),open('../datos/serie_temporal.pkl','wb'))
# estadistica de clics contados (op 0) en ventana tras pulsos, T_w=0.5
tw0=2*p.tp; cnt=[np.sum((ct>tw0)&(ct<=tw0+0.5)&(cw==0)) for ct,cw,_ in res_all]
h=np.bincount(cnt,minlength=5)/len(cnt); print('N traj',len(cnt),'P(m) MC',h[:5])
out=run_cycle(Cfg(km=10.815*TP,off=-2*TP,**PUB),p,Tmax=1.0,dT=0.1)['P'][5]; print('P(m) exacto',out)
