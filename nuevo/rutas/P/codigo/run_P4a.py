"""(a) Bloch-Siegert; (b) ME local vs vestida (secular); (c) N=3 estatico y con pulsos."""
import numpy as np, pickle, json
from simP import *
res={}
# (a) Bloch-Siegert: modelo de Rabi completo en marco de laboratorio
w=2257.0*TP; cfg=Cfg(nq=4,N=14,cap=99)
a,s,_,_=raw_ops(4,14)
def labgap(wq,rwa=False):
    Ek=np.array([k*wq+cfg.alpha*k*(k-1)/2 for k in range(4)]); H=np.kron(np.diag(Ek),np.eye(14))+w*np.kron(np.eye(4),np.diag(np.arange(14.)))
    X=(s.conj().T@a+s@a.conj().T) if rwa else (s+s.conj().T)@(a+a.conj().T)
    E,V=np.linalg.eigh(H+cfg.g*X); i1=2*14; i2=2
    wt=np.abs(V[i1])**2+np.abs(V[i2])**2; j=np.argsort(wt)[-2:]; return abs(E[j[0]]-E[j[1]])
from scipy.optimize import minimize_scalar
out={}
for rwa in (True,False):
    r=minimize_scalar(lambda x: labgap(w+x,rwa),bounds=(60*TP,80*TP),method='bounded',options={'xatol':1e-9}); out['RWA' if rwa else 'Rabi']=(r.x/TP,r.fun/2/TP)
res['bloch_siegert']=out; print('BS',out,flush=True)
# (b) local vs vestida
def dressed_P(cfg,Tmax=3.0):
    sy=Sys(cfg); H=sy.H0+sy.Hg; E,Vv=np.linalg.eigh(H); Ae=Vv.conj().T@sy.a@Vv; kap=cfg.km
    om=E[None,:]-E[:,None]; keys=np.round(om,5); groups={}
    for i in range(len(E)):
        for j in range(len(E)): groups.setdefault(keys[i,j],[]).append((i,j))
    C=[]
    for k,lst in groups.items():
        M=np.zeros_like(Ae); 
        for i,j in lst: M[i,j]=Ae[i,j]
        if abs(M).max()>1e-12: C.append(np.sqrt(kap)*Vv@M@Vv.conj().T)
    D=sy.D; I=np.eye(D); pre=lambda A:np.kron(A,I); post=lambda B:np.kron(I,B.T)
    L=sy.LH0+sy.LHg+sum(-0.5*(pre(c.conj().T@c)+post(c.conj().T@c)) for c in C); J=sum(np.kron(c,c.conj()) for c in C)
    R0=np.zeros((D,D),complex); i=sy.ket_index(2,0); R0[i,i]=1
    T,P,_=static_counts(sy,R0[None],Tmax,0.1,L=L,J=J); return summarize(P)[-1],len(C)
loc=[];dr=[]
for km in (0.5,1,2,4,8.3,12,20):
    c=Cfg(T1=np.inf,km=km*TP); Pl=run_cycle(c,None,Tmax=3.0)['P'][-1]; Pd,nc=dressed_P(c); loc.append((km,*Pl)); dr.append((km,*Pd)); print('km',km,'local P2',Pl[2],'vestida P2',Pd[2],'ncanales',nc,flush=True)
res['local']=loc; res['vestida']=dr
# (c) N=3 estatico
c3=Cfg(order=3,nq=5,N=7,cap=5,km=2*TP); sy3=Sys(c3); print('N=3 Dq0/2pi',sy3.Dq0/TP,'V3 gap/2',sy3.gap/2/TP,'6g^3/a^2',6*15.2**3/138**2,flush=True)
KM3=np.geomspace(0.3,10,9); OFF3=np.arange(-10,10.1,2.5); M3={}
for name,T1 in [('T1_1.14',1.14),('T1_10',10.0),('T1_inf',np.inf)]:
    P=np.zeros((len(KM3),len(OFF3),6))
    for i,km in enumerate(KM3):
        for j,of in enumerate(OFF3): P[i,j]=run_cycle(replace(c3,T1=T1,km=km*TP,off=of*TP),None,Tmax=3.0)['Praw'][30,:6]
    M3[name]=P; ib=np.unravel_index(P[...,3].argmax(),P[...,3].shape); print(name,'max P3 (T=3us)',P[...,3].max(),'km',KM3[ib[0]],'off',OFF3[ib[1]],flush=True)
np.savez('../datos/mapa_estatico_N3.npz',KM3=KM3,OFF3=OFF3,**M3)
json.dump(res,open('../datos/dressed_bs.json','w'),indent=1,default=float)
# N=3 con pulsos (g apagado durante pulsos) en el optimo de T1=10 e inf
P=M3['T1_1.14']; ib=np.unravel_index(P[...,3].argmax(),P[...,3].shape); kb=KM3[ib[0]]; ob=OFF3[ib[1]]
cb=replace(c3,km=kb*TP,off=ob*TP); sy=Sys(cb)
p3,cst=calibrate(sy,Pulse(tp=0.024,gate=False),nit=400); print('cal N=3 cost',cst,flush=True)
pk=pops_ket(sy,evolve_ket(sy,p3)); print('pk',pk)
PUB=dict(T1=1.14,Tphi=0.30,nth=0.03,pe=0.05,kint=0.37*TP); IMP=dict(T1=10.0,Tphi=3.0,nth=0.005,pe=0.01,kint=0.05*TP); IDE=dict(T1=np.inf,Tphi=np.inf,nth=0,pe=0,kint=0)
rows=[]
for nm_,kw in [('ideal',IDE),('T1 only 1.14',dict(IDE,T1=1.14)),('T1 only 5',dict(IDE,T1=5.0)),('T1 only 10',dict(IDE,T1=10.0)),('T1 only 50',dict(IDE,T1=50.0)),('PUB',PUB),('IMP',IMP),('IMP T1=50',dict(IMP,T1=50.0,Tphi=10.0))]:
    r=run_cycle(replace(cb,**kw),p3,Tmax=3.0); Pr=r['Praw']
    for iw in (5,10,20,30): rows.append((nm_,r['T'][iw],*Pr[iw,:5])) ; 
    print(nm_,'P3(T=.5,1,2,3)=',[round(Pr[iw,3],4) for iw in (5,10,20,30)],'P2',[round(Pr[iw,2],4) for iw in (5,10,20,30)],flush=True)
pickle.dump(dict(rows=rows,kb=kb,ob=ob,pk=pk,x=p3.x,cst=cst),open('../datos/N3_pulsos.pkl','wb'))
