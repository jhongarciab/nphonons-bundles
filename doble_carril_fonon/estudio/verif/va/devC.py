import sys;sys.path.insert(0,'../../../src'); sys.path.insert(0,'../../src')
import numpy as np, json
from tls_model import *
from mc2 import mc
from mc import cross
def build(seed,p):
    rng=np.random.default_rng(seed)
    Om,g,gam,ph=sample_device(rng,p)
    A1=amplitudes(Om,g,ph,0.0,p,1000)
    ng=p.get('glob_n',0); Ag=gg=np.array([])
    if ng:
        Ag=2*np.pi*p['glob_amp']*rng.uniform(0.3,1.0,ng); gg=np.exp(rng.uniform(np.log(p['glo_rate']),np.log(p['ghi_rate']),ng))
    d=p['fsr'];n=int(round(d/p['fsr']))
    A2=amplitudes(Om,g,ph,d,p,1000+n)
    return rng,Om,g,gam,ph,A1,A2,Ag,gg
def run(prof,seed,glob=0,ntraj=20000,Tmax=None,w=0):
    base=dict(nf=0.05,W=80.,glo=0.002,ghi=0.5,glo_rate=1e-3,ghi_rate=1e2,amp=1.0,fsr=3.29,profile=prof,glob_n=glob,glob_amp=0.005)
    p=base
    rng0=np.random.default_rng(seed);ref=phi_curve(rng0,p,[p['fsr']])[0]
    rng,Om,g,gam,ph,A1,A2,Ag,gg=build(seed,p)
    # code-side f curves
    f1=-logabs_prod(A1,gam,TG);f2=-logabs_prod(A2,gam,TG);fx=-logabs_prod(A1-A2,gam,TG)
    if glob: f1=f1-logabs_prod(Ag,gg,TG);f2=f2-logabs_prod(Ag,gg,TG)
    # index range needed
    need=[np.where(f>=1)[0] for f in (f1,f2,f1+f2,fx)]
    imax=max([x[0] for x in need if len(x)]+[10])+1
    imax=min(imax,len(TG)-1)
    if Tmax: imax=min(imax,int(np.searchsorted(TG,Tmax)))
    tg=TG[:imax+1]
    G1=np.concatenate([A1,Ag]);G2=np.concatenate([A2,Ag]);gm=np.concatenate([gam,gg])
    m1,m2,mx=mc(G1,G2,gm,tg,ntraj,seed+777)
    return dict(p=p,ref=ref,N=len(A1),Om=Om,g=g,gam=gam,A1=A1,A2=A2,Ag=Ag,gg=gg,tg=tg,
      code=(f1[:imax+1],f2[:imax+1],fx[:imax+1]),mc=(m1,m2,mx),TG=TG,codefull=(f1,f2,fx))
def summarize(r):
    tg=r['tg'];c=r['code'];m=r['mc']
    out={}
    for nm,(cf_,mf) in zip(('f1','f2','fxc'),zip(c,m)):
        sel=(cf_>0.05)&(cf_<4)
        out[nm+'_maxabs']=float(np.max(np.abs(cf_[sel]-mf[sel]))) if sel.any() else None
    def X(f): return cross(tg,f)
    tI_c=X(c[0]+c[1]);tx_c=X(c[2]);t1_c=X(c[0]);t2_c=X(c[1])
    tI_m=X(m[0]+m[1]);tx_m=X(m[2]);t1_m=X(m[0]);t2_m=X(m[1])
    out.update(ref=r['ref'].tolist(),Phi_code=1-tI_c/tx_c,Phi_mc=1-tI_m/tx_m,t1_c=t1_c,t1_m=t1_m,t2_c=t2_c,t2_m=t2_m,tI_c=tI_c,tI_m=tI_m,tx_c=tx_c,tx_m=tx_m)
    return out
if __name__=='__main__':
    prof,seed,glob=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);Tm=float(sys.argv[4]) if len(sys.argv)>4 else None
    r=run(prof,seed,glob,Tmax=Tm); s=summarize(r)
    s.update(prof=prof,seed=seed,glob=glob,N=r['N'])
    print(json.dumps(s,default=float))
    np.save(f'dev_{prof}_{seed}_{glob}.npy',np.array([r['tg'],*r['code'],*r['mc']]))
