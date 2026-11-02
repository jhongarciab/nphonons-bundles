"""Mapa de diseno: para cada (g, FSR, T1q) optimiza (Delta_tm, sigma) a T=100us con Phi=0.42 (ruido blanco), luego evalua T y escenarios.
Salida: JSON con F_c, p_pass, baselines B1 (independiente, sin check), B2 (mismo resonador, sin check), F_h (heraldado ideal)."""
import sys, json, itertools, numpy as np, time
from multiprocessing import Pool
sys.path.insert(0,'.')
import erasure_sim as E
def chik(g,Dk,alpha=185.): return 2*2*np.pi*(g**2/Dk)*alpha/(Dk+alpha)   # desplazamiento g-e por foton = 2*chi (verificado con JC completo, VB)
SC={'S1_Phi0.42_white':(0.42,1),'S2_Phi0.42_slow':(0.42,2),'S3_Phi0.90_white':(0.90,1),'S0_nodeph':(None,None)}
PL=dict(T1=534.,Tphi=107.)
def plat(T1,Tph,g,D,T1q): return dict(T1=T1,Tphi=Tph,T1q=T1q,T2q=1.5*T1q if T1q>30 else 15.,kp=(g/D)**2/T1q)
def evalpoint(a):
    g,fsr,T1q,D,sg,T,sc,T1m,Tphi1=a
    Phi,n=SC[sc]
    if Phi is None: Tph=None
    else: Tph=Tphi1/((1-Phi)**n)
    P=plat(T1m,Tph,g,D,T1q)
    r=E.run(T,0.0,P,sigma=sg,chis=(chik(g,D),chik(g,D+fsr)),zcomp=True,use_check=True)
    return r
def job_opt(a):
    g,fsr,T1q=a
    best=None
    for D in (4,6,10,20,40,80):
        if g/D>0.2: continue
        for sg in (0.5,0.7,1.2,2,3.5,6):
            r=evalpoint((g,fsr,T1q,D,sg,100.,'S1_Phi0.42_white',534.,107.))
            Q=r['p_pass']*(2*r['F_c']-1)
            if best is None or Q>best[0]: best=(Q,D,sg,r)
    return (g,fsr,T1q,best[1],best[2],best[0])
def job_eval(a):
    g,fsr,T1q,D,sg,T,sc,T1m,Tphi1=a
    r=evalpoint(a)
    Phi,n=SC[sc]
    # baselines sin transmon (kp=0, sin check)
    Tph2=None if Phi is None else Tphi1/((1-Phi)**n)
    b2=E.run(T,0.0,dict(T1=T1m,Tphi=Tph2,T1q=100.,T2q=150.,kp=0.0),use_check=False)
    b1=E.run(T,0.0,dict(T1=T1m,Tphi=Tphi1 if Phi is not None else None,T1q=100.,T2q=150.,kp=0.0),use_check=False)
    return dict(g=g,fsr=fsr,T1q=T1q,D=D,sg=sg,T=T,sc=sc,T1m=T1m,Tphi1=Tphi1,F_c=r['F_c'],p_pass=r['p_pass'],B2_Func=b2['F_unc'],B2_Fh=b2['F_h'],B1_Func=b1['F_unc'],B1_Fh=b1['F_h'])
if __name__=="__main__":
    t0=time.time()
    grid=list(itertools.product((1.1,3.0,10.0),(3.3,1.0),(28.,100.,300.,1000.)))
    with Pool(2) as pool: opt=pool.map(job_opt,grid)
    json.dump(opt,open('design_opt.json','w')); print("opt done",time.time()-t0,flush=True)
    jobs=[(g,fsr,T1q,D,sg,T,sc,534.,107.) for (g,fsr,T1q,D,sg,Q) in opt for T in (25.,50.,100.,200.,500.) for sc in ('S1_Phi0.42_white','S2_Phi0.42_slow','S3_Phi0.90_white')]
    with Pool(2) as pool: res=pool.map(job_eval,jobs)
    json.dump(res,open('design_eval_si.json','w')); print("eval done",time.time()-t0,flush=True)
