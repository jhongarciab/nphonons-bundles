"""Mapa de diseno v2 (convenciones corregidas tras revision VC).
Convencion Hitchcock: Tphi definido sobre |<a>|^2 => f_1(T)=-ln|<e^{-i phi}>| = (T/Tphi)^n /2 por modo; f_I(T)=f1+f2=(T/Tphi)^n (f_I(Tphi)=1);
f_xc=(1-Phi)^n f_I  (Gaussiano, ver R1). Lindblad equivalente rho=0 con Tphi'=2T/f_xc (exponente relativo = 2T/Tphi').
Lineas base: B0 = un solo carril {|0>,|1>} sin check (cerrada); B1 = doble carril, dos resonadores (f_I), sin check; B2 = mismo resonador (f_xc) sin check;
D = doble carril + check JPNS; Dh = doble carril con heraldado ideal (conteo perfecto).
Metricas: F_c y p_pass; F_ef = p F_c + (1-p)/2 (fallo heraldado -> adivinar)."""
import sys, json, itertools, numpy as np, time
from multiprocessing import Pool
sys.path.insert(0,'.')
import erasure_sim as E
def chik(g,Dk,alpha=185.): return 2*2*np.pi*(g**2/Dk)*alpha/(Dk+alpha)   # desplazamiento g-e por foton = 2chi
def f_exponents(T,Tphi,Phi,n):
    if Tphi is None: return 0.0,0.0
    fI=(T/Tphi)**n; return fI,((1-Phi)**n)*fI
def B0(T,T1,fI):
    p=np.exp(-T/T1); eta=np.sqrt(p)*np.exp(-fI/2); return (3+p+2*eta)/6
def evalpoint(a):
    g,fsr,T1q,D,sg,T,Tphi,Phi,n,T1m=a
    fI,fx=f_exponents(T,Tphi,Phi,n)
    Tph=None if fx<1e-9 else 2*T/fx
    P=dict(T1=T1m,Tphi=Tph,T1q=T1q,T2q=1.5*T1q if T1q>30 else 15.,kp=(g/D)**2/T1q)
    r=E.run(T,0.0,P,sigma=sg,chis=(chik(g,D),chik(g,D+fsr)),zcomp=True,use_check=True)
    b2=E.run(T,0.0,dict(T1=T1m,Tphi=Tph,T1q=100.,T2q=150.,kp=0.0),use_check=False)
    TphI=None if fI<1e-9 else 2*T/fI
    b1=E.run(T,0.0,dict(T1=T1m,Tphi=TphI,T1q=100.,T2q=150.,kp=0.0),use_check=False)
    p=r['p_pass']; pid=np.exp(-T/T1m)
    return dict(g=g,fsr=fsr,T1q=T1q,D=D,sg=sg,T=T,Tphi=Tphi,Phi=Phi,n=n,T1m=T1m,F_c=r['F_c'],p_pass=p,F_ef=p*r['F_c']+(1-p)/2,
                B0=B0(T,T1m,fI),B1=b1['F_unc'],B2=b2['F_unc'],Dh=b2['F_h'],p_ideal=pid,F_ef_ideal=pid*b2['F_h']+(1-pid)/2)
def job_opt(a):
    g,fsr,T1q=a; best=None
    for D in (6,10,20,40,80,160):
        if g/D>0.2: continue
        for sg in (0.3,0.5,0.7,1.0,1.5,2.0,3.5,6.0):
            r=evalpoint((g,fsr,T1q,D,sg,100.,107.,0.42,1,534.))
            if best is None or r['F_ef']>best[0]: best=(r['F_ef'],D,sg)
    return (g,fsr,T1q,best[1],best[2],best[0])
def job_eval(a): return evalpoint(a)
if __name__=="__main__":
    t0=time.time()
    grid=list(itertools.product((1.1,3.0,10.0),(3.3,1.0),(28.,100.,300.)))
    with Pool(3) as pool: opt=pool.map(job_opt,grid)
    json.dump(opt,open('design2_opt.json','w')); print("opt",time.time()-t0,flush=True)
    SC=[(107.,0.42,1),(107.,0.42,2),(107.,0.90,1),(10.,0.42,1),(10.,0.90,1)]
    jobs=[(g,fsr,T1q,D,sg,T,Tphi,Phi,n,534.) for (g,fsr,T1q,D,sg,_) in opt for T in (25.,50.,100.,200.,500.) for (Tphi,Phi,n) in SC]
    with Pool(3) as pool: res=pool.map(job_eval,jobs)
    json.dump(res,open('design2_eval_si.json','w')); print("eval",time.time()-t0,flush=True)
