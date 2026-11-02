from erasure_sim import *
from scan_check import chi_of
import itertools, sys
def plat(T1,Tphi,g,D,T1q):
    return dict(T1=T1,Tphi=Tphi,T1q=T1q,T2q=1.5*T1q if T1q>30 else 15.,kp=(g/D)**2/T1q)
def best(T1,Tphi,g,T1q,T,rho):
    bestv=None
    for D in (3,4,6,10):
        for sg in (0.7,1.0,1.5,2.5):
            P=plat(T1,Tphi,g,D,T1q); r=run(T,rho,P,sigma=sg,chi=chi_of(g,D))
            if bestv is None or r["F_c"]>bestv[0]["F_c"]: bestv=(r,D,sg)
    return bestv
print("SILICIO  g=1.1MHz T1=534 Tphi=107 | baseline pasivo: modos independientes (rho=0), kp=0")
print("T1q  T[us] rho | D sg | F_c(check) p_pass | F_h(base, heraldado al final) | F_unc(base sin heraldar) F_unc(doble carril sin check)")
for T1q in (28.,100.,300.):
    for T in (50.,100.,200.):
        for rho in (0.65,0.95):
            r,D,sg=best(534.,107.,1.1,T1q,T,rho)
            base=run(T,0.0,dict(T1=534.,Tphi=107.,T1q=T1q,T2q=15.,kp=0.0),use_check=False)
            dr=run(T,rho,dict(T1=534.,Tphi=107.,T1q=T1q,T2q=15.,kp=0.0),use_check=False)
            print(f"{T1q:5.0f} {T:5.0f} {rho:4} | D={D:2} sg={sg:3} | Fc={r['F_c']:.3f} p={r['p_pass']:.3f} | F_h_base={base['F_h']:.3f} F_h_dr(rho)={dr['F_h']:.3f} | F_unc_base={base['F_unc']:.3f} F_unc_dr={dr['F_unc']:.3f}")
