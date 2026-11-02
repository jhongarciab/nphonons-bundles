"""Umbral de correlacion rho* para que el doble carril con chequeo IDEAL supere al carril simple.
Dual-rail: F_h=(2+C)/3, C=exp(-(1-rho) f_I), f_I=(T/Tphi)^n, p=exp(-T/T1). F_ef=p F_h+(1-p)/2.
Carril simple: F_s=(3+p+2 sqrt(p) e^{-f_I/2})/6. (promedios sobre los 6 estados cardinales)"""
import numpy as np, json
def thresholds(T,T1,Tphi,n):
    p=np.exp(-T/T1); fI=(T/Tphi)**n; Fs=(3+p+2*np.sqrt(p)*np.exp(-fI/2))/6
    # metrica condicional: (2+C)/3 > Fs -> C > 3Fs-2
    C1=3*Fs-2; r1=1-(-np.log(C1)/fI) if 0<C1<1 and fI>0 else (-np.inf if C1<=0 else np.nan)
    # F_ef: p(2+C)/3+(1-p)/2 > Fs -> C > 3(Fs-(1-p)/2)/p-2
    C2=3*(Fs-(1-p)/2)/p-2; r2=1-(-np.log(C2)/fI) if 0<C2<1 and fI>0 else (-np.inf if C2<=0 else np.nan)
    return r1,r2,Fs
rows=[]
print("n Tphi |  T  | rho*_cond | rho*_Fef | F_single")
for n in (1,2):
    for Tphi in (107.,10.):
        for T in (10.,25.,50.,100.,200.):
            r1,r2,Fs=thresholds(T,534.,Tphi,n); rows.append(dict(n=n,Tphi=Tphi,T=T,rho_cond=float(r1),rho_Fef=float(r2),F_single=float(Fs)))
            print(f"{n} {Tphi:5.0f} | {T:4.0f} | {r1:8.3f} | {r2:8.3f} | {Fs:.3f}")
json.dump(rows,open('breakeven.json','w'))
