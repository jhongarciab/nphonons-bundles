"""Modelo de tasas (Pauli) del ciclo de bundle: estados (QD,m). Coeficientes analiticos.
 Gamma_m = 2 Omega^2 |<m+n|D|m>|^2 / gc_m , gc_m=[(2m+n)kap+gam+gph]/2   (excitacion Raman estimulada g,m -> x,m+n y retorno)
 emision: x_j->x_{j-1} (j kap), g_m->g_{m-1} (m kap) cuentan 1 fonon; decaimiento QD: x_j->g_j' a gam |<j'|D|j>|^2 (cierra ciclo)."""
import numpy as np
from bl import dmat, Omn
def cyc_rate(lam,Om,kap,gam,gph,n,M=9,K=10,pump_deph=True):
    idx=lambda q,m:q*M+m; S=2*M
    A=np.zeros((S,S)); Bc=np.zeros((S,S)); Jg=np.zeros((S,S))   # A: transiciones sin conteo (con diag), Bc: con conteo, Jg: cierre ciclo
    def add(M_,i,j,r):
        M_[j,i]+=r
        if M_ is A: A[i,i]-=r   # i->j con tasa r (la diagonal de Bc,Jg se suma luego)
    for m in range(M):
        gc=((2*m+n)*kap+gam+gph)/2
        if m+n<M:
            G=2*Om**2*dmat(lam,m+n,m)**2/gc
            add(A,idx(0,m),idx(1,m+n),G); add(A,idx(1,m+n),idx(0,m),G)
        if m>0:
            add(Bc,idx(0,m),idx(0,m-1),m*kap); add(Bc,idx(1,m),idx(1,m-1),m*kap)
        for mp in range(M):
            r=gam*dmat(lam,m,mp)**2   # x_m -> g_mp  (|<mp|D|m>|^2 simetrico)
            add(Jg,idx(1,m),idx(0,mp),r)
            if pump_deph and mp!=m-n:   # desfase inducido por el bombeo: gph*(Om d/Delta)^2 entre x_m y g_mp fuera de resonancia
                e=Om*dmat(lam,m,mp)/((m-n)-mp)
                r2=gph*e**2
                add(A,idx(1,m),idx(0,mp),r2); add(A,idx(0,mp),idx(1,m),r2)
    # A contiene diag de todas las salidas de A, Bc y Jg: reconstruir diag total
    out=np.zeros(S)
    for X in (Bc,Jg):
        out+=X.sum(axis=0)
    A0=A.copy()
    for i in range(S): A0[i,i]-=out[i]
    return A0,Bc,Jg,S,M
def cycles_rate(lam,Om,kap,gam,gph,n,M=9,K=10):
    # A0 incluye salidas por Bc y Jg en la diagonal; Bc,Jg aportan solo entradas
    A0,Bc,Jg,S,M=cyc_rate(lam,Om,kap,gam,gph,n,M,K)
    one=np.ones(S)
    Gam=lambda z: Jg@np.linalg.solve(-(A0+z*Bc),np.eye(S))
    w,v=np.linalg.eig(Gam(1.0)); rs=np.real(v[:,np.argmax(w.real)]); rs/=rs.sum()
    Mq=64; th=2*np.pi*np.arange(Mq)/Mq
    F=np.array([one@(Gam(np.exp(1j*t))@rs) for t in th])
    P=np.array([np.mean(F*np.exp(-1j*th*k)) for k in range(K+1)]).real
    T=one@np.linalg.solve(-(A0+Bc),rs)
    return P,T,(np.arange(K+1)*P).sum()/T
if __name__=='__main__':
    for kw in [dict(),dict(kap=.01),dict(kap=.001),dict(gam=2e-5),dict(lam=.2),dict(Om=.03)]:
        p=dict(lam=.14,Om=.1,kap=.003,gam=2e-4,gph=4e-4); p.update(kw)
        print(kw,np.round(cycles_rate(p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],2)[0][:6],3),'T=%.0f F=%.2e'%cycles_rate(p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],2)[1:])

def episodes_rate(lam,Om,kap,gam,gph,n,M=9,K=8):
    """Episodios (retorno a vacio de la cavidad) en el modelo de tasas."""
    A0,Bc,Jg,S,M=cyc_rate(lam,Om,kap,gam,gph,n,M,K)
    # separar Bc en transiciones m:1->0 (fin) y m>=2 (cuenta)
    End=np.zeros((S,S)); Cnt=Bc.copy()
    for q in (0,1):
        i=q*M+1; j=q*M+0
        End[j,i]=Bc[j,i]; Cnt[j,i]=0
    # A0 contiene todas las salidas; Jg entra como evolucion normal (no cuenta) -> sumar a la parte libre
    L0=A0+Jg
    one=np.ones(S)
    Gam=lambda z: z*End@np.linalg.solve(-(L0+z*Cnt),np.eye(S))
    w,v=np.linalg.eig(Gam(1.0)); rs=np.real(v[:,np.argmax(w.real)]); rs/=rs.sum()
    Mq=64; th=2*np.pi*np.arange(Mq)/Mq
    F=np.array([one@(Gam(np.exp(1j*t))@rs) for t in th])
    return np.array([np.mean(F*np.exp(-1j*th*k)) for k in range(K+1)]).real

# ---------- formulas cerradas (aproximacion de primer orden) ----------
from bl import eps_n
def Gamma_j(lam,Om,kap,gam,gph,n,j):
    gc=((2*j+n)*kap+gam+gph)/2
    return 2*Om**2*dmat(lam,n+j,j)**2/gc
def eps_parts(lam,Om,kap,gam,gph,n,mmax=8):
    """(fc, merge, off): piso Franck-Condon; fusion por decaimiento a mitad de cascada; excitacion fuera de resonancia (m != n)"""
    fc=1-np.exp(-lam**2)
    keep=1.0
    for j in range(1,n+1):
        G=Gamma_j(lam,Om,kap,gam,gph,n,j); keep*=1-gam/(j*kap+gam)*G/(G+j*kap)
    def R(m):
        gc=(m*kap+gam+gph)/2
        return 2*Om**2*dmat(lam,m,0)**2*gc/((m-n)**2+gc**2)
    Rn=R(n); Roff=sum(R(m) for m in range(1,mmax) if m!=n)
    return fc,1-keep,Roff/(Rn+Roff)
def P_closed(lam,Om,kap,gam,gph,n):
    fc,mer,off=eps_parts(lam,Om,kap,gam,gph,n); return np.exp(-lam**2)*(1-mer)*(1-off)
def flux_closed(lam,Om,kap,gam,gph,n):
    G0=Gamma_j(lam,Om,kap,gam,gph,n,0); return (n+lam**2)/(1/G0+1/gam)
def Om_max(lam,kap,gam,gph,n,X):
    """Omega maxima para pureza>=X (resolviendo P_closed=X por biseccion; mer ~ Omega^2)"""
    from scipy.optimize import brentq
    f=lambda Om:P_closed(lam,Om,kap,gam,gph,n)-X
    if f(1e-4)<0: return np.nan
    return brentq(f,1e-4,0.5) if f(0.5)<0 else 0.5
