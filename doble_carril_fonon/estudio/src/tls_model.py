"""Modelo de ruido de frecuencia por TLS en dos modos de un resonador (Monte Carlo + funcion caracteristica telegrafica exacta).
Unidades: us, rad/us. delta_omega_k(t)=sum_j A_jk s_j(t), s_j telegrafico +-1 (tasa de salto gam_j), A_jk=g_j^2 u_k(x_j)^2/(w_k-Omega_j).
Phi = 1 - t_I/t_xc con la definicion de Hitchcock et al. (arXiv:2502.16426, App. J): f(t)=-ln|<e^{-i phi}>|, f_I=f1+f2, f_xc=f(phi1-phi2); t con f=1."""
import numpy as np
def cf(delta,gam,t):
    """<exp(-i*delta*∫0^t s)> telegrafico +-1 estacionario; delta,gam arrays (N,), t escalar"""
    w=np.sqrt((gam**2-delta**2).astype(complex)); w=np.where(np.abs(w)<1e-12,1e-12,w)
    return 0.5*((1+gam/w)*np.exp((w-gam)*t)+(1-gam/w)*np.exp(-(w+gam)*t))
def logabs_prod(delta,gam,tgrid):
    out=np.empty(len(tgrid))
    for i,t in enumerate(tgrid):
        z=cf(delta,gam,t); out[i]=np.sum(np.log(np.maximum(np.abs(z),1e-300)))
    return out
def first_cross(tgrid,f,level=1.0,cap=None):
    idx=np.where(f>=level)[0]
    if len(idx)==0: return np.nan
    i=idx[0]
    if i==0: return tgrid[0]
    t0,t1,f0,f1=tgrid[i-1],tgrid[i],f[i-1],f[i]
    return t0+(level-f0)*(t1-t0)/(f1-f0)
TG=np.geomspace(0.5,30000.,90)
CENS=np.nan   # tiempo sin cruce -> NaN (censurado); Phi=NaN
def sample_device(rng,p):
    N=rng.poisson(p['nf']*2*p['W'])                      # TLS en [-W,W] MHz alrededor de w0
    Om=rng.uniform(-p['W'],p['W'],N)                     # MHz relativo a w0 (modo 0)
    lg=rng.uniform(np.log(p['glo']),np.log(p['ghi']),N)  # g (MHz) log-uniforme
    g=np.exp(lg)
    gam=np.exp(rng.uniform(np.log(p['glo_rate']),np.log(p['ghi_rate']),N))  # 1/us
    ph=rng.uniform(0,np.pi,N)  # posicion normalizada
    return Om,g,gam,ph
def amplitudes(Om,g,ph,k,p,kmode):
    """amplitud rad/us de TLS j sobre el modo con desintonia k (MHz) y indice kmode"""
    if p['profile']=='travelling': u2=np.ones_like(g)
    else: u2=2*np.sin(kmode*ph)**2   # kmode: indice entero del modo
    return 2*np.pi*p['amp']*(g**2)*u2/(k-Om)             # (MHz^2/MHz)=MHz -> rad/us con 2pi
def phi_curve(rng,p,detunings,kbase=1000):
    Om,g,gam,ph=sample_device(rng,p)
    A1=amplitudes(Om,g,ph,0.0,p,kbase)
    ng=p.get('glob_n',0)
    if ng:   # fluctuadores globales: misma amplitud en ambos modos (campo comun)
        Ag=2*np.pi*p['glob_amp']*rng.uniform(0.3,1.0,ng); gg=np.exp(rng.uniform(np.log(p['glo_rate']),np.log(p['ghi_rate']),ng))
    res=[]
    f1=-logabs_prod(A1,gam,TG)
    if ng: f1=f1-logabs_prod(Ag,gg,TG)
    t1=first_cross(TG,f1)
    for d in detunings:
        n=int(round(d/p['fsr']))
        A2=amplitudes(Om,g,ph,d,p,kbase+n)
        f2=-logabs_prod(A2,gam,TG)
        if ng: f2=f2-logabs_prod(Ag,gg,TG)
        fxc=-logabs_prod(A1-A2,gam,TG)   # los globales se cancelan
        tI=first_cross(TG,f1+f2); tx=first_cross(TG,fxc)
        res.append((1-tI/tx, t1, first_cross(TG,f2)))
    return np.array(res)

def noise_exponents(rng,p,d,Ts,kbase=1000):
    """un dispositivo: exponentes de coherencia f1,f2,f_xc en los tiempos Ts (us) para modos separados d (MHz).
    f_I=f1+f2 (ruido independiente) ; f_xc incluye correlacion (fase relativa del doble carril)."""
    Ts=np.asarray(Ts,float)
    Om,g,gam,ph=sample_device(rng,p)
    n=int(round(d/p['fsr']))
    A1=amplitudes(Om,g,ph,0.0,p,kbase); A2=amplitudes(Om,g,ph,d,p,kbase+n)
    def lp(A): return np.array([np.sum(np.log(np.maximum(np.abs(cf(A,gam,t)),1e-300))) for t in Ts])
    f1,f2,fx=-lp(A1),-lp(A2),-lp(A1-A2)
    return f1,f2,fx
