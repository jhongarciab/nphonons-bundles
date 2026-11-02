"""Mapa de diseno a nivel de memoria. Para cada dispositivo TLS (muestra) y cada T: exponentes f1,f2,fxc del modelo TLS ->
Tphi_equiv=2T/f (Lindblad blanco equivalente en el instante final, rho=0) -> F_c, p_pass (chequeo JPNS) y baselines.
Baselines: B1 = dos resonadores separados (ruido independiente f_I=f1+f2, sin chequeo); B2 = mismo resonador, sin chequeo (correlacion, f_xc); B3 = chequeo (f_xc)."""
import sys, json, time, glob
import numpy as np
sys.path.insert(0,'.')
from tls_model import *
import erasure_sim as E
import qutip as qt
def chik(g,Dk,alpha=185.): return 2*np.pi*(g**2/Dk)*alpha/(Dk+alpha)
def mem_metrics(T,fexp,Tphi_equiv_mode,plat_base,g,Dtm,fsr,sigma,T1q,use_check=True):
    """fexp: exponente de la coherencia relativa en T. Lindblad equivalente: rho=0, Tphi=2T/fexp"""
    Tphi=2*T/max(fexp,1e-12)
    P=dict(T1=plat_base['T1'],Tphi=Tphi,T1q=T1q,T2q=1.5*T1q if T1q>30 else 15.,kp=(g/Dtm)**2/T1q)
    c1,c2=chik(g,Dtm),chik(g,Dtm+fsr)
    return E.run(T,0.0,P,sigma=sigma,chis=(c1,c2),zcomp=True,use_check=use_check)
