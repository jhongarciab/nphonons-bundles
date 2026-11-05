"""Captura directa de un paquete itinerante en un estado oscuro con puerto FIJO (modelo efectivo de un fonon).
Cavidad a, estado oscuro d, acoplo Raman efectivo G(t)=Om(t) g/Delta_I (real), puerto kappa_e fijo, perdida intrinseca kappa_i, desintonia compensada (chirp ideal).
  da/dt = -(kappa/2) a - i G d - sqrt(kappa_e) u(t),   dd/dt = -i G a - (gd/2) d,   kappa = kappa_e + kappa_i.
Eficiencia de captura = |d(T)|^2 / int|u|^2.
Resultado analitico (perdidas nulas): captura perfecta <=> G(t) = -sqrt(kappa) d(delta)/dt / u, con delta^2(t) = int_0^t u^2 - u^2/kappa >= 0.
Unidades omega_b = 1."""
import numpy as np
from scipy.optimize import minimize
def pulse(kind, Tp, t, t0):
    if kind == 'gauss':
        u = np.exp(-(t - t0)**2/(4*Tp**2))
    elif kind == 'exp_dec':      # exponencial decreciente (forma natural de emision)
        u = np.where(t >= t0, np.exp(-(t - t0)/(2*Tp)), 0.0)
    elif kind == 'exp_gro':      # exponencial creciente truncada (emision invertida)
        u = np.where(t <= t0, np.exp((t - t0)/(2*Tp)), 0.0)
    elif kind == 'sech':
        u = 1/np.cosh((t - t0)/Tp)
    return u/np.sqrt(np.sum(u**2)*(t[1]-t[0]))
def simulate(G, u, t, ke, ki, gd=0.0, ret=False, dc=0.0, shift=0.0):
    dt = t[1]-t[0]; k = ke+ki; a = 0j; d = 0j; N = len(t)
    def f(a, d, G_, u_): return (-(k/2)*a - 1j*dc*a - 1j*(shift*G_**2)*a - 1j*G_*d - np.sqrt(ke)*u_, -1j*G_*a - (gd/2)*d)
    A = np.empty(N, complex); D = np.empty(N, complex)
    for n in range(N-1):
        A[n] = a; D[n] = d
        G0, G1, G2 = G[n], 0.5*(G[n]+G[n+1]), G[n+1]; u0, u1, u2 = u[n], 0.5*(u[n]+u[n+1]), u[n+1]
        k1 = f(a, d, G0, u0); k2 = f(a+dt/2*k1[0], d+dt/2*k1[1], G1, u1); k3 = f(a+dt/2*k2[0], d+dt/2*k2[1], G1, u1); k4 = f(a+dt*k3[0], d+dt*k3[1], G2, u2)
        a += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); d += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    A[-1] = a; D[-1] = d
    if ret: return A, D
    return abs(d)**2
def ideal_G(u, t, ke):
    """G(t) de captura perfecta sin perdidas (kappa = ke)."""
    dt = t[1]-t[0]; I = np.cumsum(u**2)*dt; d2 = np.clip(I - u**2/ke, 0, None); delta = np.sqrt(d2)
    ddelta = np.gradient(delta, dt)
    with np.errstate(divide='ignore', invalid='ignore'): G = -np.sqrt(ke)*ddelta/u
    return np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0), np.min(d2[u > 1e-3*u.max()]) if (u > 1e-3*u.max()).any() else 0
def optimize_G(u, t, ke, ki, Gmax, nn=14, gd=0.0, G0=None, seed=0):
    tn = np.linspace(t[0], t[-1], nn)
    prof = lambda x: np.interp(t, tn, x)
    f = lambda x: -simulate(np.clip(prof(x), -Gmax, Gmax), u, t, ke, ki, gd)
    best = None
    starts = [np.full(nn, Gmax*0.5)]
    if G0 is not None: starts.append(np.clip(np.interp(tn, t, G0), -Gmax, Gmax))
    for x0 in starts:
        r = minimize(f, x0, method='L-BFGS-B', bounds=[(-Gmax, Gmax)]*nn, options=dict(maxiter=300, ftol=1e-10))
        if best is None or r.fun < best.fun: best = r
    return np.clip(prof(best.x), -Gmax, Gmax), -best.fun

import sys
kes=[float(x) for x in sys.argv[1:]]
ki,Gmax=1e-3,0.012
print('ki=1e-3 Gmax=0.012 forma gauss; filas ke, columnas Tp: (eta optimizada | eta ideal sin perdida intrinseca)')
for ke in kes:
    row=[]
    for Tp in (50.,100.,150.,250.):
        t=np.arange(-8*Tp,8*Tp,4.0); u=pulse('gauss',Tp,t,0.0); Gi,_=ideal_G(u,t,ke+ki)
        Go,e=optimize_G(u,t,ke,ki,Gmax,nn=12,G0=Gi); e0=simulate(np.clip(Gi,-Gmax,Gmax),u,t,ke+ki,0.0)
        row.append('Tp=%d: %.3f|%.3f'%(Tp,e,e0))
    print('ke=%.3f  '%ke+'  '.join(row),flush=True)
