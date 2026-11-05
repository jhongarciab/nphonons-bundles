"""Canal completo en el modelo efectivo: captura directa de los dos paquetes, espera, lectura por emision con el control invertido, sumideros en cascada
adaptados a los paquetes ideales de salida (mismo formalismo que la fuente). Base por rama: [s0,s1,a,d1,d2,c0,c1]. Desfase comun chi=1 (mascara N=(0,0,0,1,1,0,0))."""
import numpy as np
from captura import pulse, optimize_G, ideal_G
ke, ki, Gmax, dt, Tp = 1e-2, 1e-3, 0.012, 2.0, 100.0
gph, Dss, r12 = 4e-4, 0.25, 0.0613/0.0572; r21 = 1/r12; eps = 0.02
def prof(tc, Gs, ts, t): return np.interp(t, ts+tc, Gs, left=0, right=0)
def canal(sep, Twait, Gs, ts, gph_=gph, ki_=ki):
    tc1, tc2 = 0.0, sep; tr1 = sep + Twait + 700.0; tr2 = tr1 + sep
    t = np.arange(-9*Tp, tr2+9*Tp+200, dt); n = len(t)
    u0 = pulse('gauss', Tp, t, tc1)*np.sqrt(1-eps); u1 = pulse('gauss', Tp, t, tc2)*np.sqrt(1-eps)
    cum0 = np.cumsum(u0**2)*dt; cum1 = np.cumsum(u1**2)*dt; gs0 = u0**2/np.clip(1-cum0, 1e-6, None); gs1 = u1**2/np.clip(1-cum1, 1e-6, None)
    # control: captura D1@tc1, D2@tc2 ; lectura D1@tr1, D2@tr2 con G invertido
    G1 = prof(tc1, Gs, ts, t) + prof(tr1, Gs[::-1], ts, t); G2 = prof(tc2, Gs, ts, t) + prof(tr2, Gs[::-1], ts, t)
    # sumideros: paquete ideal de salida gaussiano centrado en tr1 (rama 0) y tr2 (rama 1)
    w0 = pulse('gauss', Tp, t, tr1)*np.sqrt(1-eps); w1 = pulse('gauss', Tp, t, tr2)*np.sqrt(1-eps)
    def gc(w):
        cumw = np.cumsum(w**2)*dt   # energia acumulada (inversion temporal de la fuente)
        return w**2/np.clip(eps + cumw, 1e-9, None)
    gc0 = gc(w0); gc1 = gc(w1)
    mask = np.array([0, 0, 0, 1, 1, 0, 0.]); Dm = (mask[:, None]-mask[None, :])**2
    def M(i, tt):
        m = np.zeros((7, 7), complex); k = ke+ki_
        m[0, 0] = -gs0[i]/2; m[1, 1] = -gs1[i]/2; m[2, 2] = -k/2
        m[2, 0] = -np.sqrt(gs0[i]*ke); m[2, 1] = -np.sqrt(gs1[i]*ke)
        a1 = G1[i]+G2[i]*r21*np.exp(1j*Dss*tt); a2 = G2[i]+G1[i]*r12*np.exp(-1j*Dss*tt)
        m[2, 3] = -1j*a1; m[3, 2] = -1j*a1; m[2, 4] = -1j*a2; m[4, 2] = -1j*a2
        m[5, 5] = -gc0[i]/2; m[6, 6] = -gc1[i]/2; m[5, 2] = -np.sqrt(gc0[i]*ke); m[6, 2] = -np.sqrt(gc1[i]*ke)
        return m
    p0 = np.zeros(7, complex); p0[0] = 1; p1 = np.zeros(7, complex); p1[1] = 1
    R = [np.outer(p0, p0.conj()), np.outer(p1, p1.conj()), np.outer(p0, p1.conj())]
    f = lambda X, Mn: Mn@X + X@Mn.conj().T - 0.5*gph_*Dm*X
    for i in range(n-1):
        t0 = t[i]; Mn0 = M(i, t0); Mn1 = M(i, t0+dt/2); Mn2 = M(i+1, t0+dt); new = []
        for X in R:
            k1 = f(X, Mn0); k2 = f(X+dt/2*k1, Mn1); k3 = f(X+dt/2*k2, Mn1); k4 = f(X+dt*k3, Mn2); new.append(X+dt/6*(k1+2*k2+2*k3+k4))
        R = new
    nrm = 1-eps; r00, r11, r01 = R[0]/nrm, R[1]/nrm, R[2]/nrm   # normaliza por la energia del paquete
    P0, P1 = r00[5, 5].real, r11[6, 6].real; V = abs(r01[5, 6])/np.sqrt(P0*P1)
    return dict(P0=P0, P1=P1, V=V, leak_out0=r00[6, 6].real, leak_out1=r11[5, 5].real, rest_d=r00[3, 3].real+r11[4, 4].real)
if __name__ == '__main__':
    t = np.arange(-9*Tp, 9*Tp, dt); u = pulse('gauss', Tp, t, 0.0); Gi, _ = ideal_G(u, t, ke+ki); Gs, e1 = optimize_G(u, t, ke, ki, Gmax, G0=Gi)
    print('captura individual (efectivo) = %.3f'%e1, flush=True)
    print('%-8s %-8s | %-7s %-7s %-7s | %-9s %-12s'%('sep', 'espera', 'P0', 'P1', 'V', 's=(P0+P1)/2', 'F_h=(1+V)/2'))
    for sep, Tw in ((450., 0.), (450., 2000.), (890., 0.), (890., 2000.)):
        m = canal(sep, Tw, Gs, t); print('%-8.0f %-8.0f | %-7.3f %-7.3f %-7.3f | %-9.3f %-12.3f   (cruce de salida %.1e, resto en d %.1e)'%(sep, Tw, m['P0'], m['P1'], m['V'], (m['P0']+m['P1'])/2, (1+m['V'])/2, max(m['leak_out0'], m['leak_out1']), m['rest_d']), flush=True)
