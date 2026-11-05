"""Canal completo, modelo efectivo, protocolos comparados con la MISMA entrada y el mismo modelo de ruido:
 directa  : captura directa con puerto fijo (control G(t) optimizado para la forma de entrada), espera, lectura con G invertido.
 dos_etapas: puerto conmutable (ke_in en la ventana de carga, cerrado el resto), pi cuadrado de 190 para escribir/leer, lectura abriendo el puerto (emision exponencial).
 pasivo   : dos modos acusticos pasivos con los mismos puertos (analitico/simple).
Ruido (chi): desfase vacio-exciton gph/2 para todo chi; coherencia d1-d2 decae a (5 gph/6)(1-chi). Base [s0,s1,a,d1,d2,c0,c1]. Unidades wb=1."""
import numpy as np, sys, itertools
from multiprocessing import Pool
from captura import pulse, optimize_G, ideal_G
ke_in, ki, Gmax, dt, Tp = 1e-2, 1e-3, 0.012, 2.0, 100.0
gph, Dss, r12 = 4e-4, 0.25, 0.0613/0.0572; r21 = 1/r12; eps = 0.02; dpi = 190.0
def sinkc(w):
    cw = np.cumsum(w**2)*dt; return w**2/np.clip(eps+cw, 1e-9, None)
def source_g(u):
    cu = np.cumsum(u**2)*dt; return u**2/np.clip(1-cu, 1e-6, None)
def integrar(t, ke_t, G1, G2, u0, u1, w0, w1, chi, gph_=gph):
    n = len(t); gs0, gs1 = source_g(u0), source_g(u1); gc0, gc1 = sinkc(w0), sinkc(w1)
    mask = np.array([0, 0, 0, 1, 1, 0, 0.]); Dm = (mask[:, None]-mask[None, :])**2; Dm[3, 4] = Dm[4, 3] = (5/3)*(1-chi)
    def M(i, tt):
        m = np.zeros((7, 7), complex); k = ke_t[i]+ki
        m[0, 0] = -gs0[i]/2; m[1, 1] = -gs1[i]/2; m[2, 2] = -k/2
        m[2, 0] = -np.sqrt(gs0[i]*ke_t[i]); m[2, 1] = -np.sqrt(gs1[i]*ke_t[i])
        a1 = G1[i]+G2[i]*r21*np.exp(1j*Dss*tt); a2 = G2[i]+G1[i]*r12*np.exp(-1j*Dss*tt)
        m[2, 3] = -1j*a1; m[3, 2] = -1j*a1; m[2, 4] = -1j*a2; m[4, 2] = -1j*a2
        m[5, 5] = -gc0[i]/2; m[6, 6] = -gc1[i]/2; m[5, 2] = -np.sqrt(gc0[i]*ke_t[i]); m[6, 2] = -np.sqrt(gc1[i]*ke_t[i])
        return m
    p0 = np.zeros(7, complex); p0[0] = 1; p1 = np.zeros(7, complex); p1[1] = 1
    R = [np.outer(p0, p0.conj()), np.outer(p1, p1.conj()), np.outer(p0, p1.conj())]
    f = lambda X, Mn: Mn@X + X@Mn.conj().T - 0.5*gph_*Dm*X
    for i in range(n-1):
        t0 = t[i]; Mn0 = M(i, t0); Mn1 = M(i, t0+dt/2); Mn2 = M(i+1, t0+dt); new = []
        for X in R:
            k1 = f(X, Mn0); k2 = f(X+dt/2*k1, Mn1); k3 = f(X+dt/2*k2, Mn1); k4 = f(X+dt*k3, Mn2); new.append(X+dt/6*(k1+2*k2+2*k3+k4))
        R = new
    nrm = 1-eps; r00, r11, r01 = R[0]/nrm, R[1]/nrm, R[2]/nrm
    P0, P1 = r00[5, 5].real, r11[6, 6].real; V = abs(r01[5, 6])/np.sqrt(max(P0*P1, 1e-30))
    s = (P0+P1)/2; F = (P0+P1)/4+np.sqrt(P0*P1)*V/2
    return dict(P0=P0, P1=P1, V=V, s=s, Fh=F/s if s > 0 else 0.0)
def pasivo(kind, Twait, sep):
    """dos modos pasivos, puerto conmutable. Carga: cavidad con ke_in en la ventana; lectura: ke_in con emision exponencial. Primer bin espera sep mas."""
    t = np.arange(-9*Tp, 9*Tp, dt); u = pulse(kind, Tp, t, 0.0); win = (-800, 2) if kind == 'exp_gro' else (-300, 300)
    a = 0j; k = ke_in+ki
    for i in range(len(t)-1):
        inwin = win[0] <= t[i] <= win[1]
        if inwin: a += dt*(-(k/2)*a - np.sqrt(ke_in)*u[i])
        else: a += dt*(-(ki/2)*a)
    etaL = abs(a)**2
    etaR = ke_in/(ke_in+ki)*(1-eps)   # emision exponencial con sumidero adaptado (regularizado)
    p2 = etaL*etaR*np.exp(-ki*Twait); p1 = etaL*etaR*np.exp(-ki*(Twait+sep)); s = (p1+p2)/2; F = (p1+p2)/4+np.sqrt(p1*p2)/2
    return dict(P0=p1, P1=p2, V=2*np.sqrt(p1*p2)/(p1+p2), s=s, Fh=F/s)
def directo(kind, chi, Twait):
    win = (-300, 300) if kind == 'gauss' else (-800, 2); sep = (win[1]-win[0])+dpi+100.0
    ts = np.arange(-9*Tp, 9*Tp, dt); us = pulse(kind, Tp, ts, 0.0); Gi, _ = ideal_G(us, ts, ke_in+ki); Gs, _ = optimize_G(us, ts, ke_in, ki, Gmax, G0=Gi)
    tc1, tc2 = 0.0, sep; tr1 = tc2+900+900+Twait; tr2 = tr1+sep; t = np.arange(-9*Tp, tr2+9*Tp+200, dt)
    pr = lambda tc, G_: np.interp(t, ts+tc, G_, left=0, right=0)
    G1 = pr(tc1, Gs)+pr(tr1, Gs[::-1]); G2 = pr(tc2, Gs)+pr(tr2, Gs[::-1])
    u0 = pulse(kind, Tp, t, tc1)*np.sqrt(1-eps); u1 = pulse(kind, Tp, t, tc2)*np.sqrt(1-eps)
    okind = 'gauss' if kind == 'gauss' else 'exp_dec'
    w0 = pulse(okind, Tp, t, tr1)*np.sqrt(1-eps); w1 = pulse(okind, Tp, t, tr2)*np.sqrt(1-eps)
    return integrar(t, np.full(len(t), ke_in), G1, G2, u0, u1, w0, w1, chi), sep
def dos_etapas(kind, chi, Twait):
    win = (-300, 300) if kind == 'gauss' else (-800, 2); sep = (win[1]-win[0])+dpi+100.0; Gpi = np.pi/(2*dpi); Wr = 600.0
    tc1, tc2 = 0.0, sep; tr1 = tc2+win[1]+dpi+Twait+50.0; tr2 = tr1+dpi+Wr+100.0; t = np.arange(-9*Tp, tr2+dpi+Wr+300, dt)
    ke_t = np.zeros(len(t)); G1 = np.zeros(len(t)); G2 = np.zeros(len(t)); w0 = np.zeros(len(t)); w1 = np.zeros(len(t))
    for tc, Gc in ((tc1, G1), (tc2, G2)):
        ke_t[(t >= tc+win[0]) & (t <= tc+win[1])] = ke_in; Gc[(t > tc+win[1]) & (t <= tc+win[1]+dpi)] = Gpi
    for tr, Gc, w in ((tr1, G1, w0), (tr2, G2, w1)):
        Gc[(t >= tr) & (t < tr+dpi)] = Gpi; op = (t >= tr+dpi) & (t <= tr+dpi+Wr); ke_t[op] = ke_in; w[op] = np.exp(-(ke_in+ki)*(t[op]-(tr+dpi))/2)
    w0 = w0/np.sqrt(np.sum(w0**2)*dt)*np.sqrt(1-eps); w1 = w1/np.sqrt(np.sum(w1**2)*dt)*np.sqrt(1-eps)
    u0 = pulse(kind, Tp, t, tc1)*np.sqrt(1-eps); u1 = pulse(kind, Tp, t, tc2)*np.sqrt(1-eps)
    return integrar(t, ke_t, G1, G2, u0, u1, w0, w1, chi), sep
def job(args):
    prot, kind, chi, Tw = args
    if prot == 'directa': r, sep = directo(kind, chi, Tw)
    elif prot == 'dos_etapas': r, sep = dos_etapas(kind, chi, Tw)
    else: sep = (600 if kind == 'gauss' else 800)+dpi+100.0; r = pasivo(kind, Tw, sep)
    return args, sep, r
if __name__ == '__main__':
    jobs = [(p, k, c, T) for p in ('directa', 'dos_etapas') for k in ('gauss', 'exp_gro') for c in (1.0, 0.9, 0.0) for T in (0.0, 2000.0, 8000.0)]
    jobs += [('pasivo', k, 1.0, T) for k in ('gauss', 'exp_gro') for T in (0.0, 2000.0, 8000.0)]
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with Pool(nproc) as pool:
        print('%-10s %-8s %-4s %-6s %-5s | %-6s %-6s %-6s %-6s %-6s'%('protocolo', 'entrada', 'chi', 'espera', 'sep', 'P0', 'P1', 'V', 's', 'F_h'), flush=True)
        for args, sep, r in pool.imap(job, jobs):
            p, k, c, T = args; print('%-10s %-8s %-4.1f %-6.0f %-5.0f | %-6.3f %-6.3f %-6.3f %-6.3f %-6.3f'%(p, k, c, T, sep, r['P0'], r['P1'], r['V'], r['s'], r['Fh']), flush=True)
