"""Qubit fonónico de dos intervalos: captura directa (puerto fijo) vs carga+transferencia en dos etapas (puerto conmutable), mismo modelo, misma entrada.
Espacio de una excitacion por rama: [s0, s1, a, d1, d2] (s_k = fuente en cascada que emite el paquete u_k(t); a = cavidad; d1,d2 = estados oscuros).
Rama R=0: excitacion inicial en s0 (bin 1); rama R=1: en s1 (bin 2). Matrices rho00, rho11, rho01 con M(t) no hermitica (cascada + perdidas)
y desfase de energia comun (chi=1): d(rho_xy)/dt contiene -(gph/2)(N_x-N_y)^2 rho_xy, N=(0,0,0,1,1).
Acoplos Raman: laser 1 (resonante con d1) acopla d2 fuera de resonancia con fase exp(-i Dss t) y razon g2/g1; laser 2 analogo."""
import numpy as np
from captura import pulse, simulate, optimize_G
ki, gph = 1e-3, 4e-4; Dss = 0.25; r12 = 0.0613/0.0572; r21 = 1/r12
def gamma_s(u, dt):
    cum = np.cumsum(u**2)*dt; return u**2/np.clip(1 - cum, 1e-6, None)
def propagate(t, u0, u1, ke_t, G1, G2, gph_=gph, ki_=ki, eps=0.02):
    dt = t[1]-t[0]; N = len(t); u0 = u0*np.sqrt(1-eps); u1 = u1*np.sqrt(1-eps); gs0 = gamma_s(u0, dt); gs1 = gamma_s(u1, dt)
    mask = np.array([0, 0, 0, 1, 1.0]); Dm = (mask[:, None]-mask[None, :])**2
    def M(n, tt):
        k = ke_t[n]+ki_; m = np.zeros((5, 5), complex)
        m[0, 0] = -gs0[n]/2; m[1, 1] = -gs1[n]/2; m[2, 2] = -k/2
        m[2, 0] = -np.sqrt(gs0[n]*ke_t[n]); m[2, 1] = -np.sqrt(gs1[n]*ke_t[n])
        # Raman: laser1 (G1) resonante con d1, laser2 (G2) con d2
        a1 = G1[n]+G2[n]*r21*np.exp(1j*Dss*tt); a2 = G2[n]+G1[n]*r12*np.exp(-1j*Dss*tt)
        m[2, 3] = -1j*a1; m[3, 2] = -1j*a1; m[2, 4] = -1j*a2; m[4, 2] = -1j*a2
        return m
    psi0 = np.zeros(5, complex); psi0[0] = 1; psi1 = np.zeros(5, complex); psi1[1] = 1
    r00 = np.outer(psi0, psi0.conj()); r11 = np.outer(psi1, psi1.conj()); r01 = np.outer(psi0, psi1.conj())
    def d(R, Mn, a_, b_):  # R rho: M R + R M^dagger - (gph/2) Dm*R ; a_,b_ indican si usar Dm
        return Mn@R + R@Mn.conj().T - 0.5*gph_*Dm*R
    traj = []
    for n in range(N-1):
        t0 = t[n]; Mn0 = M(n, t0); Mn1 = M(n, t0+dt/2); Mn2 = M(n+1, t0+dt)
        def f(R, Mn): return d(R, Mn, 0, 0)
        for_each = []
        new = []
        for R in (r00, r11, r01):
            k1 = f(R, Mn0); k2 = f(R+dt/2*k1, Mn1); k3 = f(R+dt/2*k2, Mn1); k4 = f(R+dt*k3, Mn2)
            new.append(R+dt/6*(k1+2*k2+2*k3+k4))
        r00, r11, r01 = new
        if n % 5 == 0: traj.append((t[n], r00[3, 3].real, r11[4, 4].real, r00[2, 2].real))
    return r00, r11, r01, np.array(traj)
def metrics(r00, r11, r01, eps=0.02):
    nrm = 1-eps; r00 = r00/nrm; r11 = r11/nrm; r01 = r01/nrm
    p0, p1 = r00[3, 3].real, r11[4, 4].real; c = abs(r01[3, 4])
    return dict(P0=p0, P1=p1, leak0=r00[4, 4].real, leak1=r11[3, 3].real, cav0=r00[2, 2].real, cav1=r11[2, 2].real,
                V=c/np.sqrt(p0*p1) if p0*p1 > 0 else 0.0)
if __name__ == '__main__':
    ke_in, Gmax = 1e-2, 0.012; dt = 2.0
    Tp = 100.0; kind = 'gauss'; sep = 700.0
    tc1, tc2 = 0.0, sep; t = np.arange(-8*Tp, tc2+8*Tp+300, dt)
    u0 = pulse(kind, Tp, t, tc1); u1 = pulse(kind, Tp, t, tc2)
    # control directo: optimizado para un paquete (stage 1), desplazado
    ts = np.arange(-8*Tp, 8*Tp, dt); us = pulse(kind, Tp, ts, 0.0)
    from captura import ideal_G
    Gi, _ = ideal_G(us, ts, ke_in+ki)
    Gs, eta1 = optimize_G(us, ts, ke_in, ki, Gmax, G0=Gi)
    print('captura individual optimizada (gauss Tp=%d): eta=%.3f'%(Tp, eta1), flush=True)
    G1 = np.interp(t, ts+tc1, Gs, left=0, right=0); G2 = np.interp(t, ts+tc2, Gs, left=0, right=0)
    res = {}
    r = propagate(t, u0, u1, np.full(len(t), ke_in), G1, G2)
    res['directa'] = metrics(*r[:3])
    # dos etapas: puerto abierto solo en la ventana de carga [tc-3Tp, tc+3Tp]; luego pi (G cuadrado, duracion dpi) con puerto cerrado
    dpi = 190.0; Gpi = np.pi/(2*dpi)
    ke_t = np.zeros(len(t)); G1b = np.zeros(len(t)); G2b = np.zeros(len(t))
    for tc, Gc in ((tc1, G1b), (tc2, G2b)):
        win = (t >= tc-3*Tp) & (t <= tc+3*Tp); ke_t[win] = ke_in
        wr = (t > tc+3*Tp) & (t <= tc+3*Tp+dpi); Gc[wr] = Gpi
    r2 = propagate(t, u0, u1, ke_t, G1b, G2b); res['dos etapas (conmutable)'] = metrics(*r2[:3])
    # dos etapas con puerto FIJO (cavidad siempre acoplada)
    r3 = propagate(t, u0, u1, np.full(len(t), ke_in), G1b, G2b); res['dos etapas (puerto fijo)'] = metrics(*r3[:3])
    print('%-28s %-7s %-7s %-7s %-7s %-7s %-6s'%('protocolo', 'P0', 'P1', 'fuga0', 'fuga1', 'cav', 'V_c'))
    for k, m in res.items(): print('%-28s %-7.3f %-7.3f %-7.4f %-7.4f %-7.4f %-6.3f'%(k, m['P0'], m['P1'], m['leak0'], m['leak1'], m['cav0'], m['V']))
    for k, m in res.items():
        s = ((m['P0']*m['P1'])**0.5)**2   # recuperacion ~ (captura)(lectura simetrica)
        Vt = m['V']**2
        print('  %-28s s~P^2=%.3f  F_h~(1+V_c^2)/2=%.3f'%(k, s, (1+Vt)/2))
