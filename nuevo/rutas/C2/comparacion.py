import numpy as np
from captura import pulse, optimize_G, ideal_G
import qubit
ke_in, ki, Gmax, dt, sep = 1e-2, 1e-3, 0.012, 2.0, 700.0
def correr(kind, Tp, win):
    sep = (win[1]-win[0]) + 190.0 + 100.0
    tc1, tc2 = 0.0, sep; t = np.arange(-9*Tp, tc2+9*Tp+300, dt)
    u0 = pulse(kind, Tp, t, tc1); u1 = pulse(kind, Tp, t, tc2)
    ts = np.arange(-9*Tp, 9*Tp, dt); us = pulse(kind, Tp, ts, 0.0); Gi, _ = ideal_G(us, ts, ke_in+ki)
    Gs, e1 = optimize_G(us, ts, ke_in, ki, Gmax, G0=Gi)
    G1 = np.interp(t, ts+tc1, Gs, left=0, right=0); G2 = np.interp(t, ts+tc2, Gs, left=0, right=0)
    out = {'individual optimizada': e1}
    out['directa'] = qubit.metrics(*qubit.propagate(t, u0, u1, np.full(len(t), ke_in), G1, G2)[:3])
    dpi = 190.0; Gpi = np.pi/(2*dpi); ke_t = np.zeros(len(t)); G1b = np.zeros(len(t)); G2b = np.zeros(len(t))
    for tc, Gc in ((tc1, G1b), (tc2, G2b)):
        w = (t >= tc+win[0]) & (t <= tc+win[1]); ke_t[w] = ke_in
        wr = (t > tc+win[1]) & (t <= tc+win[1]+dpi); Gc[wr] = Gpi
    out['dos conmutable'] = qubit.metrics(*qubit.propagate(t, u0, u1, ke_t, G1b, G2b)[:3])
    out['dos puerto fijo'] = qubit.metrics(*qubit.propagate(t, u0, u1, np.full(len(t), ke_in), G1b, G2b)[:3])
    return out
cases = [('gauss', 60., (-180, 180)), ('gauss', 100., (-300, 300)), ('gauss', 150., (-450, 450)), ('exp_gro', 100., (-800, 2)), ('exp_gro', 200., (-1600, 2))]
print('separacion entre paquetes = ventana de carga + 190 (pi) + 100 (guarda); ke=%.0e ki=%.0e Gmax=%.3f'%(ke_in, ki, Gmax))
print('%-8s %-5s | %-10s | %-26s | %-26s | %-26s'%('forma', 'Tp', 'indiv.opt', 'directa (P0,P1,V)', 'dos etapas conmutable', 'dos etapas puerto fijo'))
for kind, Tp, win in cases:
    o = correr(kind, Tp, win); f = lambda m: '%.3f %.3f  V=%.3f'%(m['P0'], m['P1'], m['V'])
    print('%-8s %-5.0f | %-10.3f | %-26s | %-26s | %-26s'%(kind, Tp, o['individual optimizada'], f(o['directa']), f(o['dos conmutable']), f(o['dos puerto fijo'])), flush=True)
