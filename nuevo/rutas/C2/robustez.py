import numpy as np
from captura import pulse, optimize_G, ideal_G, simulate
import qubit
ke, ki, Gmax, dt = 1e-2, 1e-3, 0.012, 2.0
Tp = 100.0; t = np.arange(-9*Tp, 9*Tp, dt); u = pulse('gauss', Tp, t, 0.0); Gi, _ = ideal_G(u, t, ke+ki)
G, e0 = optimize_G(u, t, ke, ki, Gmax, G0=Gi); print('referencia: eta=%.3f'%e0)
print('-- error de tiempo de llegada (paquete corrido respecto al control), en unidades 1/wb')
for dtau in (-60, -30, -15, 0, 15, 30, 60): print('   dt=%+4d  eta=%.3f'%(dtau, simulate(G, pulse('gauss', Tp, t, dtau), t, ke, ki)))
print('-- escala de amplitud del control (Om)')
for sc in (0.8, 0.9, 1.0, 1.1, 1.2): print('   x%.1f  eta=%.3f'%(sc, simulate(np.clip(G*sc, -1, 1), u, t, ke, ki)))
print('-- desintonia paquete-cavidad (detuning dc, en unidades de kappa=%.3f)'%(ke+ki))
for dc in (-0.01, -0.005, 0, 0.005, 0.01): print('   dc=%+.3f (=%+.2f kappa)  eta=%.3f'%(dc, dc/(ke+ki), simulate(G, u, t, ke, ki, dc=dc)))
print('-- compensacion imperfecta del corrimiento Stark por el laser (fraccion no compensada f; shift = f*Delta_I/g^2 * G^2, Delta_I=1.35, g=0.057)')
for f in (0.0, 0.02, 0.05, 0.1): print('   f=%.2f  eta=%.3f'%(f, simulate(G, u, t, ke, ki, shift=f*1.35/0.057**2)))
print('-- forma del paquete (control optimizado para gaussiano, paquete distinto con misma energia y ancho comparable)')
for kind in ('gauss', 'sech', 'exp_dec', 'exp_gro'): print('   %-8s eta=%.3f'%(kind, simulate(G, pulse(kind, Tp, t, 0.0), t, ke, ki)))
print('-- separacion entre paquetes (captura directa, gauss Tp=100): P0,P1,V_c')
tt = np.arange(-9*Tp, 2000, dt)
for sep in (450., 600., 890.):
    t2 = np.arange(-9*Tp, sep+9*Tp+300, dt); u0 = pulse('gauss', Tp, t2, 0.0); u1 = pulse('gauss', Tp, t2, sep)
    G1 = np.interp(t2, t, G, left=0, right=0); G2 = np.interp(t2, t+sep, G, left=0, right=0)
    m = qubit.metrics(*qubit.propagate(t2, u0, u1, np.full(len(t2), ke), G1, G2)[:3])
    print('   sep=%4d  P0=%.3f P1=%.3f V_c=%.3f  leak=%.1e  F_h~(1+V_c^2)/2=%.3f'%(sep, m['P0'], m['P1'], m['V'], max(m['leak0'], m['leak1']), (1+m['V']**2)/2), flush=True)
