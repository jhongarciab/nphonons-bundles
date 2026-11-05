"""Mapa eta_opt(kappa_e, Tp) gaussiano, ki=1e-3, Gmax=0.012. dt adaptativo (Tp/25 acotado por 0.3/kappa). Uso: python datos_mapas.py parte total"""
import sys, numpy as np
from captura import pulse, optimize_G, ideal_G, simulate
ki, Gmax = 1e-3, 0.012
KE = np.geomspace(0.003, 0.06, 9); TP = np.geomspace(30, 300, 9)
part, tot = int(sys.argv[1]), int(sys.argv[2]); out = {}
for i, ke in enumerate(KE):
    if i % tot != part: continue
    for j, Tp in enumerate(TP):
        dt = min(Tp/25, 0.3/(ke+ki)); t = np.arange(-8*Tp, 8*Tp, dt); u = pulse('gauss', Tp, t, 0.0); Gi, _ = ideal_G(u, t, ke+ki)
        Gid = np.clip(Gi, -Gmax, Gmax); e_id = simulate(Gid, u, t, ke, ki)
        Go, e = optimize_G(u, t, ke, ki, Gmax, nn=8, G0=Gi); out[(i, j)] = (max(e, e_id), e_id)
        print(i, j, 'ke=%.4f Tp=%.0f eta_opt=%.3f eta_idealG=%.3f'%(ke, Tp, max(e, e_id), e_id), flush=True)
np.save('mapa_parte%d.npy'%part, np.array([[k[0], k[1], v[0], v[1]] for k, v in out.items()]))
