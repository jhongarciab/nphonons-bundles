"""Post-proceso barato: P_e promediado en un período (C5) para las series viejas de data/fig2/ (ω_q = ω_p, N = 20)
que no lo guardaron. Integra un período desde la ρ estroboscópica cacheada; guarda data/fig2/pe_<archivo>.npz.
Uso: python calc_pe_fig2viejas.py archivo.npz"""
import sys, os
import numpy as np
import qutip as qt
import comun as C

W, GZ, KAP = 6.0, 0.3 * np.cos(np.pi / 4), 0.03
f = sys.argv[1]
out = os.path.join(os.path.dirname(f), 'pe_' + os.path.basename(f))
if not os.path.exists(out) or '--rerun' in sys.argv:
    z = np.load(f); N = int(z['N']); wp = float(z['wp'])
    a, sm, sz, sx = C.ops(N)
    H = C.hamiltoniano(N, W, float(z['wq']), float(z['gx']), GZ, float(z['Om']), wp)
    ts = np.linspace(0, 2 * np.pi / wp, 41)
    r = qt.mesolve(H, qt.Qobj(z['rho'], dims=[[N, 2], [N, 2]]), ts, [np.sqrt(KAP) * sm], e_ops={'Pe': sm.dag() * sm}, options=C.OPTS)
    np.savez(out, Pe_prom=np.mean(np.real(r.e_data['Pe'][:-1])))
