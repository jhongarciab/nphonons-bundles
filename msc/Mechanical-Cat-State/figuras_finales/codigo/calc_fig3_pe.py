"""Fig. 3, post-proceso barato: P_e promediado en un período (C5) desde la ρ estacionaria cacheada.
Integra un período T_p con 40 muestras (mesolve) y guarda data/fig3/pe_<clave>.npz. Uso: python calc_fig3_pe.py archivo.npz"""
import sys, os
import numpy as np
import qutip as qt
import comun as C

f = sys.argv[1]
out = os.path.join(os.path.dirname(f), 'pe_' + os.path.basename(f))
if not os.path.exists(out) or '--rerun' in sys.argv:
    z = np.load(f); N = int(z['N'])
    a, sm, sz, sx = C.ops(N)
    H = C.hamiltoniano(N, float(z['w']), float(z['wp']), float(z['gx']), float(z['gz']), float(z['Om']), float(z['wp']))
    ts = np.linspace(0, 2 * np.pi / float(z['wp']), 41)
    r = qt.mesolve(H, qt.Qobj(z['rho'], dims=[[N, 2], [N, 2]]), ts, [np.sqrt(0.03) * sm], e_ops={'Pe': sm.dag() * sm}, options=C.OPTS)
    pe = np.real(r.e_data['Pe'])
    np.savez(out, Pe_prom=pe[:-1].mean(), Pe_min=pe.min(), Pe_max=pe.max(), Pe_estrob=pe[0])
    print(os.path.basename(f), pe[:-1].mean())
