"""Figura central (b): tasa de confinamiento del modelo mínimo con el qubit decayendo a través de un filtro.
H = G[(a² − α²)σ₊ + h.c.] + J(σ₊b + σ₋b†),  L = κ_f D[b]  (qubit sin decaimiento directo), 4J²/κ_f = κ = 1,
|α|² = 4, filtro resonante con el qubit en el marco rotante (centrado en 2ω), N_f = 2.
Dimensión 2N·N_f: la diagonalización densa es cara, así que la dinámica se obtiene con mesolve (Liouvilliano
estático disperso) desde |0>|g>|0_f> en una rejilla logarítmica de tiempos hasta t_max = 80/Δ_plano(κ₂).
Tasa dinámica: misma definición que calc_minimo.tasa_dinamica (cola de |P_c(t) − P_c(t_max)|).
Guarda data/minimo_filtro/k<κ₂/κ>_kf<κ_f>_N<N>.npz.  Uso: python calc_minimo_filtro.py k2k kf N [Nf] [--rerun]
"""
import sys, os, glob
import numpy as np
import qutip as qt
import comun as C
import calc_minimo as CM

AL2 = 4.0


def delta_plano(k2k):
    T = []
    for f in glob.glob(os.path.join(C.DATA, 'minimo', 'k*_N24.npz')):
        z = np.load(f); T.append((float(z['k2k']), CM.tasa_dinamica(z['t'], z['Pc_din'][0])))
    T = np.array(sorted(T))
    return np.exp(np.interp(np.log(k2k), np.log(T[:, 0]), np.log(T[:, 1])))


def punto(k2k, kf, N, Nf=2, rerun=False):
    f = os.path.join(C.DATA, 'minimo_filtro', f'k{k2k:.6g}_kf{kf:g}_N{N}_Nf{Nf}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    G = np.sqrt(k2k) / 2; J = np.sqrt(kf) / 2
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(Nf)]
    op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm, b = op(0, qt.destroy(N)), op(1, qt.sigmam()), op(2, qt.destroy(Nf))
    H = G * ((a * a - AL2) * sm.dag() + (a.dag() * a.dag() - AL2) * sm) + J * (sm.dag() * b + sm * b.dag())
    tmax = 80 / delta_plano(k2k)
    t = np.concatenate([[0], np.geomspace(1e-2, tmax, 600)])
    rho0 = qt.ket2dm(qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(Nf, 0)))
    al = np.sqrt(AL2)
    ca, cb = qt.coherent(N, al), qt.coherent(N, -al)
    Pc = qt.tensor((ca + cb).unit().proj() + (ca - cb).unit().proj(), qt.qeye(2), qt.qeye(Nf))
    r = qt.mesolve(H, rho0, t, [np.sqrt(kf) * b], e_ops={'Pc': Pc}, options=dict(atol=1e-11, rtol=1e-9, nsteps=10**7))
    serie = np.real(r.e_data['Pc'])
    res = dict(k2k=k2k, kf=kf, N=N, Nf=Nf, t=t, Pc=serie, conf=CM.tasa_dinamica(t[1:], serie[1:]))
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    k2k, kf, N = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    Nf = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else 2
    r = punto(k2k, kf, N, Nf, '--rerun' in sys.argv)
    dp = delta_plano(k2k)
    print(f"κ₂/κ={k2k:.4g} κ_f={kf} N={N} Nf={Nf}: conf={float(r['conf']):.5e}  plano={dp:.5e}  filtro/plano={float(r['conf'])/dp:.4f}")
