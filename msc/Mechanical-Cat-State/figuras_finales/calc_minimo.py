"""Figura principal 3: tasa de confinamiento del modelo mínimo H = G[(a² − α²)σ₊ + h.c.], κD[σ₋], |α|² = 4, κ = 1.
Liouvilliano estático, diagonalización densa L = R Λ R⁻¹.
  - Dinámica exacta: P_c(t) = Σ_k e^{λ_k t} c_k p_k desde |0>|g> y |1.3α>|g> (código |±α>, sin desplazamiento:
    el modelo mínimo no tiene g_z). Tasa dinámica = ajuste de la cola de |P_c(t) − P_c(∞)| (P_c(∞) propio de
    cada estado inicial: el espacio oscuro de 4 modos de tasa cero conserva memoria).
  - Espectral (C9): menor tasa no nula con peso de borde < 0.5; estabilidad en N, N+6, N+12.
Guarda data/minimo/k<κ₂/κ>_N<N>.npz. Uso: python calc_minimo.py k2k N [--rerun]
"""
import sys, os
import numpy as np
import qutip as qt
import comun as C

AL2 = 4.0


def punto(k2k, N, rerun=False):
    f = os.path.join(C.DATA, 'minimo', f'k{k2k:.6g}_N{N}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    G = np.sqrt(k2k) / 2                               # κ₂ = 4G²/κ, κ = 1
    al = np.sqrt(AL2)
    a = qt.tensor(qt.destroy(N), qt.qeye(2)); sm = qt.tensor(qt.qeye(N), qt.sigmam())
    H = G * ((a * a - AL2) * sm.dag() + (a.dag() * a.dag() - AL2) * sm)
    L = qt.liouvillian(H, [sm]).full()
    lam, R = np.linalg.eig(L)
    Linv = np.linalg.inv(R)
    D = 2 * N
    Pc_op = C.proyector_codigo(N, al, 0.0)
    p = np.einsum('ij,jik->k', Pc_op, R.reshape(D, D, -1, order='F'))
    borde = np.repeat(np.arange(N), 2) > N - 6
    tasas = -lam.real
    orden = np.argsort(tasas)
    ceros = int(np.sum(np.abs(lam) < 1e-9))
    modos = []
    for k in orden[:40]:
        Mk = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(Mk)
        pb = 1 - np.linalg.norm(Mk[np.ix_(~borde, ~borde)])**2 / nrm**2
        modos.append([lam[k].real, lam[k].imag, pb])
    modos = np.array(modos)
    fis = [m for m in modos if -m[0] > 1e-9 and m[2] < 0.5]
    brecha_esp = -fis[0][0] if fis else np.nan
    kets = [qt.tensor(qt.basis(N, 0), qt.basis(2, 1)), qt.tensor(qt.coherent(N, 1.3 * al), qt.basis(2, 1))]
    t = np.geomspace(1e-2, 400 / min(k2k, 1.0) * 10, 3000)
    serie = []
    for kt in kets:
        c = Linv @ qt.ket2dm(kt).full().reshape(-1, order='F')
        serie.append(np.real(np.exp(np.outer(t, lam)) @ (c * p)))
    res = dict(k2k=k2k, N=N, G=G, t=t, Pc_din=np.array(serie), brecha_esp=brecha_esp, ceros=ceros, modos=modos)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


def tasa_dinamica(t, serie):
    """Cola exponencial de |P_c(t) − P_c(∞)|: desde que el exceso cae bajo el 10% de su máximo hasta 1e3 veces
    el piso numérico (mediana del último 10%, ruido de la reconstrucción espectral) o 1e-9, lo que sea mayor."""
    e = np.abs(serie - serie[-1])
    i1 = np.argmax(e < 0.1 * e.max())
    piso = max(1e-9, 1e3 * np.median(e[int(0.9 * len(e)):-1]))
    fin = np.where(e < piso)[0]; i2 = fin[fin > i1][0] if np.any(fin > i1) else len(e) - 1
    if i2 - i1 < 10:
        return np.nan
    return -np.polyfit(t[i1:i2], np.log(e[i1:i2]), 1)[0]


if __name__ == '__main__':
    k2k, N = float(sys.argv[1]), int(sys.argv[2])
    r = punto(k2k, N, '--rerun' in sys.argv)
    td = [tasa_dinamica(r['t'], s) for s in r['Pc_din']]
    print(f"κ₂/κ={k2k:.4g} N={N}: ceros={int(r['ceros'])} brecha espectral={float(r['brecha_esp']):.5e} "
          f"dinámica={td[0]:.5e}/{td[1]:.5e}  (Δ/κ₂: {float(r['brecha_esp'])/k2k:.4f}, {td[0]/k2k:.4f}, {td[1]/k2k:.4f})")
