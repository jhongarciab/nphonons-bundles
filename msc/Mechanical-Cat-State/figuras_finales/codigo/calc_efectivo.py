"""P9(b): modelo efectivo estático (buffer explícito, orden g²) en el marco re-sintonizado (oscilador a ω_p/2,
qubit a ω_p, ω_p = ω_q = 2(ω − 4g_x²/3ω)), derivado de R1:
    H = χ n |e><e| + (g_x²/ω)(|e><e| − |g><g|/3) − G(σ₊a² + h.c.) + Ω(σ₊ + σ₋),   χ = 8g_x²/(3ω),
con κD[σ₋]. El término χ n|e><e| (corrimiento del qubit dependiente de n) se incluye o se quita (flag chi=0/1).
Tasa de confinamiento dinámica desde |0>|g> (misma definición que calc_minimo), código |±α>, α² = Ω/G.
Uso: python calc_efectivo.py gx w gz_sobre_kappa al2 chi N [--rerun]  -> data/efectivo/*.npz
"""
import sys, os
import numpy as np
import qutip as qt
import comun as C
import calc_minimo as CM

KAP = 0.03


def punto(gx, w, gzk, al2, chi, N, rerun=False):
    f = os.path.join(C.DATA, 'efectivo', f'gx{gx}_w{w}_gz{gzk}_al{al2}_chi{chi}_N{N}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    gz = gzk * KAP; G = 2 * gx * gz / w; Om = al2 * G
    a = qt.tensor(qt.destroy(N), qt.qeye(2)); sm = qt.tensor(qt.qeye(N), qt.sigmam())
    Pe = sm.dag() * sm; Pg = 1 - Pe
    H = (gx**2 / w) * (Pe - Pg / 3) - G * (sm.dag() * a * a + sm * a.dag() * a.dag()) + Om * (sm.dag() + sm)
    if chi:
        H = H + (8 * gx**2 / (3 * w)) * a.dag() * a * Pe
    L = qt.liouvillian(H, [np.sqrt(KAP) * sm]).full()
    lam, R = np.linalg.eig(L); Linv = np.linalg.inv(R)
    D = 2 * N
    al = np.sqrt(complex(Om / G))
    Pc_op = C.proyector_codigo(N, al, 0.0)
    p = np.einsum('ij,jik->k', Pc_op, R.reshape(D, D, -1, order='F'))
    kap2 = 4 * G**2 / KAP
    t = np.geomspace(1e-1, 400 / kap2, 3000)
    c = Linv @ qt.ket2dm(qt.tensor(qt.basis(N, 0), qt.basis(2, 1))).full().reshape(-1, order='F')
    serie = np.real(np.exp(np.outer(t, lam)) @ (c * p))
    res = dict(gx=gx, w=w, gzk=gzk, al2=al2, chi=chi, N=N, kap2=kap2, t=t, Pc=serie, conf=CM.tasa_dinamica(t, serie))
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    gx, w, gzk, al2 = map(float, sys.argv[1:5]); chi, N = int(sys.argv[5]), int(sys.argv[6])
    r = punto(gx, w, gzk, al2, chi, N, '--rerun' in sys.argv)
    print(f"gx={gx} w={w} gz/κ={gzk} chi={chi}: κ₂/κ={float(r['kap2'])/KAP:.4f} conf={float(r['conf']):.5e}")
