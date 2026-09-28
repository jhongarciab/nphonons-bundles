"""Mapa de validez de la figura de mérito con el modelo efectivo estático (qubit explícito, el confirmado en P9).

Marco re-sintonizado (oscilador a ω_p/2, qubit a ω_p = ω_q = 2(ω − 4g_x²/3ω)):
    H = χ n|e><e| + (g_x²/ω)(|e><e| − |g><g|/3) − G(σ₊a² + h.c.) + Ω(σ₊ + σ₋),   χ = (8/3)g_x²/ω,  G = 2g_xg_z/ω
    L = κD[σ₋] + Γ₁⁻D[a] + Γ₁⁺D[a†],  Γ₁⁻ = g_x²κ/(ω² + κ²/4),  Γ₁⁺ = g_x²κ/(9ω² + κ²/4),  Ω = |α|²G.
Tasa de paridad espectral (modo real con mayor |Tr(PR)|, sin el estacionario), |α_eff²| = |⟨a²⟩| estacionario,
κ₁ por inversión de C1, κ₂ = 4G²/κ, D = (κ₁/κ₂)(g_z/κ)²/(5/72).
Unidades: κ = 1 en la rejilla; para comparar con el modelo completo se usan sus parámetros físicos (κ = 0.03).
Uso: python calc_mapa_validez.py rejilla [gz_k] [N]      -> data/mapa_validez/rejilla_gz<gz>_N<N>.npz
     python calc_mapa_validez.py comparar [N]           -> compara con los puntos del modelo completo
"""
import sys, os, glob
import numpy as np
import qutip as qt
from multiprocessing import Pool
import comun as C

AL2 = 4.0


def efectivo(gx, w, gz, kap, al2=AL2, N=20):
    G = 2 * gx * gz / w; Om = al2 * G
    gm = gx**2 * kap / (w**2 + kap**2 / 4); gp = gx**2 * kap / (9 * w**2 + kap**2 / 4)
    a = qt.tensor(qt.destroy(N), qt.qeye(2)); sm = qt.tensor(qt.qeye(N), qt.sigmam())
    Pe = sm.dag() * sm; Pg = 1 - Pe
    H = (8 * gx**2 / (3 * w)) * a.dag() * a * Pe + (gx**2 / w) * (Pe - Pg / 3) \
        - G * (sm.dag() * a * a + sm * a.dag() * a.dag()) + Om * (sm.dag() + sm)
    L = qt.liouvillian(H, [np.sqrt(kap) * sm, np.sqrt(gm) * a, np.sqrt(gp) * a.dag()]).full()
    lam, R = np.linalg.eig(L)
    D = 2 * N
    tasas = -lam.real
    orden = np.argsort(tasas)
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2)).full()
    borde = np.repeat(np.arange(N), 2) > N - 6
    k0 = orden[0]
    M = R[:, k0].reshape(D, D, order='F'); M = M / np.trace(M); M = (M + M.conj().T) / 2
    cand = []
    for k in orden[1:12]:
        Mk = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(Mk)
        pb = 1 - np.linalg.norm(Mk[np.ix_(~borde, ~borde)])**2 / nrm**2
        if abs(lam[k].imag) < 1e-9 * max(1, abs(lam[k].real)) + 1e-12 and pb < 0.5:
            cand.append((abs(np.trace(P @ Mk)) / nrm, tasas[k]))
    gpf = max(cand)[1]
    a2 = abs(np.trace((a * a).full() @ M))
    r = gp / gm
    k1 = gpf * (1 + r) / (2 * (a2 * (1 + r) + r))
    k2 = 4 * G**2 / kap
    Dval = k1 / k2 * (gz / kap)**2 / (5 / 72)
    return dict(gpf=gpf, a2=a2, k1=k1, k2=k2, D=Dval, val=np.array(C.validar(M)), borde=max(pb for _ in [0]))


def punto_rejilla(args):
    k2k, cchi, gzk, N = args
    gxw = np.sqrt(k2k / (16 * gzk**2))
    wk = (cchi / AL2) / ((8 / 3) * gxw**2)
    r = efectivo(gxw * wk, wk, gzk, 1.0, N=N)
    return [k2k, cchi, gzk, wk, gxw, r['gpf'], r['a2'], r['D'], *r['val']]


def rejilla(gzk=7.0, N=20, n=25):
    ks = np.geomspace(1e-3, 3, n); cs = np.geomspace(1e-3, 3, n)
    tareas = [(k, c, gzk, N) for k in ks for c in cs]
    with Pool(int(os.environ.get('NPROC', 6))) as p:
        res = p.map(punto_rejilla, tareas)
    f = os.path.join(C.DATA, 'mapa_validez', f'rejilla_gz{gzk:g}_N{N}.npz')
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, res=np.array(res), ks=ks, cs=cs, gzk=gzk, N=N)
    return f


def comparar(N=20):
    import principal_fig2 as PF2
    T = PF2.cargar()
    print("g_x/ω   ω/κ  g_z/κ  |α|²  κ₂/κ   χ|α|²/κ  D_completo  D_efectivo  dif")
    filas = []
    for r in T:
        gx, w, gzk, al2 = r[0], r[1], r[2], r[3]
        e = efectivo(gx, w, gzk * 0.03, 0.03, al2=al2, N=max(N, 26 if al2 > 4 else N))
        Dc = r[11] / (5 / 72)
        filas.append([gx / w, w / 0.03, gzk, al2, r[6], r[7], r[8], Dc, e['D']])
        print(f"{gx/w:.4f} {w/0.03:5.0f} {gzk:5.1f} {al2:4.0f} {r[6]:6.3f} {r[7]:8.3f}  {Dc:.4f}    {e['D']:.4f}   {e['D']-Dc:+.4f}   P_c={r[8]:.3f}")
    np.savetxt(os.path.join(C.DATA, 'mapa_validez', 'comparacion.csv'), np.array(filas), delimiter=',', comments='',
               header='g_x/omega, omega/kappa, g_z/kappa, |alpha|^2, kappa_2/kappa, chi|alpha|^2/kappa, P_c full, D full, D effective')


if __name__ == '__main__':
    os.makedirs(os.path.join(C.DATA, 'mapa_validez'), exist_ok=True)
    if sys.argv[1] == 'rejilla':
        gzk = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        print(rejilla(gzk, N))
    else:
        comparar(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
