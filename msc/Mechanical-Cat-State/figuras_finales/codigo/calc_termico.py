"""Figura térmica (P7): sesgo η = γ_pf/γ_bf en función de x = h f_q/(k_B T), con el modelo efectivo estático
(qubit explícito, validado en P9; filtro explícito como en calc_minimo_filtro) y baños térmicos.

Definición de η (la de la Tarea 37, validacion/tarea37_worker.py): η = γ_pf/γ_bf, con γ_pf la tasa del modo lento
con mayor traslape con la paridad P = e^{iπa†a} y γ_bf la del modo con mayor traslape con a (modo de pozo).

Marco rotante re-sintonizado (oscilador a ω_p/2, qubit a ω_p = ω_q), unidades κ = 1:
  H = χ n|e><e| + (g_x²/ω)(|e><e| − |g><g|/3) − G(σ₊a² + h.c.) + Ω(σ₊ + σ₋) [+ J(σ₊b + σ₋b†)],
  χ = (8/3)g_x²/ω, G = 2g_xg_z/ω, Ω = |α|²G, 4J²/κ_f = κ.
Baños (convención de la Tarea 37: el baño del qubit/filtro está a la frecuencia del qubit, f_q):
  plano:   κ[(n_q+1)D[σ₋] + n_q D[σ₊]]
  filtro:  κ_f[(n_q+1)D[b] + n_q D[b†]]   (qubit sin decaimiento directo)
  canales de un fonón mediados por el qubit: Γ₁⁻[(n_q+1)D[a] + n_q D[a†]] + Γ₁⁺[(n_q+1)D[a†] + n_q D[a]]
     (Γ₁± planos o filtrados; mismo baño térmico n_q, igual que en el modelo completo con un solo baño de Lindblad)
  pérdida intrínseca del oscilador: γ[(n_m+1)D[a] + n_m D[a†]], n_m a f_q/2.
n_q = 1/(e^x − 1), n_m = 1/(e^{x/2} − 1).
Espectro: eigs disperso (shift-invert cerca de 0) de los 12 modos más lentos. Validaciones de ρ_ss ANTES de hermitizar.
Uso como módulo: punto(x, k2k, gzk, filtro, gam, N, Nf=2, wk=200, al2=4, kfw=0.05)
"""
import os, sys
import numpy as np
import qutip as qt
import scipy.sparse.linalg as sla
import comun as C


def punto(x, k2k, gzk, filtro, gam, N, Nf=2, wk=200.0, al2=4.0, kfw=0.05, rerun=False):
    f = os.path.join(C.DATA, 'termico', f'x{x:.5g}_k{k2k:.4g}_gz{gzk:.4g}_f{int(filtro)}_g{gam:g}_N{N}_Nf{Nf}_w{wk:g}_a{al2:g}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    nq = 1 / np.expm1(x); nm = 1 / np.expm1(x / 2)
    w = wk; gxw = np.sqrt(k2k / (16 * gzk**2)); gx = gxw * w; gz = gzk
    G = 2 * gx * gz / w; Om = al2 * G; chi = 8 * gx**2 / (3 * w)
    kf = kfw * w
    ke = (lambda d: kf**2 / (4 * d**2 + kf**2)) if filtro else (lambda d: 1.0)
    Gm = gx**2 * ke(w) / w**2; Gp = gx**2 * ke(3 * w) / (9 * w**2)
    nf = Nf if filtro else 1
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(nf)]
    op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm = op(0, qt.destroy(N)), op(1, qt.sigmam())
    Pe = sm.dag() * sm; Pg = 1 - Pe
    H = chi * a.dag() * a * Pe + (gx**2 / w) * (Pe - Pg / 3) - G * (sm.dag() * a * a + sm * a.dag() * a.dag()) + Om * (sm.dag() + sm)
    c = []
    if filtro:
        b = op(2, qt.destroy(nf)); J = np.sqrt(kf) / 2
        H = H + J * (sm.dag() * b + sm * b.dag())
        c += [np.sqrt(kf * (nq + 1)) * b, np.sqrt(kf * nq) * b.dag()]
    else:
        c += [np.sqrt(nq + 1) * sm, np.sqrt(nq) * sm.dag()]
    down = Gm * (nq + 1) + Gp * nq + gam * (nm + 1)
    up = Gm * nq + Gp * (nq + 1) + gam * nm
    c += [np.sqrt(down) * a, np.sqrt(up) * a.dag()]
    L = qt.liouvillian(H, c).data.as_scipy().tocsc()
    lam, R = sla.eigs(L, k=12, sigma=-1e-10, which='LM', tol=1e-14, maxiter=100000)
    orden = np.argsort(-lam.real); lam, R = lam[orden], R[:, orden]
    D = 2 * N * nf
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2), qt.qeye(nf)).full()
    A = a.full()
    borde = np.repeat(np.arange(N), 2 * nf) > N - 6
    modos = []
    for k in range(len(lam)):
        Mk = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(Mk)
        pb = 1 - np.linalg.norm(Mk[np.ix_(~borde, ~borde)])**2 / nrm**2
        modos.append([-lam[k].real, lam[k].imag, abs(np.trace(P @ Mk)) / nrm, abs(np.trace(A @ Mk)) / nrm, pb])
    modos = np.array(modos)
    # estado estacionario (modo 0) y validaciones ANTES de hermitizar
    M = R[:, 0].reshape(D, D, order='F'); M = M / np.trace(M)
    herm_cruda = np.linalg.norm(M - M.conj().T)
    Mh = (M + M.conj().T) / 2
    tr_err = abs(np.trace(Mh) - 1); mineig = np.linalg.eigvalsh(Mh).min()
    # modos 1..3 (como la Tarea 37): paridad = mayor traslape con P; bit-flip = mayor traslape con a
    sub = modos[1:4]
    ipf = int(np.argmax(sub[:, 2])); ibf = int(np.argmax(sub[:, 3]))
    gpf, gbf = sub[ipf, 0], sub[ibf, 0]
    al = np.sqrt(complex(Om / G))
    Pc = C.proyector_codigo(N, al, 0.0)
    rc = np.einsum('iaib->ab', Mh.reshape(N, 2 * nf, N, 2 * nf).transpose(1, 0, 3, 2)) if False else None
    Pc_full = np.kron(Pc, np.eye(nf)) if nf > 1 else Pc
    pcode = np.real(np.trace(Pc_full @ Mh))
    res = dict(x=x, nq=nq, nm=nm, k2k=k2k, gzk=gzk, filtro=filtro, gam=gam, N=N, Nf=nf, wk=wk, al2=al2, kfw=kfw,
               Gm=Gm, Gp=Gp, chi_a2=chi * al2, gpf=gpf, gbf=gbf, eta=gpf / gbf, modos=modos, pcode=pcode,
               herm_cruda=herm_cruda, tr_err=tr_err, mineig=mineig, ipf=ipf, ibf=ibf)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    x, k2k, gzk = map(float, sys.argv[1:4]); filtro = int(sys.argv[4]); gam = float(sys.argv[5]); N = int(sys.argv[6])
    r = punto(x, k2k, gzk, filtro, gam, N, rerun='--rerun' in sys.argv)
    print(f"x={x} n_q={float(r['nq']):.3e} filtro={filtro} N={N}: γ_pf={float(r['gpf']):.4e} γ_bf={float(r['gbf']):.4e} η={float(r['eta']):.4g} "
          f"P_c={float(r['pcode']):.5f} herm={float(r['herm_cruda']):.1e} tr={float(r['tr_err']):.1e} mineig={float(r['mineig']):.1e}")
    for m in r['modos'][:6]:
        print(f"   tasa={m[0]:.4e} Im={m[1]:+.1e} ⟨P⟩={m[2]:.3f} ⟨a⟩={m[3]:.3f} borde={m[4]:.1e}")
