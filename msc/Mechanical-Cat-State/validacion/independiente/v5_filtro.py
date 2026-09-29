"""V5: baño filtrado. El qubit decae a través de un modo filtro b (frecuencia wf = 2w, ancho κ_f, 4J²/κ_f = κ).

H = w a†a + (wq/2)σz + (a+a†)(gx σx + gz σz) + wf b†b + J(σ+ b + σ- b†) [+ drive Ω(σ+ e^{-iwp t} + h.c.)]
Disipación: κ_f D[b] (filtro) o κ D[σ-] (baño plano, sin filtro).
Parámetros (Tarea 43): Ma (w=6, g=0.3, θ=π/4, κ=0.03), wp = wq = 2(w - 4gx²/(3w)), Ω = |α|²G.

Modo 'estatico' (sin drive): Liouvilliano estático en el laboratorio.
   - autovalor real no nulo más lento = relajación de población del oscilador = Γ₁⁻ - Γ₁⁺
   - n̄ estacionario menos n virtual del fundamental exacto de H: Γ₁⁺ = (Γ₁⁻ - Γ₁⁺)·(n̄ - n_virt)
Modo 'floquet' (con drive): propagador de un período T_p = 2π/wp; tasa del modo de paridad (λ real ≈ +1).
Uso: python v5_filtro.py estatico kf N Nf       (kf = 0 -> baño plano)
     python v5_filtro.py floquet  kf N Nf al2 [salida.npz]
"""
import sys, time
import numpy as np
import qutip as qt
import scipy.sparse.linalg as sla

w, g, th, KAP = 6.0, 0.3, np.pi / 4, 0.03
import os
gx, gz = -g * np.sin(th), g * np.cos(th)
GZ_EST = 0.0 if os.environ.get("GZ0") else gz   # g_z en el modo estático (GZ0=1 lo apaga)
G = 2 * gx * gz / w
wp = 2 * (w - 4 * gx**2 / (3 * w)); wq = wp; wf = 2 * w
OPTS = dict(atol=1e-12, rtol=1e-10, nsteps=10**6)


def keff(d, kf):
    return KAP if kf == 0 else KAP * kf**2 / (4 * d**2 + kf**2)


def predic(kf):
    gm = gx**2 * keff(w, kf) / w**2
    gp = gx**2 * keff(3 * w, kf) / (9 * w**2)
    return gm, gp


def modelo(kf, N, Nf, gzz=None):
    gzz = gz if gzz is None else gzz
    if kf == 0:
        Nf = 1
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(Nf)]
    op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm, sz, sx = op(0, qt.destroy(N)), op(1, qt.sigmam()), op(1, qt.sigmaz()), op(1, qt.sigmax())
    H0 = w * a.dag() * a + 0.5 * wq * sz + (a + a.dag()) * (gx * sx + gzz * sz)
    if kf == 0:
        return H0, a, sm, [np.sqrt(KAP) * sm], Nf
    b = op(2, qt.destroy(Nf))
    J = np.sqrt(KAP * kf) / 2
    H0 = H0 + wf * b.dag() * b + J * (sm.dag() * b + sm * b.dag())
    return H0, a, sm, [np.sqrt(kf) * b], Nf


def estatico(kf, N, Nf):
    H0, a, sm, c, Nf = modelo(kf, N, Nf, GZ_EST)
    L = qt.liouvillian(H0, c)
    # diagonalización densa (dimensión ≤ 48² ); eigs con sigma=0 perdía el modo real entre los complejos
    ev = np.linalg.eigvals(L.full())
    real = sorted([-e.real for e in ev if abs(e.imag) < 1e-7 and -e.real > 1e-12])
    r = real[0]
    rho = qt.steadystate(H0, c, method='direct')
    nbar = qt.expect(a.dag() * a, rho)
    nvirt = qt.expect(a.dag() * a, H0.groundstate()[1])
    gp = r * (nbar - nvirt); gm = r + gp
    pm, pp = predic(kf)
    print(f"estatico kf={kf} N={N} Nf={Nf}: Γ⁻-Γ⁺={r:.5e} (pred {pm-pp:.5e}, {r/(pm-pp):.4f})  n̄={nbar:.5e} n_virt={nvirt:.3e}")
    print(f"   Γ₁⁻={gm:.5e} pred {pm:.5e} razón {gm/pm:.4f} | Γ₁⁺={gp:.5e} pred {pp:.5e} razón {gp/pp:.4f} | κ₁=Γ⁻+Γ⁺={gm+gp:.5e} pred {pm+pp:.5e} razón {(gm+gp)/(pm+pp):.4f}")
    print(f"   reales lentos: {[f'{x:.3e}' for x in real[:5]]}")
    M = rho.full()
    print(f"   validación ρ_ss: |Tr-1|={abs(np.trace(M)-1):.1e} ‖ρ-ρ†‖={np.linalg.norm(M-M.conj().T):.1e} mín eig={np.linalg.eigvalsh((M+M.conj().T)/2).min():.1e}")


def floquet(kf, N, Nf, al2, out):
    H0, a, sm, c, Nf = modelo(kf, N, Nf)
    Om = al2 * G
    H = [H0, [Om * sm.dag(), lambda t: np.exp(-1j * wp * t)], [Om * sm, lambda t: np.exp(1j * wp * t)]]
    Tp = 2 * np.pi / wp
    t0 = time.time()
    U = qt.propagator(H, Tp, c, options=OPTS).full()
    tprop = time.time() - t0
    D = 2 * N * Nf
    lam, R = np.linalg.eig(U)
    rate = -np.log(np.abs(lam)) / Tp
    idx = np.argsort(rate)[:12]
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2), qt.qeye(Nf)).full()
    nidx = np.repeat(np.arange(N), 2 * Nf)
    borde = nidx > N - 6
    filas = []
    for k in idx:
        M = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(M)
        pb = 1 - np.linalg.norm(M[np.ix_(~borde, ~borde)])**2 / nrm**2
        filas.append((k, lam[k], rate[k], abs(np.trace(P @ M)) / nrm, pb))
    cand = [f for f in filas if f[2] > 1e-12 and f[1].real > 0 and f[4] < 0.5]
    kp = max(cand, key=lambda f: f[3]); r = kp[2]
    rho = qt.Qobj(R[:, idx[0]].reshape(D, D, order='F'), dims=[[N, 2, Nf], [N, 2, Nf]])
    rho = rho / rho.tr(); rho = (rho + rho.dag()) / 2
    a2 = abs(qt.expect(a * a, rho))
    al = np.sqrt(complex(Om / G)); ca, cb = qt.coherent(N, al), qt.coherent(N, -al)
    Pc = qt.expect(qt.tensor((ca + cb).unit().proj() + (ca - cb).unit().proj(), qt.qeye(2), qt.qeye(Nf)), rho)
    pm, pp = predic(kf)
    A = lambda x: 2 * x * (pm + pp); B = lambda x: 2 * (pm * x + pp * (x + 1))
    M = rho.full()
    print(f"floquet kf={kf} N={N} Nf={Nf} |α|²={al2} t_prop={tprop:.0f}s")
    for f in filas[:6]:
        print(f"    {f[1].real:+.9f}{f[1].imag:+.2e}i r={f[2]:.4e} par={f[3]:.3f} borde={f[4]:.1e}" + ("  <- paridad" if f[0] == kp[0] else ""))
    print(f"  r_par={r:.5e} |⟨a²⟩|={a2:.4f} P_c={Pc:.5f}  κ₁ pred={pm+pp:.5e}")
    print(f"  razón A(nom)={r/A(al2):.4f} A(|a2|)={r/A(a2):.4f} B(nom)={r/B(al2):.4f} B(|a2|)={r/B(a2):.4f}")
    print(f"  validación ρ_ss: |Tr-1|={abs(np.trace(M)-1):.1e} ‖ρ-ρ†‖={np.linalg.norm(M-M.conj().T):.1e} mín eig={np.linalg.eigvalsh(M).min():.1e}")
    np.savez(out, kf=kf, N=N, Nf=Nf, al2=al2, r=r, a2=a2, Pc=Pc, pm=pm, pp=pp, lam=lam[idx], tprop=tprop)


if __name__ == '__main__':
    modo, kf, N, Nf = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    if modo == 'estatico':
        estatico(kf, N, Nf)
    else:
        al2 = float(sys.argv[5])
        floquet(kf, N, Nf, al2, sys.argv[6] if len(sys.argv) > 6 else f"res_v5/fl_kf{kf}_N{N}_Nf{Nf}.npz")
