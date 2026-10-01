"""[NO VALIDADO] Barrido de la densidad espectral del baño: s = J(ω)/J(2ω) en el efectivo plano y filtrado con ocupación física (n(ω), n(3ω)) y el ansatz del
desplazamiento, en (κ₂/κ, x) = (0.25, 6.86), γ/κ = 2e-5, N = 22. J(2ω) es la del baño del qubit (κ). Variante A: J(3ω) = J(ω) = s·J(2ω). Variante B: J(3ω) = J(2ω) (solo el
canal de ω escala). Escalan por s (o s3): Γ₁⁻ (fotón a ω), Γ₁⁺ (3ω) y los canales del desplazamiento (σ₊a y σ₋a† a ω; σ₊a† y σ₋a a 3ω). No escalan: el baño del qubit (2ω)
ni la pérdida intrínseca γ. Solo lee/calcula el efectivo (no guarda). Salida: data/barrido_s_ocupacion.txt"""
import sys, numpy as np, qutip as qt, scipy.sparse.linalg as sla
def eff(x, s1, s3, filtro, k2k=0.25, gzk=14.0, gam=2e-5, N=22, w=200.0, al2=4.0, kfw=0.05):
    nq = 1 / np.expm1(x); nm = 1 / np.expm1(x / 2); n1 = nm; n3 = 1 / np.expm1(3 * x / 2)
    gx = w * np.sqrt(k2k / (16 * gzk**2)); G = 2 * gx * gzk / w; Om = al2 * G; chi = 8 * gx**2 / (3 * w); kf = kfw * w
    ke = (lambda d: kf**2 / (4 * d**2 + kf**2)) if filtro else (lambda d: 1.0)
    Gm = s1 * gx**2 * ke(w) / w**2; Gp = s3 * gx**2 * ke(3 * w) / (9 * w**2); nf = 2 if filtro else 1
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(nf)]; op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm = op(0, qt.destroy(N)), op(1, qt.sigmam()); Pe = sm.dag() * sm; Pg = 1 - Pe
    H = chi * a.dag() * a * Pe + (gx**2 / w) * (Pe - Pg / 3) - G * (sm.dag() * a * a + sm * a.dag() * a.dag()) + Om * (sm.dag() + sm)
    c = []
    if filtro:
        b = op(2, qt.destroy(nf)); J = np.sqrt(kf) / 2; H = H + J * (sm.dag() * b + sm * b.dag()); c += [np.sqrt(kf * (nq + 1)) * b, np.sqrt(kf * nq) * b.dag()]
    else: c += [np.sqrt(nq + 1) * sm, np.sqrt(nq) * sm.dag()]
    c += [np.sqrt(Gm * (n1 + 1) + Gp * n3 + gam * (nm + 1)) * a, np.sqrt(Gm * n1 + Gp * (n3 + 1) + gam * nm) * a.dag()]
    b2 = (2 * gzk / w)**2 * ke(w)
    c += [np.sqrt(s1 * b2 * (n1 + 1)) * sm * a.dag(), np.sqrt(s3 * b2 * (n3 + 1)) * sm * a, np.sqrt(s1 * b2 * n1) * sm.dag() * a, np.sqrt(s3 * b2 * n3) * sm.dag() * a.dag()]
    L = qt.liouvillian(H, c).data.as_scipy().tocsc()
    lam, R = sla.eigs(L, k=12, sigma=-1e-10, which='LM', tol=1e-14, maxiter=100000)
    o = np.argsort(-lam.real); lam, R = lam[o], R[:, o]; D = 2 * N * nf
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2), qt.qeye(nf)).full(); A = a.full(); mod = []
    for k in range(len(lam)):
        M = R[:, k].reshape(D, D, order='F'); nr = np.linalg.norm(M); mod.append([-lam[k].real, abs(np.trace(P @ M)) / nr, abs(np.trace(A @ M)) / nr])
    mod = np.array(mod)[1:4]
    return mod[np.argmax(mod[:, 1]), 0], mod[np.argmax(mod[:, 2]), 0]
if __name__ == '__main__':
    x = 6.86; SS = (0.1, 0.2, 0.5, 1.0, 2.0); out = []
    for var, f3 in (('A (J(3ω) = J(ω) = s·J(2ω))', lambda s: s), ('B (J(3ω) = J(2ω))', lambda s: 1.0)):
        out.append(f'\nVariante {var}; efectivo, κ₂/κ = 0.25, x = 6.86, κ = 1, ocupación física, ansatz del desplazamiento\n  s    | plano γ_pf  γ_bf        η     | filtro γ_pf  γ_bf        η     | γ_pf plano/filtro | γ_bf plano/filtro')
        rat = []
        for s in SS:
            pp, pb = eff(x, s, f3(s), 0); fp, fb = eff(x, s, f3(s), 1); rat.append(pb / fb)
            out.append(f'  {s:<4g} | {pp:.4e} {pb:.4e} {pp/pb:6.2f} | {fp:.4e} {fb:.4e} {fp/fb:6.3f} | {pp/fp:8.3f}          | {pb/fb:8.4f}')
        sl = np.log(SS); r = np.array(rat); cr = [np.exp(np.interp(0, [np.log(r[i]), np.log(r[i + 1])], [sl[i], sl[i + 1]])) for i in range(len(r) - 1) if (np.log(r[i])) * (np.log(r[i + 1])) <= 0]
        out.append(f'  cociente γ_bf plano/filtrado cruza 1 en s = {", ".join(f"{c:.3f}" for c in cr) if cr else "no cruza en [0.1, 2]"} (interpolación log-log)')
    txt = '\n'.join(out); print(txt); open('data/barrido_s_ocupacion.txt', 'w').write(txt + '\n')
