"""Verificación (2026-10-01): (a) α_eff² frente al cociente γ_pf completo/efectivo, (b) descomposición de la diferencia de aumentos térmicos con filtro, y efectivo con filtro y ansatz del desplazamiento con n_q y con n(ω), n(3ω). Solo lee data/ y calcula el efectivo (sin guardar). Salida: data/handoff_verificacion_referencia.txt"""
import sys
sys.path.insert(0, 'codigo')
import numpy as np, qutip as qt, scipy.sparse.linalg as sla, glob, os
K = 0.03; FC = 'data/filtro_completo/'
def eff(x, mode=None, gamma_real=False, filtro=1, k2k=0.25, gzk=14.0, gam=2e-5, N=22, w=200.0, al2=4.0, kfw=0.05, ret_state=False):
    """efectivo (con o sin filtro). mode None / 'nq' / 'real' (ver verif2)."""
    nq = 1 / np.expm1(x); nm = 1 / np.expm1(x / 2); n1 = nm; n3 = 1 / np.expm1(3 * x / 2)
    gx = w * np.sqrt(k2k / (16 * gzk**2)); G = 2 * gx * gzk / w; Om = al2 * G; chi = 8 * gx**2 / (3 * w); kf = kfw * w
    ke = (lambda d: kf**2 / (4 * d**2 + kf**2)) if filtro else (lambda d: 1.0)
    Gm = gx**2 * ke(w) / w**2; Gp = gx**2 * ke(3 * w) / (9 * w**2); nf = 2 if filtro else 1
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(nf)]; op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm = op(0, qt.destroy(N)), op(1, qt.sigmam()); Pe = sm.dag() * sm; Pg = 1 - Pe
    H = chi * a.dag() * a * Pe + (gx**2 / w) * (Pe - Pg / 3) - G * (sm.dag() * a * a + sm * a.dag() * a.dag()) + Om * (sm.dag() + sm)
    c = []
    if filtro:
        b = op(2, qt.destroy(nf)); J = np.sqrt(kf) / 2; H = H + J * (sm.dag() * b + sm * b.dag()); c += [np.sqrt(kf * (nq + 1)) * b, np.sqrt(kf * nq) * b.dag()]
    else: c += [np.sqrt(nq + 1) * sm, np.sqrt(nq) * sm.dag()]
    q1, q3 = (n1, n3) if gamma_real else (nq, nq)
    c += [np.sqrt(Gm * (q1 + 1) + Gp * q3 + gam * (nm + 1)) * a, np.sqrt(Gm * q1 + Gp * (q3 + 1) + gam * nm) * a.dag()]
    if mode:
        b2 = (2 * gzk / w)**2 * ke(w)
        na, nb, nc, nd = ((nq, nq, nq, nq) if mode == 'nq' else (n1, n3, n1, n3))
        c += [np.sqrt(b2 * (nc + 1)) * sm * a.dag(), np.sqrt(b2 * (nd + 1)) * sm * a, np.sqrt(b2 * na) * sm.dag() * a, np.sqrt(b2 * nb) * sm.dag() * a.dag()]
    L = qt.liouvillian(H, c).data.as_scipy().tocsc()
    lam, R = sla.eigs(L, k=12, sigma=-1e-10, which='LM', tol=1e-14, maxiter=100000)
    o = np.argsort(-lam.real); lam, R = lam[o], R[:, o]; D = 2 * N * nf
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2), qt.qeye(nf)).full(); A = a.full(); mod = []
    for k in range(len(lam)):
        M = R[:, k].reshape(D, D, order='F'); nr = np.linalg.norm(M); mod.append([-lam[k].real, abs(np.trace(P @ M)) / nr, abs(np.trace(A @ M)) / nr])
    mod = np.array(mod)[1:4]; gpf = mod[np.argmax(mod[:, 1]), 0]; gbf = mod[np.argmax(mod[:, 2]), 0]
    if ret_state:
        M = R[:, 0].reshape(D, D, order='F'); M = M / np.trace(M); M = (M + M.conj().T) / 2
        return gpf, gbf, dict(a2=abs(np.trace(A @ A @ M)), nn=np.real(np.trace(a.dag().full() @ A @ M)))
    return gpf, gbf
print('=== (a) α_eff² y el cociente γ_pf completo/efectivo')
H = {}
for ln in open('data/handoff_cifras.txt'):
    if ln.startswith('#') or '=' not in ln: continue
    a, b = ln.split(' = '); H[a.strip()] = float(b.split('|')[0])
GX = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}; XM = {0.05: 9.0, 0.1: 9.0, 0.25: 10.1, 0.4: 9.5}
def fpred(k, x, al2):   # fórmula analítica del efectivo con filtro: 2[down α² + up (α²+1)]
    w = 200.0; gzk = 14.0; gx = w * np.sqrt(k / (16 * gzk**2)); kf = 10.0; ke = lambda d: kf**2 / (4 * d**2 + kf**2)
    Gm = gx**2 * ke(w) / w**2; Gp = gx**2 * ke(3 * w) / (9 * w**2); gam = 2e-5
    nq = 1 / np.expm1(x) if x < 50 else 0.0; nm = 1 / np.expm1(x / 2) if x < 50 else 0.0
    down = Gm * (nq + 1) + Gp * nq + gam * (nm + 1); up = Gm * nq + Gp * (nq + 1) + gam * nm
    return 2 * (down * al2 + up * (al2 + 1))
print('npz del completo guardan:', sorted(np.load(FC + 'gx0.0535714_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz').files)[:30])
for k in (0.05, 0.1, 0.25, 0.4):
    z0 = np.load(FC + f'gx{GX[k]:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz'); a0 = float(np.real(z0['al2eff']))
    for x in (6.86, XM[k]):
        z = np.load(FC + f'gx{GX[k]:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07_x{x:g}.npz'); a1 = float(np.real(z['al2eff']))
        r0 = H[f'gpf0_c[{k}]'] / H[f'gpf0_e[{k}]']; rx = H[f'gpf_c[{k},{x:g}]'] / H[f'gpf_e[{k},{x:g}]']
        d_c = H[f'gpf_c[{k},{x:g}]'] - H[f'gpf0_c[{k}]']; d_e = H[f'gpf_e[{k},{x:g}]'] - H[f'gpf0_e[{k}]']
        p0 = fpred(k, 60, a0) / fpred(k, 60, 4.0); px = fpred(k, x, a1) / fpred(k, x, 4.0)
        pd = (fpred(k, x, a1) - fpred(k, 60, a0)) / (fpred(k, x, 4.0) - fpred(k, 60, 4.0))
        print(f'κ₂/κ={k} x={x:<5g} α_eff²(T=0)={a0:.4f} α_eff²(x)={a1:.4f} | cociente T=0 medido {r0:.4f} predicho {p0:.4f} | cociente térmico medido {rx:.4f} predicho {px:.4f} | cociente de incrementos medido {d_c/d_e:.4f} predicho {pd:.4f}')
print('\nα_eff² y ⟨a†a⟩ del efectivo con filtro (estado estacionario), κ₂/κ = 0.25:')
for x in (60.0, 6.86):
    p, b, s = eff(x, ret_state=True); print(f'  x={x:g}: |⟨a²⟩|={s["a2"]:.4f}  ⟨a†a⟩={s["nn"]:.4f}  (completo: α_eff² = {float(np.real(np.load(FC + "gx0.0535714_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07" + ("_x6.86" if x < 50 else "") + ".npz")["al2eff"])):.4f}; ⟨a†a⟩ no está guardado en los .npz del completo)')
print('\n=== (b) descomposición de la diferencia de incrementos: diff = (r_x − r_0)·γ_e(x) + (r_0 − 1)·Δ_e')
for k in (0.05, 0.1, 0.25, 0.4):
    for x in (6.86, XM[k]):
        ge, gc = H[f'gpf_e[{k},{x:g}]'], H[f'gpf_c[{k},{x:g}]']; ge0, gc0 = H[f'gpf0_e[{k}]'], H[f'gpf0_c[{k}]']; r0 = gc0 / ge0; rx = gc / ge
        t1 = (rx - r0) * ge; t2 = (r0 - 1) * (ge - ge0); print(f'  κ₂/κ={k} x={x:<5g} r_0={r0:.4f} r_x={rx:.4f} | (r_x−r_0)γ_e={t1:+.2e}  (r_0−1)Δ_e={t2:+.2e}  suma={t1+t2:+.2e}  diff medida={(gc-gc0)-(ge-ge0):+.2e}  n_q={1/np.expm1(x):.2e}')
print('\n=== cálculo nuevo: efectivo con filtro, κ₂/κ = 0.25, x = 6.86 (κ=1)')
rows = {}
for nombre, mode, gr in (('sin desplazamiento', None, False), ('ansatz con n_q', 'nq', False), ('ansatz con n(ω), n(3ω); Γ₁± con n_q', 'real', False), ('ansatz con n(ω), n(3ω); Γ₁± con n(ω), n(3ω)', 'real', True), ('sin ansatz; Γ₁± con n(ω), n(3ω)', None, True)):
    p, b = eff(6.86, mode, gr); rows[nombre] = (p, b); print(f'  FILTRO {nombre:48s} γ_pf={p:.5e} γ_bf={b:.5e} η={p/b:.4g}')
pl = {}
for nombre, mode, gr in (('sin desplazamiento', None, False), ('ansatz con n_q', 'nq', False), ('ansatz con n(ω), n(3ω); Γ₁± con n_q', 'real', False), ('ansatz con n(ω), n(3ω); Γ₁± con n(ω), n(3ω)', 'real', True), ('sin ansatz; Γ₁± con n(ω), n(3ω)', None, True)):
    p, b = eff(6.86, mode, gr, filtro=0); pl[nombre] = (p, b)
print('\n  cociente efectivo plano / efectivo filtrado (mismo modelo en ambos):')
for n in rows: print(f'  {n:48s} γ_pf plano/filtrado={pl[n][0]/rows[n][0]:.4g}  γ_bf plano/filtrado={pl[n][1]/rows[n][1]:.4g}  η plano={pl[n][0]/pl[n][1]:.4g} η filtrado={rows[n][0]/rows[n][1]:.4g}')
print('  completo (ruido blanco): γ_pf plano/filtrado = %.4g, γ_bf plano/filtrado = %.4g' % (1.36936e-3 / 1.652555e-4, 2.86910e-5 / 3.348950e-5))
