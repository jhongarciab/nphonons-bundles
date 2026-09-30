"""Baño plano, completo, N = 22, κ₂/κ = 0.25: identifica el "modo vecino" de γ_pf. Base lógica del gato en el marco polarónico (código fijo):
|C±> = D(g_z/ω)(|α> ± |−α>)/norma; P_L = |C+><C+| − |C−><C−| (paridad, mide la coherencia Re), Y_L = i(|C+><C−| − |C−><C+|) (la otra coherencia),
Z_L = |C+><C−| + |C−><C+| (diferencia de poblaciones |α>/|−α>, bit-flip). Reporta, para los modos con tasa cerca de γ_pf, el traslape
|Tr(O† M_k)|/‖M_k‖ con cada operador lógico (O ⊗ 1_qubit). Uso: python plano_modos.py [x]  (sin x = T = 0)."""
import sys, numpy as np, qutip as qt
import comun as C
x = float(sys.argv[1]) if len(sys.argv) > 1 else None
gx, w, gz, kap, al2, N, gam = 0.0535714, 6.0, 0.42, 0.03, 4.0, 22, 6e-7
a, sm, sz, sx = (qt.tensor(o, qt.qeye(2)) if i == 0 else qt.tensor(qt.qeye(N), o) for i, o in enumerate([qt.destroy(N), qt.sigmam(), qt.sigmaz(), qt.sigmax()]))
wp = 2 * (w - 4 * gx**2 / (3 * w)); G = 2 * gx * gz / w; Om = al2 * G; Tp = 2 * np.pi / wp
H0 = w * a.dag() * a + 0.5 * wp * sz + (a + a.dag()) * (gx * sx + gz * sz)
H = [H0, [Om * sm.dag(), lambda t: np.exp(-1j * wp * t)], [Om * sm, lambda t: np.exp(1j * wp * t)]]
nq = 1 / np.expm1(x) if x else 0.0; nm = 1 / np.expm1(x / 2) if x else 0.0
cops = [np.sqrt(kap * (nq + 1)) * sm] + ([np.sqrt(kap * nq) * sm.dag()] if nq else []) + [np.sqrt(gam * (nm + 1)) * a] + ([np.sqrt(gam * nm) * a.dag()] if nm else [])
U = qt.propagator(H, Tp, cops, options=C.OPTS).full()
lam, R = np.linalg.eig(U); D = 2 * N; rate = -np.log(np.abs(lam)) / Tp
Dd = qt.displace(N, gz / w); al = np.sqrt(al2)
cp = (Dd * (qt.coherent(N, al) + qt.coherent(N, -al))).unit(); cm = (Dd * (qt.coherent(N, al) - qt.coherent(N, -al))).unit()
I2 = np.eye(2)
ops = {'P_L (paridad)': qt.tensor(cp.proj() - cm.proj(), qt.qeye(2)).full(),
       'Y_L (otra coherencia)': qt.tensor(1j * (cp * cm.dag() - cm * cp.dag()), qt.qeye(2)).full(),
       'Z_L (bit-flip)': qt.tensor(cp * cm.dag() + cm * cp.dag(), qt.qeye(2)).full()}
orden = np.argsort(rate)[:10]
print(f"{'x=%g' % x if x else 'T=0'}: modos lentos (Re λ, tasa_Ma, " + ", ".join(f'|<{k}>|' for k in ops) + ")")
for k in orden:
    Mk = R[:, k].reshape(D, D, order='F'); n = np.linalg.norm(Mk)
    print(f"  Reλ={lam[k].real:+.6f} Imλ={lam[k].imag:+.1e} tasa={rate[k]:.5e} " + " ".join(f"{abs(np.trace(O.conj().T @ Mk)) / n:.4f}" for O in ops.values()))
