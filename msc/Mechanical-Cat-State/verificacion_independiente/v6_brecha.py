"""V6: brecha exacta del modelo mínimo H = G[(a² - α²)σ+ + h.c.], κ D[σ-], α = 0.
Liouvilliano estático, diagonalización densa. Brecha = menor |Re λ| no nulo (se excluyen los 4 ceros del
espacio oscuro span{|0>,|1>}⊗|g>). Predicción: Δ = (κ/4)[1 - sqrt(1 - 8Γ₂/κ)] si Γ₂ ≤ κ/8, κ/4 si no; Γ₂ = 4G²/κ.
Se reporta además el autovector (qué sector de Fock domina) y el peso de borde (n > N-6)."""
import numpy as np, qutip as qt
KAP = 1.0
def brecha(G, N):
    a = qt.tensor(qt.destroy(N), qt.qeye(2)); sm = qt.tensor(qt.qeye(N), qt.sigmam())
    H = G * (a * a * sm.dag() + a.dag() * a.dag() * sm)
    L = qt.liouvillian(H, [np.sqrt(KAP) * sm]).full()
    ev, R = np.linalg.eig(L)
    D = 2 * N; nidx = np.repeat(np.arange(N), 2); borde = nidx > N - 6
    ceros = np.sum(np.abs(ev) < 1e-10)
    orden = np.argsort(-ev.real)
    for k in orden:
        if abs(ev[k]) < 1e-10: continue
        M = R[:, k].reshape(D, D, order='F'); M = M / np.linalg.norm(M)
        pb = 1 - np.linalg.norm(M[np.ix_(~borde, ~borde)])**2
        return -ev[k].real, ev[k].imag, ceros, pb
def pred(G):
    g2 = 4 * G**2 / KAP
    return KAP / 4 * (1 - np.sqrt(1 - 8 * g2 / KAP)) if g2 <= KAP / 8 else KAP / 4
print(" Γ₂/κ     G/κ      Δ(N=20)       Im        pred          rel        Δ(N=26)     rel N  ceros borde")
for x in [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.124, 0.125, 0.126, 0.13, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
    G = np.sqrt(x * KAP / 4)
    d1, im, c, pb = brecha(G, 20); d2, *_ = brecha(G, 26); p = pred(G)
    print(f"{x:6.3f}  {G:7.4f}  {d1:.8e}  {im:+.1e}  {p:.8e}  {d1/p-1:+.1e}  {d2:.8e}  {d2/d1-1:+.0e}  {c}  {pb:.0e}")
