"""V1: verificación por diagonalización exacta del Hamiltoniano sin drive.

H = w a†a + (wq/2) σz + (a + a†)(gx σx + gz σz),  w = 1, wq = 2.

Predicciones de segundo orden (derivación en V1_derivacion.md):
  - corrimiento del oscilador con qubit en |g>:  -4 gx²/(3w)
  - corrimiento del oscilador con qubit en |e>:  +4 gx²/(3w)
  - corrimiento del qubit (n=0):                 +4 gx²/(3w)
  - constante común de g_z:                      -gz²/w (se cancela en diferencias)
  - doblete {|g,2>, |e,0>} (resonante en wq=2w): separación sqrt((4gx²/w)² + 8 G²), G = 2 gx gz / w
Los niveles se identifican por máximo solapamiento con los estados desnudos.
"""
import numpy as np
import qutip as qt

W, WQ, N = 1.0, 2.0, 40


def espectro(gx, gz, wq=WQ):
    a = qt.tensor(qt.destroy(N), qt.qeye(2))
    sz = qt.tensor(qt.qeye(N), qt.sigmaz())
    sx = qt.tensor(qt.qeye(N), qt.sigmax())
    H = W * a.dag() * a + 0.5 * wq * sz + (a + a.dag()) * (gx * sx + gz * sz)
    return H.eigenstates()


def desnudo(n, q):
    # q = 'e' -> sigmaz=+1 -> basis(2,0); q = 'g' -> basis(2,1)
    return qt.tensor(qt.basis(N, n), qt.basis(2, 0 if q == 'e' else 1))


def energia(ev, vecs, n, q):
    ov = [abs(desnudo(n, q).overlap(v)) ** 2 for v in vecs]
    k = int(np.argmax(ov))
    return ev[k], ov[k]


print("=== (A) g_z = 0: corrimientos del oscilador y del qubit ===")
print(f"{'gx':>6} {'dosc_g num':>12} {'pred':>12} {'rel':>9} {'dosc_e num':>12} {'rel':>9} {'dq num':>12} {'rel':>9}")
for gx in [0.005, 0.01, 0.02, 0.04]:
    ev, vecs = espectro(gx, 0.0)
    Eg0, _ = energia(ev, vecs, 0, 'g'); Eg1, _ = energia(ev, vecs, 1, 'g')
    Ee0, _ = energia(ev, vecs, 0, 'e'); Ee1, _ = energia(ev, vecs, 1, 'e')
    pred = 4 * gx**2 / (3 * W)
    dg = (Eg1 - Eg0) - W
    de = (Ee1 - Ee0) - W
    dq = (Ee0 - Eg0) - WQ
    print(f"{gx:6.3f} {dg:12.4e} {-pred:12.4e} {dg/(-pred)-1:9.2e} {de:12.4e} {de/pred-1:9.2e} {dq:12.4e} {dq/pred-1:9.2e}")

print("\n=== (B) g_z != 0: corrimiento de |g> (no afectado por pares) y doblete |g,2>,|e,0> ===")
print(f"{'gx':>6} {'gz':>6} {'dosc_g num':>12} {'rel':>9} {'split num':>12} {'split pred':>12} {'rel':>9} {'const num':>12} {'const pred':>12}")
for gx, gz in [(0.01, 0.01), (0.01, 0.03), (-0.02, 0.02), (0.02, 0.04), (-0.01, 0.05)]:
    ev, vecs = espectro(gx, gz)
    Eg0, _ = energia(ev, vecs, 0, 'g'); Eg1, _ = energia(ev, vecs, 1, 'g')
    pred = 4 * gx**2 / (3 * W)
    dg = (Eg1 - Eg0) - W
    # doblete: los dos autoestados con mayor peso en span{|g,2>,|e,0>}
    P = [abs(desnudo(2, 'g').overlap(v))**2 + abs(desnudo(0, 'e').overlap(v))**2 for v in vecs]
    k = np.argsort(P)[-2:]
    split = abs(ev[k[0]] - ev[k[1]])
    G = 2 * gx * gz / W
    split_p = np.sqrt((4 * gx**2 / W)**2 + 8 * G**2)
    # E(g,0) = -wq/2 - gx²/(3w) - gz²/w
    cons = Eg0 + WQ / 2
    cons_p = -gx**2 / (3 * W) - gz**2 / W
    print(f"{gx:6.3f} {gz:6.3f} {dg:12.4e} {dg/(-pred)-1:9.2e} {split:12.4e} {split_p:12.4e} {split/split_p-1:9.2e} {cons:12.4e} {cons_p:12.4e}")

print("\n=== (C) convergencia en N (gx=0.02, gz=0.04) ===")
for n in (40, 46):
    N = n
    ev, vecs = espectro(0.02, 0.04)
    print(N, energia(ev, vecs, 1, 'g')[0] - energia(ev, vecs, 0, 'g')[0])
