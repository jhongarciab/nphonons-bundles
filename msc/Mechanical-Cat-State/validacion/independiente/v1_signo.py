"""V1b: signo del elemento de pares <e,0|H_ef|g,2> = -G*sqrt(2), G = 2 gx gz / w.
Se reconstruye el bloque 2x2 efectivo desde los autovectores exactos proyectados
(ortonormalizados) en span{|g,2>,|e,0>}: H_ef = sum_k E_k |p_k><p_k|."""
import numpy as np, qutip as qt
import v1_diag as v
for gx, gz in [(0.01, 0.02), (-0.01, 0.02), (0.02, -0.03), (-0.02, -0.03)]:
    ev, vecs = v.espectro(gx, gz)
    b = [v.desnudo(2, 'g'), v.desnudo(0, 'e')]
    P = [sum(abs(x.overlap(s))**2 for x in b) for s in vecs]
    k = np.argsort(P)[-2:]
    M = np.array([[x.overlap(vecs[i]) for x in b] for i in k])  # filas: autovector proyectado
    Q, _ = np.linalg.qr(M.T)                                     # ortonormaliza (Löwdin aproximado)
    # Löwdin simétrico
    S = M.conj() @ M.T
    w_, U = np.linalg.eigh(S); Sm = U @ np.diag(w_**-0.5) @ U.conj().T
    C = Sm @ M                                                   # vectores ortonormales
    Hef = sum(ev[i] * np.outer(C[j], C[j].conj()) for j, i in enumerate(k))
    G = 2 * gx * gz / v.W
    print(f"gx={gx:+.2f} gz={gz:+.2f}  H_ef[g2,e0]={Hef[0,1].real:+.4e}  -G*sqrt2={-G*np.sqrt(2):+.4e}  rel={Hef[0,1].real/(-G*np.sqrt(2))-1:+.1e}")
