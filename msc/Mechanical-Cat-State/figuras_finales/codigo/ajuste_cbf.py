"""Ajuste directo γ_bf = c_bf n_q κ (régimen frío convergido) en la rejilla κ₂/κ ∈ [0.02, 0.4], g_z/κ = 14, γ/κ = 2e-5, N = 22.
En cada κ₂/κ: γ_bf,0 = γ_bf en el x más frío (piso T→0); se ajusta γ_bf − γ_bf,0 = c_bf n_q^p con n_q ∈ [1e-5, 1e-2] y se da
c_bf a p = 1 (mediana de (γ_bf − γ_bf,0)/n_q, con incertidumbre = semi-rango intercuartil). Luego c_bf ∝ (G/κ)^s,
G/κ = √(κ₂/κ)/2. Escribe data/termico_cbf.csv."""
import os, glob
import numpy as np
import comun as C
import figura_termica as FT

T = FT.cargar()
X, NQ, K2, GZ, FIL, GAM, NN, GPF, GBF = T.T[:9]
filas = []
for fil in (1, 0):
    for k in np.unique(K2[(abs(GZ - 14) < 1e-9) & (GAM == 2e-5) & (NN == 22)]):
        m = (abs(GZ - 14) < 1e-9) & (K2 == k) & (FIL == fil) & (GAM == 2e-5) & (NN == 22)
        if m.sum() < 10:
            continue
        o = np.argsort(NQ[m]); nq, gb = NQ[m][o], GBF[m][o]
        g0 = gb[0]
        s = (nq >= 1e-5) & (nq <= 1e-2)
        r = (gb[s] - g0) / nq[s]
        p = np.polyfit(np.log(nq[s]), np.log(gb[s] - g0), 1)[0]
        q1, q3 = np.percentile(r, [25, 75])
        filas.append([fil, k, np.sqrt(k) / 2, np.median(r), (q3 - q1) / 2, p, g0, s.sum()])
F = np.array(filas)
np.savetxt(os.path.join(C.DATA, 'termico_cbf.csv'), F, delimiter=',', comments='',
           header='filter(1)/flat(0), kappa_2/kappa, G/kappa, c_bf = median((gamma_bf-gamma_bf0)/n_q) [kappa], half-IQR, '
                  'local exponent p in n_q, gamma_bf0 (T->0 floor) [kappa], n points (n_q in [1e-5,1e-2])')
for fil in (1, 0):
    f = F[F[:, 0] == fil]
    (s, lnA), cov = np.polyfit(np.log(f[:, 2]), np.log(f[:, 3]), 1, cov=True)
    print(f"{'filtro' if fil else 'plano'}: c_bf ∝ (G/κ)^s, s = {s:.3f} ± {np.sqrt(cov[0,0]):.3f}, A = {np.exp(lnA):.3e}")
    for r in f[::3]:
        print(f"   κ₂/κ={r[1]:.4f} G/κ={r[2]:.4f} c_bf={r[3]:.4e} ± {r[4]:.1e}  p={r[5]:.3f}  piso={r[6]:.1e}")
    for k in (0.02, 0.05, 0.2, 0.4):
        i = np.argmin(abs(f[:, 1] - k)); print(f"   c_bf(κ₂/κ≈{f[i,1]:.3f}) = {f[i,3]:.3e}")
