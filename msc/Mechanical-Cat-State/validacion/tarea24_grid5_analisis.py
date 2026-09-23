"""Analiza las rejillas 5x5 de la Tarea 24 (Gamma2/kappa = 0.03, 0.13, 2.07)."""
import numpy as np, glob
out = {}
np.set_printoptions(linewidth=160)
for gz, lab in [("0.12028", "0.03"), ("0.25039", "0.13"), ("0.99913", "2.07")]:
    rows = []
    for f in glob.glob(f"tarea24_cache/gz{gz}_dm*_dq*.npz"):
        d = np.load(f)
        rows.append([float(d[k]) for k in ("delta_m", "Delta_q", "gamma_bf", "im_bf", "gamma_pf", "gamma_conf")])
    R = np.array(sorted(rows))
    dms = np.unique(R[:, 0]); dqs = np.unique(R[:, 1])
    print(f"\n=== Gamma2/kappa={lab} (gz={gz}) {len(R)} celdas ===")
    print("gamma_bf (filas dm, cols dq):", dqs)
    G = np.full((len(dms), len(dqs)), np.nan); I = G.copy()
    for r in R:
        G[list(dms).index(r[0]), list(dqs).index(r[1])] = r[2]
        I[list(dms).index(r[0]), list(dqs).index(r[1])] = r[3]
    for i, dm in enumerate(dms): print(f"dm={dm:+.3f}", " ".join(f"{x:.3e}" for x in G[i]))
    print("Im(lam_bf):")
    for i, dm in enumerate(dms): print(f"dm={dm:+.3f}", " ".join(f"{x:+.2e}" for x in I[i]))
    i, j = np.unravel_index(np.nanargmin(G), G.shape)
    print(f"MIN gamma_bf={G[i,j]:.4e} en (dm,dq)=({dms[i]:+.3f},{dqs[j]:+.3f}) Im={I[i,j]:+.2e}")
    out[f"G_{lab}"] = G; out[f"I_{lab}"] = I; out[f"dms_{lab}"] = dms; out[f"dqs_{lab}"] = dqs
np.savez("tarea24_grid5_analisis.npz", **out)
