"""D7: rol del desfase gph y del decaimiento radiativo gam: P_2 (episodios) y flujo; barridos 1D y mapa 2D; exacto vs cerrado."""
from runlib import *
lam, Om, kap, gam0, gph0 = (BASE[k] for k in ('lam', 'Om', 'kap', 'gam', 'gph'))
if __name__ == '__main__':
    name = 'D7_dephasing.npz'
    if cached(name): sys.exit()
    n = 2
    gphs = np.concatenate([[1e-6], np.geomspace(1e-5, 1e-2, 19)]); gams = np.geomspace(2e-5, 2e-3, 20)
    P1, J1, _, _ = points([(n, lam, Om, kap, gam0, g, 'dressed', 'episode', None) for g in gphs])
    P2, J2, _, _ = points([(n, lam, Om, kap, g, gph0, 'dressed', 'episode', None) for g in gams])
    G, F = np.meshgrid(np.geomspace(5e-5, 2e-3, 14), np.geomspace(1e-5, 1e-2, 14))   # gamma (cols) x gph (rows)
    P3, J3, _, _ = points([(n, lam, Om, kap, g, f, 'dressed', 'episode', None) for g, f in zip(G.ravel(), F.ravel())])
    cl = lambda g, f: (P_closed(lam, Om, kap, g, f, n), flux_closed(lam, Om, kap, g, f, n))
    c1 = np.array([cl(gam0, g) for g in gphs]); c2 = np.array([cl(g, gph0) for g in gams]); c3 = np.array([cl(g, f) for g, f in zip(G.ravel(), F.ravel())])
    save(name, dict(n=n, lam=lam, Om=Om, kap=kap, gam0=gam0, gph0=gph0, note='1D: gph barrido (gam=gam0); gam barrido (gph=gph0); 2D: filas gph, cols gam; dL=delta_res(2,lam,Om) fijo (no recentrado)'),
         gph=gphs, gam=gams, P_vs_gph=P1, J_vs_gph=J1, Pcl_vs_gph=c1[:, 0], Jcl_vs_gph=c1[:, 1], P_vs_gam=P2, J_vs_gam=J2, Pcl_vs_gam=c2[:, 0], Jcl_vs_gam=c2[:, 1],
         G2=G[0], F2=F[:, 0], P_map=P3.reshape(G.shape), J_map=J3.reshape(G.shape), Pcl_map=c3[:, 0].reshape(G.shape), Jcl_map=c3[:, 1].reshape(G.shape))
