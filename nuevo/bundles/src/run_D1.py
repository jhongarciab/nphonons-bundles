"""D1: espectro de resonancias: flujo exacto J y distribucion de episodios vs dL en [-4.6,0.6] (parametros base, N=10)."""
from runlib import *
lam, Om, kap, gam, gph = (BASE[k] for k in ('lam', 'Om', 'kap', 'gam', 'gph'))
def lines():
    pos = {n: delta_res(n, lam, Om) for n in range(1, 5)}
    g = list(np.arange(-4.6, 0.6001, 0.01))
    for c in list(pos.values()) + [lam**2]:
        g += list(c + np.arange(-0.03, 0.03001, 0.0005))
    return np.unique(np.round(np.array(g), 6)), pos
if __name__ == '__main__':
    name = 'D1_spectrum.npz'
    if not cached(name):
        dls, pos = lines()
        args = [(10 - 6, lam, Om, kap, gam, gph, 'dressed', 'episode', d) for d in dls]   # N = n+6 = 10
        P, J, dist, T = points(args)
        # P devuelve P[n=4]; usamos la distribucion completa
        Pdom = dist[:, 1:].max(1); kdom = dist[:, 1:].argmax(1) + 1
        # lineshape cerrado (suma de lorentzianas m>=1): J = sum_m R_m (m+lam^2)/(1+R_tot/gam)
        def Jl(dL):
            d = dL - lam**2; R = []
            for m in range(1, 8):
                gc = (m * kap + gam + gph) / 2
                R.append((m, 2 * Om**2 * dmat(lam, m, 0)**2 * gc / ((d + m - eps_n(lam, Om, m))**2 + gc**2)))
            Rt = sum(r for _, r in R)
            return sum(r * (m + lam**2) for m, r in R) / (1 + Rt / gam)
        Jlor = np.array([Jl(d) for d in dls])
        save(name, dict(lam=lam, Om=Om, kap=kap, gam=gam, gph=gph, N=10, jump='dressed', K=10,
                        note='dist[:,k]=P_ep(k), k=0..10; Pdom=max_k>=1 P_ep(k); positions: dL=lam^2-n+eps_n'),
             dL=dls, J=J, dist=dist, T_ep=T, Pdom=Pdom, kdom=kdom, J_lorentz=Jlor,
             pos_n=np.array([1, 2, 3, 4]), pos_dL=np.array([pos[n] for n in range(1, 5)]), pos_dL_bin=np.array([-np.sqrt(n * n - 4 * Om**2) + lam**2 for n in range(1, 5)]))
