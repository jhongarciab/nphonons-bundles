"""D2 (+D3 exacto): P_n (episodios) y flujo exactos y cerrados en el plano (Om,kap), n=2,3, lam=0.1,0.14,0.2; curvas Om_max(kap;X), kappa*(n)."""
from runlib import *
NG = 20
Oms = np.geomspace(0.02, 0.3, NG); kaps = np.geomspace(3e-4, 1e-2, NG)
gam, gph = BASE['gam'], BASE['gph']
if __name__ == '__main__':
    for n in (2, 3):
        for lam in (0.14, 0.1, 0.2):
            name = f'D2_n{n}_lam{lam}.npz'
            if cached(name): continue
            args = [(n, lam, O, k, gam, gph, 'dressed', 'episode', None) for k in kaps for O in Oms]
            P, J, dist, T = points(args)
            P = P.reshape(NG, NG); J = J.reshape(NG, NG)
            Pc = np.array([[P_closed(lam, O, k, gam, gph, n) for O in Oms] for k in kaps])
            Pc_v1 = np.array([[P_closed(lam, O, k, gam, gph, n, fcmid=False) for O in Oms] for k in kaps])
            Pc_ep = np.array([[P_closed(lam, O, k, gam, gph, n, 'ep') for O in Oms] for k in kaps])
            Jc = np.array([[flux_closed(lam, O, k, gam, gph, n) for O in Oms] for k in kaps])
            xi = np.array([[omega_eff(n, lam, O) / gamma_c(n, 0, k, gam, gph) for O in Oms] for k in kaps])  # Om_n / gamma_c(0): >~1 => Rabi coherente
            Of = np.geomspace(0.02, 0.3, 120); kf = np.geomspace(3e-4, 1e-2, 120)
            Pcf = np.array([[P_closed(lam, O, k, gam, gph, n) for O in Of] for k in kf])
            Jcf = np.array([[flux_closed(lam, O, k, gam, gph, n) for O in Of] for k in kf])
            om = {X: np.array([Om_max(k, X, lam, gam, gph, n, Omcap=0.3) for k in kf]) for X in (0.9, 0.95)}
            ks = np.array([kappa_star(n, lam, O, gam, gph, hi=0.1)[0] for O in Oms])
            save(name, dict(n=n, lam=lam, gam=gam, gph=gph, jump='dressed', kind='episode', N='n+6', note='rows=kap, cols=Om; dL=delta_res(n,lam,Om)'),
                 Om=Oms, kap=kaps, P_exact=P, J_exact=J, P_closed=Pc, P_closed_ep=Pc_ep, P_closed_v1=Pc_v1, J_closed=Jc, xi=xi, dist=dist.reshape(NG, NG, -1),
                 Om_fine=Of, kap_fine=kf, P_closed_fine=Pcf, J_closed_fine=Jcf, Ommax_X0p9=om[0.9], Ommax_X0p95=om[0.95], kappa_star_vs_Om=ks)
