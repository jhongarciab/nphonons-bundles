"""D5: cota de Franck-Condon. P_n^max(lam) = max_kap P_n (exacto, episodios y ciclos) vs analitico (cerrada optimizada y pisos)."""
from runlib import *
from scipy.optimize import minimize_scalar
gam, gph = BASE['gam'], BASE['gph']
lams = np.linspace(0.05, 0.3, 11)
def Pex(n, lam, Om, k, kind='episode'): return exact_point(n, lam, Om, k, gam, gph, kind=kind, K=2 * n + 6)['P']
def opt(n, lam, Om, kind):
    ks = np.geomspace(3e-4, 5e-2, 14)
    Ps = [Pex(n, lam, Om, k, kind) for k in ks]; i = int(np.argmax(Ps))
    if i in (0, len(ks) - 1): return ks[i], Ps[i], 1
    r = minimize_scalar(lambda lk: -Pex(n, lam, Om, np.exp(lk), kind), bounds=(np.log(ks[i - 1]), np.log(ks[i + 1])), method='bounded', options=dict(xatol=1e-3))
    return float(np.exp(r.x)), -r.fun, 0
if __name__ == '__main__':
    name = 'D5_floor.npz'
    if cached(name): sys.exit()
    out = {}
    for n, Om in ((2, 0.1), (3, 0.2)):
        r = {k: [] for k in ('kex', 'Pex', 'edge', 'kcl', 'Pcl', 'kcyc', 'Pcyc')}
        for lam in lams:
            k, P, e = opt(n, lam, Om, 'episode'); r['kex'].append(k); r['Pex'].append(P); r['edge'].append(e)
            kc, Pc = kappa_star(n, lam, Om, gam, gph, hi=0.1); r['kcl'].append(kc); r['Pcl'].append(Pc)
            k2, P2, _ = opt(n, lam, Om, 'cycle'); r['kcyc'].append(k2); r['Pcyc'].append(P2)
        out.update({f'n{n}_{k}': np.array(v) for k, v in r.items()})
    save(name, dict(gam=gam, gph=gph, Om_n2=0.1, Om_n3=0.2, kap_range=[3e-4, 5e-2], note='maximo sobre kap a Om fijo. floor_exp=e^{-lam^2}; floor_ep=1/(2-e^{-lam^2}); edge=1 si el maximo esta en el borde de la rejilla'),
         lam=lams, floor_exp=np.exp(-lams**2), floor_ep=1 / (2 - np.exp(-lams**2)), **out)
