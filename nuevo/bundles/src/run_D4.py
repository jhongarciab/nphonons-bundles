"""D4: maestra local vs vestida. P_n(kap) por ciclos y por episodios, flujo, nube kap lam^2/gam y prediccion P_loc ~ P_dr exp(-kap lam^2/gam)."""
from runlib import *
gam, gph, lam = BASE['gam'], BASE['gph'], BASE['lam']
kaps = np.geomspace(5e-4, 2e-2, 24)
if __name__ == '__main__':
    name = 'D4_local_vs_dressed.npz'
    if cached(name): sys.exit()
    out = {}
    for n, Om in ((2, 0.1), (3, 0.2)):
        for jump in ('dressed', 'local'):
            for kind in ('cycle', 'episode'):
                args = [(n, lam, Om, k, gam, gph, jump, kind, None) for k in kaps]
                P, J, dist, T = points(args)
                out.update({f'n{n}_{jump}_{kind}_P': P, f'n{n}_{jump}_{kind}_dist': dist, f'n{n}_{jump}_{kind}_T': T})
                if kind == 'cycle': out[f'n{n}_{jump}_J'] = J
        # fonones medios por ciclo y exceso vs nube
        for jump in ('dressed', 'local'):
            d = out[f'n{n}_{jump}_cycle_dist']; out[f'n{n}_{jump}_kmean'] = (d * np.arange(d.shape[1])).sum(1) / d.sum(1)
        out[f'n{n}_cloud_pred'] = kaps * lam**2 / gam
        out[f'n{n}_Ploc_pred'] = out[f'n{n}_dressed_cycle_P'] * np.exp(-kaps * lam**2 / gam)
        out[f'n{n}_Pclosed'] = np.array([P_closed(lam, Om, k, gam, gph, n) for k in kaps])
    save(name, dict(lam=lam, gam=gam, gph=gph, Om_n2=0.1, Om_n3=0.2, note='cloud=kap lam^2/gam = fonones extra/ciclo (maestra local); Ploc_pred = P_dressed_cycle*exp(-cloud)'), kap=kaps, **out)
