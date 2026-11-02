"""D3: frente de Pareto pureza-flujo. Exacto (envolvente de la malla D2) vs cerrado; puntos del frente cerrado evaluados exactamente."""
from runlib import *
gam, gph = BASE['gam'], BASE['gph']
def front(J, P):
    """envolvente no dominada: para cada flujo, maxima pureza (J y P 1D)."""
    o = np.argsort(-J); best = -1; idx = []
    for i in o:
        if P[i] > best: idx.append(i); best = P[i]
    return np.array(idx)
if __name__ == '__main__':
    name = 'D3_pareto.npz'
    if cached(name): sys.exit()
    out = {}
    for n in (2, 3):
        for lam in (0.1, 0.14, 0.2):
            d = np.load(os.path.join(DATA, f'D2_n{n}_lam{lam}.npz'))
            Pe, Je = d['P_exact'].ravel(), d['J_exact'].ravel()
            ie = front(Je, Pe)
            Of = np.geomspace(0.02, 0.3, 90); kf = np.geomspace(3e-4, 1e-2, 90)
            OO, KK = np.meshgrid(Of, kf)
            Jc = np.array([flux_closed(lam, o, k, gam, gph, n) for o, k in zip(OO.ravel(), KK.ravel())])
            Pc = np.array([P_closed(lam, o, k, gam, gph, n) for o, k in zip(OO.ravel(), KK.ravel())])
            ic = front(Jc, Pc)
            # 8 puntos del frente cerrado (espaciados en log J) evaluados exactamente
            sel = ic[np.unique(np.linspace(0, len(ic) - 1, 8).astype(int))]
            args = [(n, lam, OO.ravel()[i], KK.ravel()[i], gam, gph, 'dressed', 'episode', None) for i in sel]
            Px, Jx, _, _ = points(args)
            key = f'n{n}_lam{lam}'
            out.update({f'{key}_Jex': Je[ie], f'{key}_Pex': Pe[ie], f'{key}_Jcl': Jc[ic], f'{key}_Pcl': Pc[ic],
                        f'{key}_sel_Om': OO.ravel()[sel], f'{key}_sel_kap': KK.ravel()[sel], f'{key}_sel_Jcl': Jc[sel], f'{key}_sel_Pcl': Pc[sel],
                        f'{key}_sel_Jex': Jx, f'{key}_sel_Pex': Px})
    save(name, dict(gam=gam, gph=gph, note='frente: max P a J dado. *_Jex/_Pex: envolvente de la malla exacta 20x20 (D2); *_Jcl/_Pcl: frente cerrado sobre malla 90x90 (Om in [0.02,0.3], kap in [3e-4,1e-2]); *_sel_*: 8 puntos del frente cerrado evaluados exactamente'), **out)
