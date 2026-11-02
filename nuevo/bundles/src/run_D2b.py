"""D2b: modelo de tasas (Pauli, Raman lorentziano) sobre la misma malla que D2 (n=2,3; lam=0.1,0.14,0.2), pureza y flujo."""
from runlib import *
gam, gph = BASE['gam'], BASE['gph']
if __name__ == '__main__':
    for n in (2, 3):
        for lam in (0.14, 0.1, 0.2):
            name = f'D2b_rate_n{n}_lam{lam}.npz'
            if cached(name): continue
            d = np.load(os.path.join(DATA, f'D2_n{n}_lam{lam}.npz')); Oms, kaps = d['Om'], d['kap']
            R = [[rate_model(lam, O, k, gam, gph, n) for O in Oms] for k in kaps]
            save(name, dict(n=n, lam=lam, gam=gam, gph=gph, raman='lorentz', note='rows=kap, cols=Om (misma malla que D2)'),
                 P_rate=np.array([[r['P'][n] for r in row] for row in R]), J_rate=np.array([[r['flux'] for r in row] for row in R]))
