"""Test B: Mpemba bosonico (Fock vs coherente vs gato) con no linealidad. Espectro de Liouville exacto (dim N=24) y coeficientes de las condiciones iniciales en los modos lentos."""
import numpy as np, qutip as qt, scipy.linalg as sl
N = 24
a = qt.destroy(N); n = a.dag()*a
def analyse(label, H, cs, inits, nslow=4):
    L = qt.liouvillian(H, cs).full(); w, V = np.linalg.eig(L); Vi = np.linalg.inv(V)
    order = np.argsort(-w.real); ss = order[0]
    print('==', label, ' modo estacionario Re=%.2e'%w[ss].real)
    slow = [k for k in order[1:nslow+1]]
    print('  modos lentos (Re, Im):', ['(%.3f,%.3f)'%(w[k].real, w[k].imag) for k in slow])
    for nm, r0 in inits.items():
        c = Vi @ qt.operator_to_vector(r0).full().ravel()
        print('  %-14s |coef|: '%nm + ' '.join('%.3e'%abs(c[k]) for k in slow))
kap, U, k2 = 1.0, 0.7, 0.3
nbar = 4
inits = {'Fock n=4': qt.fock_dm(N, 4), 'coherente a=2': qt.coherent_dm(N, 2.0)}
cat = (qt.coherent(N, 2.0) + qt.coherent(N, -2.0)).unit(); inits['gato par']= cat*cat.dag()
th = qt.thermal_dm(N, 4); inits['termico nbar=4'] = th
analyse('lineal', 0*n, [np.sqrt(kap)*a], inits)
analyse('Kerr U=0.7 + perdida', U*a.dag()*a.dag()*a*a, [np.sqrt(kap)*a], inits)
analyse('perdida 1 y 2 fotones (k2=0.3)', 0*n, [np.sqrt(kap)*a, np.sqrt(k2)*a*a], inits)
analyse('Kerr + perdida 1 y 2 fotones', U*a.dag()*a.dag()*a*a, [np.sqrt(kap)*a, np.sqrt(k2)*a*a], inits)
