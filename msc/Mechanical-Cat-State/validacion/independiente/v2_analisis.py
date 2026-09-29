"""V2: análisis del barrido (máximo de P_c por spline), micromovimiento y validaciones."""
import glob, numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
d = {}
for f in glob.glob('res_v2/barr_wp*_N22.npz'):
    z = np.load(f); d[float(z['wp'])] = z
wps = np.array(sorted(d)); Pc = np.array([d[w]['reg'][-1, 1] for w in wps])
cs = CubicSpline(wps, Pc)
m = minimize_scalar(lambda x: -cs(x), bounds=(11.978, 11.982), method='bounded')
print(f"máximo discreto: wp={wps[Pc.argmax()]} Pc={Pc.max():.5f}; spline: wp*={m.x:.5f} Pc={cs(m.x):.5f}")
x = np.linspace(wps[0], wps[-1], 20001); y = cs(x); base = Pc.min()
print("\nwp        Pc(estrob)  Pe(estrob) Pe prom(periodo) Pe[min,max]      |a2| estrob  |a2| prom   Pc micro[min,max]")
for w in wps:
    z = d[w]; f = z['reg'][-1]; mp = z['micro_Pe'].real; ma = np.abs(z['micro_a2']); mc = z['micro_Pc'].real
    print(f"{w:<9} {f[1]:.5f}    {f[2]:.5f}    {mp[:-1].mean():.5f}         [{mp.min():.4f},{mp.max():.4f}]  {f[3]:.4f}      {ma[:-1].mean():.4f}     [{mc.min():.5f},{mc.max():.5f}]")
print("\nvalidaciones (peor en toda la serie):")
for f in sorted(glob.glob('res_v2/*.npz')):
    z = np.load(f); p = z['peor']; print(f"  {f:38s} |Tr-1|={p[0]:.1e} ||ρ-ρ†||={p[1]:.1e} mín eig={p[2]:.1e}")
for f in ['res_v2/wp11.9800_N22.npz', 'res_v2/wp12.0000_N22.npz', 'res_v2/wp11.9800_N28.npz']:
    z = np.load(f); r = z['reg']
    for Gt in (30, 60, 150, 300):
        i = np.argmin(abs(r[:, 0] - Gt))
        if abs(r[i, 0] - Gt) < 1: print(f"  {f[7:]:22s} Γt={r[i,0]:6.1f} Pc={r[i,1]:.5f} Pe={r[i,2]:.5f} |a2|={r[i,3]:.4f} par={r[i,6]:+.4f}")
