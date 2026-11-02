"""Convergencia en la truncacion N (marco de polaron, vestida) y en K: |P(N)-P(N=n+12)| y |J(N)-J(ref)|. Salida data/convergence.txt"""
from runlib import *
cases = [(2, .14, .1, .003, 2e-4, 4e-4), (2, .2, .3, .0003, 2e-4, 4e-4), (3, .14, .3, .0015, 2e-4, 4e-4), (3, .2, .4, .003, 2e-4, 4e-4), (4, .14, .1, .003, 2e-4, 4e-4), (4, .14, .4, .003, 2e-4, 4e-4)]
out = []
for c in cases:
    n = c[0]; ref = exact_point(*c, N=n + 12, K=2 * n + 8); line = f'n={n} lam={c[1]} Om={c[2]} kap={c[3]}: ref(N={n + 12}) P={ref["P"]:.8f} J={ref["flux"]:.6e} sumP={ref["sumP"]:.8f}'
    out.append(line); print(line)
    for N in range(n + 2, n + 10, 1):
        r = exact_point(*c, N=N, K=2 * n + 8)
        l = f'   N={N:2d} |dP|={abs(r["P"] - ref["P"]):.1e} |dJ|/J={abs(r["flux"] / ref["flux"] - 1):.1e}'
        out.append(l); print(l)
open(os.path.join(DATA, 'convergence.txt'), 'w').write('\n'.join(out))
