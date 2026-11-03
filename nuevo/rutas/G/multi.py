"""Emisores en nodos, reservorio compartido por arista: L_e = sqrt(g)(s_i + s_j). Plena inversion inicial.
Mide: dim del nucleo comun de los L_e (estados oscuros en TODOS los sectores), exc. residual a tiempo largo, pico de intensidad."""
import numpy as np, qutip as qt, networkx as nx, sys, time
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
sm = [qt.tensor(*[qt.destroy(2) if k == i else qt.qeye(2) for k in range(N)]) for i in range(N)]   # destroy(2) = |0><1| (1=excitado)
nt = sum(s.dag()*s for s in sm)
def kernel_dim(G):
    Ls = np.vstack([ (sm[i]+sm[j]).full() for i, j in G.edges ]); 
    sv = np.linalg.svd(Ls, compute_uv=False); return int((sv < 1e-9).sum()) + (2**N - len(sv))
def run(G, T=60.0, nt_pts=121):
    cs = [(sm[i]+sm[j]) for i, j in G.edges]
    psi = qt.tensor(*[qt.basis(2, 1)]*N)
    tl = np.linspace(0, T, nt_pts)
    r = qt.mesolve(0*nt, psi*psi.dag(), tl, cs, e_ops=[nt] + [c.dag()*c for c in cs], options=dict(atol=1e-9, rtol=1e-9, nsteps=10**6))
    n = np.array(r.expect[0]); I = np.sum([np.array(e) for e in r.expect[1:]], axis=0)
    return tl, n, I
C8 = nx.cycle_graph(N); C8c = C8.copy(); C8c.add_edge(0, 2)      # cuerda -> ciclo impar
gs = {'camino P8 (bip)': nx.path_graph(N), 'ciclo C8 (bip)': C8, 'C8+cuerda (no bip)': C8c, 'estrella S8 (bip)': nx.star_graph(N-1),
      'completo K8': nx.complete_graph(N), 'bipartito completo K4,4': nx.complete_bipartite_graph(N//2, N//2), 'ER(8,.4) conexo': None}
for s in range(50):
    g = nx.erdos_renyi_graph(N, .45, seed=s)
    if nx.is_connected(g): gs['ER(8,.4) conexo'] = g; break
print('%-24s %-5s %-9s %-10s %-10s %-10s'%('grafo', 'bip', 'dim_nucleo', 'n(t=60)', 'I_pico', 't_pico')); sys.stdout.flush()
out = {}
for nm, G in gs.items():
    t0 = time.time(); tl, n, I = run(G)
    print('%-24s %-5s %-9d %-10.3f %-10.3f %-10.2f  (%.0fs)'%(nm, nx.is_bipartite(G), kernel_dim(G), n[-1], I.max(), tl[I.argmax()], time.time()-t0)); sys.stdout.flush()
    out[nm] = (tl, n, I)
np.savez('multi.npz', **{k.split(' ')[0]+'_t': v[0] for k, v in out.items()}, **{k.split(' ')[0]+'_n': v[1] for k, v in out.items()})
