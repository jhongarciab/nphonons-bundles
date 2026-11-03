import numpy as np, qutip as qt, networkx as nx, sys, time
def trapped(G, N, T=150.0):
    sm = [qt.tensor(*[qt.destroy(2) if k == i else qt.qeye(2) for k in range(N)]) for i in range(N)]
    nt = sum(s.dag()*s for s in sm); cs = [(sm[i]+sm[j]) for i, j in G.edges]
    psi = qt.tensor(*[qt.basis(2, 1)]*N)
    r = qt.mesolve(0*nt, psi*psi.dag(), [0, T/2, T], cs, e_ops=[nt], options=dict(atol=1e-9, rtol=1e-9, nsteps=10**6))
    return r.expect[0][1], r.expect[0][2]
for N in (4, 6, 8):
    gs = {'camino': nx.path_graph(N), 'ciclo par': nx.cycle_graph(N), 'estrella': nx.star_graph(N-1), 'K(N/2,N/2)': nx.complete_bipartite_graph(N//2, N//2)}
    gb = nx.cycle_graph(N); gb.add_edge(0, 2); gs['ciclo+cuerda (no bip)'] = gb
    for nm, G in gs.items():
        t0 = time.time(); a, b = trapped(G, N)
        print('N=%d %-22s n(T/2)=%.4f n(T)=%.4f  (%.0fs)'%(N, nm, a, b, time.time()-t0), flush=True)
