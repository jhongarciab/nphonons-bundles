import numpy as np, networkx as nx, trap2, itertools
def spec(G, N):
    states, idx = trap2.sectors(N); edges = list(G.edges); S = np.array([[1.0]]); gmins = {}
    for n in range(N, 1, -1):
        D = {i: trap2.lower(N, states, idx, n, i) for i in range(N)}; Le = [D[i]+D[j] for i, j in edges]
        Gam = sum((L.T@L) for L in Le).toarray(); g, U = np.linalg.eigh(Gam); gmins[n] = g.min()
        sig = U @ (2*(U.T@S@U)/(g[:, None]+g[None, :])) @ U.T; S = np.asarray(sum(L@sig@L.T for L in Le))
    D1 = {i: trap2.lower(N, states, idx, 1, i) for i in range(N)}
    Q = sum(((D1[i]+D1[j]).T@(D1[i]+D1[j])) for i, j in edges).toarray(); g, U = np.linalg.eigh(Q)
    w = np.einsum('ia,ij,ja->a', U, S, U); return g, w, gmins
def odd_chords(G, k, rng):
    """aÃ±ade k aristas que crean ciclos impares (entre nodos a distancia par)"""
    G = G.copy(); d = dict(nx.all_pairs_shortest_path_length(G)); cand = [(i, j) for i in G for j in G if i < j and not G.has_edge(i, j) and d[i][j] % 2 == 0]
    for e in [cand[t] for t in rng.permutation(len(cand))[:k]]: G.add_edge(*e)
    return G
N = 10; rng = np.random.default_rng(0)
for nm, base in (('ciclo C10', nx.cycle_graph(N)), ('camino P10', nx.path_graph(N))):
    print(nm)
    for k in (0, 1, 2, 3, 5, 8):
        gs_ = []; 
        for rep in range(5 if k else 1):
            G = odd_chords(base, k, rng); g, w, gm = spec(G, N)
            nz = g > 1e-9; gmin = g[nz].min() if k else 0.0
            tail = lambda t: float(np.sum(w*np.exp(-g*t)))
            gs_.append((gmin if k else g[1], tail(50), tail(200), tail(1000), min(gm.values())))
        a = np.mean(gs_, axis=0)
        print('  k=%d  Q_menor_tasa=%.4f  exc.retenida(sector1) t=50: %.4f  t=200: %.4f  t=1000: %.4f  | tasa min sectores n>=2: %.3f'%(k, *a))
