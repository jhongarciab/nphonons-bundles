import time, networkx as nx, numpy as np, trap2, json
res = {}
def rtree(N, s): return nx.from_prufer_sequence(list(np.random.default_rng(s).integers(0, N, N-2)))
def rbip(N, s):
    for k in range(200):
        g = nx.bipartite.random_graph(N//2, N-N//2, .35, seed=s*1000+k)
        if nx.is_connected(g): return g
fams = {'camino': lambda N: nx.path_graph(N), 'ciclo par': lambda N: nx.cycle_graph(N), 'estrella': lambda N: nx.star_graph(N-1),
        'K(N/2,N/2)': lambda N: nx.complete_bipartite_graph(N//2, N//2), 'arbol aleatorio': lambda N: rtree(N, 7), 'bipartito aleatorio': lambda N: rbip(N, 3)}
for N in (6, 8, 10, 12, 14):
    for nm, f in fams.items():
        t = time.time(); G = f(N); p, nd = trap2.ptrap(G, N)
        res[f'{nm}|{N}'] = (p, nd, G.number_of_edges()); print('N=%2d %-20s P_trap=%.4f dim_oscuro=%d aristas=%d (%.0fs)'%(N, nm, p, nd, G.number_of_edges(), time.time()-t), flush=True)
json.dump(res, open('scale.json', 'w'))
