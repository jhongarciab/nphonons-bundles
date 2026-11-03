"""Emisores en los nodos de un grafo; cada arista = reservorio compartido que acopla a sus dos extremos con amplitud 1.
Matriz de decaimiento Gamma = gamma * B B^T = gamma * Q (laplaciano sin signo). Tasas colectivas = gamma * autovalores de Q."""
import numpy as np, networkx as nx
def rates(G): 
    A = nx.to_numpy_array(G); Q = np.diag(A.sum(1)) + A; return np.linalg.eigvalsh(Q), A
N = 60; rng = np.random.default_rng(0)
fams = {'ER p=.1': lambda s: nx.erdos_renyi_graph(N, .1, seed=s), 'BA m=2': lambda s: nx.barabasi_albert_graph(N, 2, seed=s),
        'WS k=4 p=.1': lambda s: nx.connected_watts_strogatz_graph(N, 4, .1, seed=s), 'arbol aleatorio': lambda s: nx.random_labeled_tree(N, seed=s) if hasattr(nx,'random_labeled_tree') else nx.random_tree(N, seed=s),
        'ER bipartito': lambda s: nx.bipartite.random_graph(30, 30, .1, seed=s), 'completo': lambda s: nx.complete_graph(N)}
print('%-16s %-10s %-12s %-12s %-12s'%('familia','dark(0)','rate min','rate mediana','rate max'))
for nm, f in fams.items():
    mins=[]; ndark=[]; med=[]; mx=[]
    for s in range(20):
        G = f(s)
        w, A = rates(G); w = np.clip(w, 0, None)
        ncomp_bip = sum(nx.is_bipartite(G.subgraph(c)) for c in nx.connected_components(G))
        ndark.append((int((w < 1e-9).sum()), ncomp_bip)); mins.append(w[w > 1e-9].min() if (w > 1e-9).any() else 0); med.append(np.median(w)); mx.append(w.max())
    ok = all(a == b for a, b in ndark)
    print('%-16s dark=#bip:%-5s %-12.3f %-12.3f %-12.3f'%(nm, ok, np.mean(mins), np.mean(med), np.mean(mx)))
# cuasi-oscuro: grafo bipartito + k aristas intra-clase ; tasa minima vs k
G0 = nx.bipartite.random_graph(30, 30, .1, seed=1)
print('\nbipartito + k aristas extra que rompen la bipartitud (cada una en una de 20 realizaciones):')
for k in (0, 1, 2, 5, 10, 20):
    r=[]
    for s in range(20):
        G = G0.copy(); g = np.random.default_rng(s)
        cnt=0
        while cnt<k:
            i, j = g.choice(range(30), 2, replace=False)   # ambos en la clase 0 -> ciclo impar
            if not G.has_edge(i, j): G.add_edge(i, j); cnt+=1
        w,_ = rates(G); r.append(max(w.min(), 0))
    print('  k=%2d  tasa minima media/gamma = %.4f'%(k, np.mean(r)))
