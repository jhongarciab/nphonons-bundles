import numpy as np, networkx as nx, g2net, json, sys
rng = np.random.default_rng(1)
N = 30; U = 0.1
def graphs():
    yield 'ring', nx.cycle_graph(N)
    yield 'ER p=.15', nx.connected_watts_strogatz_graph(N, 4, 1.0, seed=3) if False else nx.erdos_renyi_graph(N, .15, seed=4)
    yield 'BA m=2', nx.barabasi_albert_graph(N, 2, seed=5)
    yield 'WS k=4 p=.1', nx.connected_watts_strogatz_graph(N, 4, .1, seed=6)
    yield 'star', nx.star_graph(N-1)
    yield 'complete', nx.complete_graph(N)
res = {}
for name, G in graphs():
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy(); G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G); n = len(A); lam = np.linalg.eigvalsh(A).max()
    deg = A.sum(1); d_hub = int(deg.argmax()); d_leaf = int(deg.argmin())
    out = {}
    for dname, d in (('hub', d_hub), ('leaf', d_leaf)):
        for Jt in (0.5, 1.0, 2.0):
            J = Jt/lam; best = np.full(n, 9.0)
            for dl in np.linspace(-1.5*Jt-1, 1.5*Jt+1, 121):
                g, nn = g2net.g2_all(A, d, dl, U, J); best = np.minimum(best, g)
            out[f'{dname} J~={Jt}'] = dict(med=float(np.median(best)), min=float(best.min()), drivenode=float(best[d]), frac_lt_0p1=float((best < .1).mean()), frac_lt_0p5=float((best < .5).mean()))
    res[name] = dict(n=n, lam=float(lam), out=out)
    print(name, n, 'lam', round(lam, 2), flush=True)
    for k, v in out.items(): print('  ', k, {a: round(b, 3) for a, b in v.items()}, flush=True)
json.dump(res, open('study1.json', 'w'))
