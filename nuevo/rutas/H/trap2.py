"""Probabilidad de atrapamiento en el estado oscuro, exacta por sectores de excitacion (sin trayectorias):
H_eff conserva n; los saltos L_e=s_i+s_j bajan n en 1. sigma_n = int rho_n dt resuelve G sigma + sigma G = 2 S_n (G = sum L_e^T L_e);
S_{n-1} = sum_e L_e sigma_n L_e^T. P_trap = <D|S_1|D> con D el nucleo de Q en n=1. Inversion total inicial."""
import numpy as np, scipy.sparse as sp, itertools, networkx as nx, time
from math import comb
def sectors(N):
    idx = {}; states = {}
    for n in range(N+1):
        st = [sum(1 << i for i in c) for c in itertools.combinations(range(N), n)]; states[n] = st; idx[n] = {s: k for k, s in enumerate(st)}
    return states, idx
def lower(N, states, idx, n, i):   # sitio i, sector n -> n-1
    rows, cols = [], []
    for k, s in enumerate(states[n]):
        if s >> i & 1: rows.append(idx[n-1][s ^ (1 << i)]); cols.append(k)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(states[n-1]), len(states[n])))
def ptrap(G, N, cache={}):
    key = N
    if key not in cache: cache[key] = sectors(N)
    states, idx = cache[key]
    edges = list(G.edges)
    S = np.array([[1.0]])           # S_N: estado totalmente excitado
    for n in range(N, 1, -1):
        D = {i: lower(N, states, idx, n, i) for i in range(N)}
        Le = [D[i] + D[j] for i, j in edges]
        Gam = sum((L.T @ L) for L in Le).toarray()
        g, U = np.linalg.eigh(Gam); St = U.T @ S @ U
        sig = 2*St/(g[:, None] + g[None, :]); sig = U @ sig @ U.T
        S = sum(L @ sig @ L.T for L in Le)
        S = np.asarray(S)
    # n = 1: Q = Gamma_1
    D1 = {i: lower(N, states, idx, 1, i) for i in range(N)}
    Q = sum(((D1[i]+D1[j]).T @ (D1[i]+D1[j])) for i, j in edges).toarray()
    g, U = np.linalg.eigh(Q); null = U[:, g < 1e-9]
    return float(np.trace(null.T @ S @ null)), null.shape[1]
if __name__ == '__main__':
    for N in (4, 6):
        for nm, G in {'camino': nx.path_graph(N), 'ciclo par': nx.cycle_graph(N), 'estrella': nx.star_graph(N-1), 'K(N/2,N/2)': nx.complete_bipartite_graph(N//2, N//2)}.items():
            print(N, nm, ptrap(G, N))
    G = nx.cycle_graph(6); G.add_edge(0, 2); print('ciclo+cuerda', ptrap(G, 6))
