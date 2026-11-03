"""Test A: Mpemba en emisores sobre grafos (reservorio por arista). A tiempos largos n(t) ~ c e^{-g t}, g = tasa menor de Q.
c_full: peso de la inversion total en ese modo (llegada al sector 1 via Lyapunov por sectores); c_i = |u_i|^2 para |i> de 1 excitacion; c_W = |sum_i u_i|^2/N."""
import numpy as np, networkx as nx, trap2, qutip as qt
def spec(G, N):
    """coeficiente asintotico c_full del modo lento a de Q: <a|rho_1(t)|a> -> c e^{-g_a t}; usa la integral ponderada X = int e^{lam s} rho_n ds, lam=g_a,
    que cumple (Gam X + X Gam)/2 - lam X = Y + rho_n(0)."""
    states, idx = trap2.sectors(N); edges = list(G.edges)
    D1 = {i: trap2.lower(N, states, idx, 1, i) for i in range(N)}
    Q = sum(((D1[i]+D1[j]).T@(D1[i]+D1[j])) for i, j in edges).toarray(); g1, U1 = np.linalg.eigh(Q)
    lam = g1.min()
    if lam < 1e-9: raise ValueError('grafo bipartito (modo oscuro)')
    Y = np.array([[0.0]]); R0 = np.array([[1.0]])
    for n in range(N, 1, -1):
        D = {i: trap2.lower(N, states, idx, n, i) for i in range(N)}; Le = [D[i]+D[j] for i, j in edges]
        Gam = sum((L.T@L) for L in Le).toarray(); g, U = np.linalg.eigh(Gam)
        T = U.T@(Y+R0)@U; den = (g[:, None]+g[None, :])/2 - lam
        assert den.min() > 0, 'sin convergencia'
        X = U@(T/den)@U.T
        Y = np.asarray(sum(L@X@L.T for L in Le)); R0 = np.zeros_like(Y)
    a = int(np.argmin(g1)); u = U1[:, a]
    return g1, U1, np.einsum('i,ij,j->', u, Y, u)*np.ones(len(g1))*0 + np.array([float(np.einsum('i,ij,j->', U1[:, k], Y, U1[:, k])) if k == a else 0.0 for k in range(len(g1))])
def validate(N=6):
    G = nx.cycle_graph(N); G.add_edge(0, 2)
    g, U, w = spec(G, N); a = int(np.argmin(g))
    sm = [qt.tensor(*[qt.destroy(2) if k == i else qt.qeye(2) for k in range(N)]) for i in range(N)]; nt = sum(s.dag()*s for s in sm)
    cs = [sm[i]+sm[j] for i, j in G.edges]; psi = qt.tensor(*[qt.basis(2, 1)]*N); tl = np.array([20., 30., 40.])
    r = qt.mesolve(0*nt, psi*psi.dag(), tl, cs, e_ops=[nt], options=dict(atol=1e-11, rtol=1e-11, nsteps=10**6))
    print('validacion N=%d: n(t) mesolve'%N, r.expect[0], ' c e^{-g t} =', [w[a]*np.exp(-g[a]*t) for t in tl], ' (g=%.4f, c_full=%.4f)'%(g[a], w[a]))
validate(6)
rng = np.random.default_rng(1)
def fams(N):
    for s in range(40):
        g = nx.erdos_renyi_graph(N, .4, seed=s)
        if nx.is_connected(g) and not nx.is_bipartite(g): yield 'ER', g
    for s in range(20): yield 'BA m=2', nx.barabasi_albert_graph(N, 2, seed=s)
    for s in range(20):
        g = nx.connected_watts_strogatz_graph(N, 4, .3, seed=s)
        if not nx.is_bipartite(g): yield 'WS', g
    yield 'C9-like impar', nx.cycle_graph(N if N % 2 else N-1)
for N in (8, 9):
    print('N=%d'%N); tab = {}
    for nm, G in fams(N):
        n = G.number_of_nodes(); g, U, w = spec(G, n)
        if nx.is_bipartite(G): continue
        a = int(np.argmin(g)); u = U[:, a]; cfull = w[a]; ci = u**2; cW = (u.sum())**2/n
        tab.setdefault(nm, []).append((cfull, ci.min(), ci.max(), cW, np.mean(cfull < ci), cfull < cW, g[a]))
    for nm, v in tab.items():
        v = np.array(v, float)
        print('  %-14s #grafos=%2d  c_full medio=%.3f  c_i(min..max medios)=%.3f..%.3f  c_W medio=%.3f | frac(i: c_full<c_i)=%.2f  frac(c_full<c_W)=%.2f  g_min medio=%.3f'%(nm, len(v), v[:,0].mean(), v[:,1].mean(), v[:,2].mean(), v[:,3].mean(), v[:,4].mean(), v[:,5].mean(), v[:,6].mean()))
