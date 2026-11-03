"""Ciclo de N emisores con fase phi en una arista: L_e = s_i + exp(i phi) s_j (flujo sintetico). Excitacion retenida (sector 1) a tiempos largos."""
import numpy as np, networkx as nx, trap2
def spec_phase(N, edges, phases):
    states, idx = trap2.sectors(N); S = np.array([[1.0+0j]])
    def Ls(n):
        D = {i: trap2.lower(N, states, idx, n, i) for i in range(N)}; return [D[i] + np.exp(1j*p)*D[j] for (i, j), p in zip(edges, phases)]
    for n in range(N, 1, -1):
        L = Ls(n); Gam = sum((l.conj().T@l) for l in L).toarray(); g, U = np.linalg.eigh(Gam)
        sig = U @ (2*(U.conj().T@S@U)/(g[:, None]+g[None, :])) @ U.conj().T; S = np.asarray(sum(l@sig@l.conj().T for l in L))
    L = Ls(1); Q = sum((l.conj().T@l) for l in L).toarray(); g, U = np.linalg.eigh(Q)
    w = np.real(np.einsum('ia,ij,ja->a', U.conj(), S, U)); return g, w
for N in (8, 9):
    edges = list(nx.cycle_graph(N).edges); print('C%d (dark a flujo 0 si par, pi si impar)'%N)
    for phi in (0, 0.1, 0.3, 0.6, 1.0, 1.57, 2.5, np.pi):
        ph = [0]*(len(edges)-1) + [phi]; g, w = spec_phase(N, edges, ph)
        print('  flujo=%.2f  tasa_min(Q)=%.4f  exc.retenida t=50: %.4f  t=200: %.4f  t=1000: %.4f'%(phi, g.min(), *[float(np.sum(w*np.exp(-g*t))) for t in (50, 200, 1000)]))
