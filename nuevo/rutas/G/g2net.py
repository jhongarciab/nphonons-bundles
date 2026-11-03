"""g2 estacionaria a forzamiento debil de cavidades Kerr sobre un grafo. Amplitudes de 1 y 2 fotones (H efectivo no hermitico).
H = sum_i w a+a + U a+a+aa + J sum_ij A_ij a_i+ a_j + F(a_d+ + h.c.); perdida kappa. delta = w - wL. Unidades kappa=1."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, networkx as nx
def g2_all(A, d, delta, U, J, kappa=1.0):
    N = A.shape[0]; I = sp.identity(N, format='csr')
    M = (delta - 0.5j*kappa)*I + J*sp.csr_matrix(A)
    e = np.zeros(N, complex); e[d] = 1
    c = spl.spsolve(M.tocsc(), -e)                      # F=1
    S = np.zeros((N, N), complex); S[:, d] += c; S[d, :] += c
    Kr = sp.kron(M, I) + sp.kron(I, M)
    diag = np.arange(N)*N + np.arange(N)
    P = sp.csr_matrix((np.ones(N), (diag, diag)), shape=(N*N, N*N))
    C = spl.spsolve((Kr + 2*U*P).tocsc(), -S.reshape(-1)).reshape(N, N)
    n1 = np.abs(c)**2
    return np.abs(np.diag(C))**2/np.maximum(n1**2, 1e-300), n1
if __name__ == '__main__':
    # prueba N=1: g2=|M|^2/|M+U|^2
    A = np.zeros((1, 1)); g, _ = g2_all(A, 0, 0.3, 2.0, 1.0)
    M = 0.3 - 0.5j; print('N=1', g[0], abs(M)**2/abs(M+2.0)**2)
    # prueba dos cavidades: no-lineal debil, U=0.1 kappa, barrer delta y J
    A = np.array([[0, 1], [1, 0.]])
    best = (9, None)
    for J in np.linspace(0.2, 5, 60):
        for dl in np.linspace(-3, 3, 241):
            g, n = g2_all(A, 0, dl, 0.1, J); 
            if g[0] < best[0]: best = (g[0], (J, dl))
    print('2 sitios, drive en 0, U=0.1: min g2 en nodo 0', best)
