import numpy as np, qutip as qt, g2net
N=2; nc=5; a=[qt.tensor(*[qt.destroy(nc) if k==i else qt.qeye(nc) for k in range(N)]) for i in range(N)]
J, dl, U, F = 2.0, 0.25, 0.1, 0.01
H = sum(dl*x.dag()*x + U*x.dag()*x.dag()*x*x for x in a) + J*(a[0].dag()*a[1]+a[1].dag()*a[0]) + F*(a[0]+a[0].dag())
rho = qt.steadystate(H, [a[0], a[1]])
for i in range(N):
    n = qt.expect(a[i].dag()*a[i], rho); g = qt.expect(a[i].dag()**2*a[i]**2, rho)/n**2; print('qutip nodo', i, 'g2', g)
g, n = g2net.g2_all(np.array([[0,1],[1,0.]]), 0, dl, U, J); print('debil  g2', g)
