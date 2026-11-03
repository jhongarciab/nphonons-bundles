import numpy as np, networkx as nx, g2net
def ref1(U):  # una cavidad sola: min sobre delta de |M|^2/|M+U|^2
    d=np.linspace(-6,6,2001); M=d-0.5j; return float((abs(M)**2/abs(M+U)**2).min())
gs = [('ring',nx.cycle_graph(30)),('ER p=.15',nx.erdos_renyi_graph(30,.15,seed=4)),('BA m=2',nx.barabasi_albert_graph(30,2,seed=5)),
      ('WS',nx.connected_watts_strogatz_graph(30,4,.1,seed=6)),('star',nx.star_graph(29)),('complete',nx.complete_graph(30))]
for U in (1.0, 5.0):
    print('U=%.1f kappa; una cavidad sola: min g2 = %.4f'%(U, ref1(U)))
    for name,G in gs:
        if not nx.is_connected(G): G=nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G),key=len)))
        A=nx.to_numpy_array(G); n=len(A); lam=np.linalg.eigvalsh(A).max(); deg=A.sum(1)
        out=[]
        for dn,d in (('hub',int(deg.argmax())),('leaf',int(deg.argmin()))):
            for Jt in (1.0,3.0):
                J=Jt/lam; best=9; bn=0
                for dl in np.linspace(-1.5*Jt-1,1.5*Jt+1,91):
                    g,n1=g2net.g2_all(A,d,dl,U,J)
                    if g[d]<best: best=g[d]; bn=n1[d]
                out.append('%s J~%.0f: %.3f'%(dn,Jt,best))
        print('  %-9s'%name, ' | '.join(out), flush=True)
