import numpy as np, networkx as nx, g2net
U=0.1
for name, G, Jt in (('WS k=4 p=.1', nx.connected_watts_strogatz_graph(30,4,.1,seed=6), 2.0), ('ER p=.15', nx.erdos_renyi_graph(30,.15,seed=4), 2.0), ('star(leaf drive)', nx.star_graph(29), 2.0), ('ring',nx.cycle_graph(30),1.0)):
    if not nx.is_connected(G): G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)))
    A = nx.to_numpy_array(G); n=len(A); lam=np.linalg.eigvalsh(A).max(); deg=A.sum(1)
    d = int(deg.argmin()) if 'leaf' in name or 'ring' in name else int(deg.argmax()); J=Jt/lam
    rows=[]
    for dl in np.linspace(-1.5*Jt-1,1.5*Jt+1,241):
        g,n1=g2net.g2_all(A,d,dl,U,J)
        # poblacion relativa al nodo con maxima poblacion a ese delta
        for i in range(n):
            if g[i]<0.1: rows.append((g[i], n1[i]/n1.max(), dl, i))
    rows.sort()
    print(name, 'd=',d,'n puntos (nodo,delta) con g2<0.1:', len(rows))
    for r in rows[:6]: print('   g2=%.4f  n_rel=%.2e  delta=%.2f nodo=%d'%r)
    if rows:
        print('   mejor n_rel entre los de g2<0.1: %.2e ; con g2<0.3 y n_rel>0.05?'%max(r[1] for r in rows))
