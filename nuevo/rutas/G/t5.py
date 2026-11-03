import time, networkx as nx, trap2
for N in (8, 10, 12):
    t=time.time(); p = trap2.ptrap(nx.path_graph(N), N); print(N, 'camino', p, round(time.time()-t,1), 's', flush=True)
