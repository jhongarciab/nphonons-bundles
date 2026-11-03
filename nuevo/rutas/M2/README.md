# Paso 1: la ventaja de Q de la molecula (bsc) con el MISMO lambda
repro.py reimplementa el metodo de bsc (pureza de Munoz, maestra local, barrido de Delta de 10 puntos +-0.2, parametros del paper: Om=0.2, gam=2e-4, gph=4e-4, J=0.5) y reproduce sus cifras de la molecula a lam=0.22: pi3 = 0.996 (Q909; paper 0.989), 0.977 (Q130; >0.97), 0.965 (Q91; >0.95).
Resultado (repro.out): a lam=0.22 el emisor unico da lo mismo: 0.992/0.979/0.969 (Q909/130/91). La cifra "Q~500 para 95 % en Bin" corresponde a lam=0.14 (0.957 a Q500), no a lam=0.22. La "ventaja 5x de Q" compara lam=0.22 (molecula) con lam=0.14 (emisor unico).
A lam=0.14 la molecula mejora algo (0.993 vs 0.956 a Q909; 0.932 vs 0.912 a Q130), pero un emisor unico con Om*sqrt2 da 0.980 / 0.923: la diferencia es casi toda el Rabi efectivo.
No hecho: version con maestra vestida (no necesaria para esta conclusion).
