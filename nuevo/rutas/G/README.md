# Bloqueo de fotones / g2 en redes complejas (forzamiento debil, Kerr debil)
Metodo: amplitudes de 1 y 2 fotones (`g2net.py`), validado: N=1 exacto; dimero vs QuTiP (nodo 1: 0.92226 vs 0.92225; nodo 0: 0.0025 vs 0.0035, diferencia por F finito donde g2~0).
Parametros: U=0.1 kappa, N=30, J escalado por el autovalor maximo de la adyacencia.
Resultado (study1/study2): con no linealidad debil la mediana de g2 es 0.6-0.8 en todos los grafos; el grafo completo y el hub de la estrella no dan antibunching en ningun nodo;
los pocos nodos con g2<0.1 tienen poblacion 0.1-0.6 % (anillo: 1e-7) de la del nodo mas poblado, es decir, son zeros de interferencia sin flujo util.
Conclusion: en este regimen la topologia no da un bloqueo controlable; resultado negativo. No explorado: no linealidad fuerte (U>=kappa), forzamiento no debil, emisores JC, otros tamanos.

## Barrido 2: no linealidad fuerte (study3.py), g2 minimo en el nodo forzado
Referencia de una cavidad sola: U=1k -> 0.172; U=5k -> 0.0098.
Redes (anillo, ER, BA, WS): g2 minimo entre 0.185 y 0.45 (U=1k) y 0.011-0.063 (U=5k): la red solo empeora el bloqueo (dilucion).
Grafo completo: igual a una cavidad (0.177/0.010). Unica mejora: hub de la estrella con J~=3: 0.117 (U=1k) y 0.006 (U=5k); sin explicar, no explorado.

## Barrido 3: emisores en los nodos, reservorio compartido por arista (dark.py)
Gamma = gamma*B B^T = gamma*Q (laplaciano sin signo). Comprobado en 6 familias x 20 realizaciones: #estados perfectamente oscuros = #componentes conexas bipartitas (incluye nodos aislados).
Tasa minima no nula (en unidades de gamma, N=60): ER 1.0, BA 0.58, WS 0.96, arbol 0.013, completo 58 (Dicke). Bipartito + k aristas que rompen la bipartitud: 0.003 (k=2), 0.027 (k=5), 0.082 (k=10), 0.30 (k=20).
Es modelo propio (reservorios por arista); novedad NO verificada. Excitacion unica solamente.

## Barrido 4: varias excitaciones (multi.py, trap.py), inversion total inicial, L_e=sqrt(g)(s_i+s_j) por arista
Nucleo comun de los L_e (estados oscuros en todos los sectores): bipartito -> dim 2 (vacio + 1 estado oscuro de 1 excitacion); no bipartito -> dim 1 (solo vacio). N=6.
Excitacion media atrapada a tiempo largo (N=4/6/8): camino 0.141/0.137/0.128; ciclo par 0.115/0.120; estrella 0.224/0.197; K(N/2,N/2) 0.115/0.123; ciclo+cuerda (no bip) 0.000.
Es decir ~12-22 % de probabilidad de quedar atrapado en el estado oscuro (n_inf = P_dark), casi independiente de N entre 4 y 8; cero en no bipartitos.
No hay rafaga superradiante: la intensidad es maxima en t=0 (reservorios independientes por arista).
N=8 solo completado para el camino (423 s); el resto no se corrio. Sin comprobar tamanos mayores.
