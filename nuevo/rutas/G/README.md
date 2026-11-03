# Bloqueo de fotones / g2 en redes complejas (forzamiento debil, Kerr debil)
Metodo: amplitudes de 1 y 2 fotones (`g2net.py`), validado: N=1 exacto; dimero vs QuTiP (nodo 1: 0.92226 vs 0.92225; nodo 0: 0.0025 vs 0.0035, diferencia por F finito donde g2~0).
Parametros: U=0.1 kappa, N=30, J escalado por el autovalor maximo de la adyacencia.
Resultado (study1/study2): con no linealidad debil la mediana de g2 es 0.6-0.8 en todos los grafos; el grafo completo y el hub de la estrella no dan antibunching en ningun nodo;
los pocos nodos con g2<0.1 tienen poblacion 0.1-0.6 % (anillo: 1e-7) de la del nodo mas poblado, es decir, son zeros de interferencia sin flujo util.
Conclusion: en este regimen la topologia no da un bloqueo controlable; resultado negativo. No explorado: no linealidad fuerte (U>=kappa), forzamiento no debil, emisores JC, otros tamanos.
