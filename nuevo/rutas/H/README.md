# Pruebas Mpemba (A: emisores en grafos, B: oscilador no lineal)
## B_kerr.py: espectro de Liouville exacto (dim 24), kappa=1
Modos lentos (Re = -0.5, coherencias |0><1|, |1><0|) y -1. Fock n=4: coeficiente 0 en los modos de Re=-0.5 en los 4 casos (lineal, Kerr U=0.7, perdida de 1 y 2 fotones, Kerr+2 fotones): selector exacto por covariancia U(1) (los estados diagonales en Fock no tienen coherencias).
Coherente a=2: coeficiente 2.0 (lineal) -> 0.14 (Kerr U=0.7) -> 1.14 (2 fotones) -> 0.41 (ambos); nunca cero. El Kerr no cambia el criterio (Fock mas rapido que coherente), solo reduce la amplitud de la cola lenta del coherente (x14).
Conclusion: extension del Mpemba bosonico de Longhi sin comportamiento nuevo; resultado nulo/incremental.
## A_emitters.py: emisores en grafos no bipartitos, reservorio por arista
Coeficiente asintotico del modo lento c: n(t)~c e^{-g t}; validado contra mesolve (N=6, t=20: 1.3396e-4 en ambos). NOTA: una primera version con la llegada integrada (sin ponderar e^{g s}) estaba mal y fue descartada tras la validacion.
Inversion total (c_full ~0.33-0.37) vs |i> (c_i) y W: la inversion total relaja mas rapido que |i> solo en 8-16 % de los pares (grafo,nodo) en ER/BA/WS (33-57 % en el ciclo impar), nunca que W (c_W 0.02-0.09). Sin Mpemba fuerte (c_full != 0).
