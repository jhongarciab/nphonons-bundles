# Bin et al. (n=2) con las proporciones del dispositivo de referencia vs fuente de pares (rutas/P)
`bin_vs_pares.py` usa `nuevo/bundles/src/bundles_core.py` (maestra vestida, exacta): omega_b/2pi=2.257 GHz, gam=1/T1, gph=2/Tphi, kappa_m = 0.37 MHz (medido, sin guia) o 10 MHz (guia supuesta), Om<=0.1 wb (grid {0.005,0.02,0.05,0.1}; el flujo crece con Om, el optimo esta en el borde; el dispositivo ofrece Rabi~0.02 wb).
Resultados en `resultados.txt`. Fuente de pares (rutas/P): P(2)=0.72, pureza 0.80, flujo 1.1 MHz con reset activo (kappa_m=10.8 MHz supuesta).
Limitaciones: el lam de Bin es longitudinal; el dispositivo no lo tiene, se toma el cociente g/wb=0.0067 solo como escala hipotetica. Un solo juego de (T1,Tphi); Om no optimizado por encima de 0.1.
