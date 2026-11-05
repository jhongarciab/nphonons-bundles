# Segunda parte de la tesis: captura directa de un qubit fononico itinerante en dos excitones oscuros con puerto fijo (modelo efectivo)
Modelo: una excitacion por rama, fuente en cascada -> cavidad (kappa_e fijo, kappa_i=1e-3) -> estados oscuros d1,d2 con acoplo Raman efectivo G(t)=Om(t) g/Delta_I (G_max=0.012 wb ~ Om 0.3 wb), desfase comun gph=4e-4 (chi=1), fuga fuera de resonancia entre ranuras (Dss=0.25).
Archivos: captura.py (un paquete, un estado oscuro; condicion analitica), qubit.py (dos intervalos con coherencia entre ramas), comparacion.py/.out, robustez.py/.out.
## Resultados
1. Condicion analitica de captura perfecta con puerto fijo y sin perdidas: G(t) = -sqrt(kappa) d(delta)/dt / u(t), delta^2(t) = int_0^t u^2 - u^2/kappa >= 0 (el paquete no puede crecer mas rapido que e^{kappa t}). Derivacion propia; no contrastada con la literatura de captura en cavidad.
2. Un paquete gaussiano (Tp=100-200): sin perdida intrinseca la captura llega a 0.98-1.00; con kappa_i/kappa_e=0.1 y G_max=0.012 queda en 0.89-0.91 (limite por perdida intrinseca; el control optimizado no mejora al ideal).
3. Dos intervalos, misma entrada para ambos protocolos, separacion = ventana de carga+190(pi)+100 (minima para dos etapas):
   gauss Tp=60/100/150: directa P=0.82/0.88/0.88 ; dos etapas conmutable 0.41/0.25/0.13 ; dos etapas puerto fijo 0.19/0.12/0.07.
   exp creciente (forma optima del dos-etapas) Tp=100: directa 0.78 ; dos etapas conmutable 0.81 ; fijo 0.35.  Tp=200: directa 0.77 ; conmutable 0.69 ; fijo 0.30.
   => con puerto fijo el protocolo de dos etapas es mucho peor; con puerto conmutable y forma de entrada optima (exp. creciente) el de dos etapas es comparable o algo mejor (0.81 vs 0.78 a Tp=100); la ventaja directa depende de la forma del paquete.
4. Coherencia: a igual separacion V_c es igual en ambos protocolos (lo fija la entrada). La captura directa admite paquetes mas cercanos: gauss Tp=100: sep=450 -> V_c=0.93 (P0=0.80, fuga 5.5e-3), 600 -> 0.90, 890 -> 0.85 (F_h~(1+V_c^2)/2 = 0.93/0.91/0.86; aproximacion: lectura simetrica, P0~P1, sin decoherencia durante la espera).
5. Selectividad: poblacion en la ranura no direccionada <=5.5e-3 (sep=450), 3e-5 (sep=890).
6. Robustez (gauss Tp=100, G optimizado, eta ref 0.892): tiempo de llegada +-30 -> 0.873 ; amplitud +-10 % -> 0.87 ; desintonia paquete-cavidad +-0.45 kappa -> 0.77, +-0.9 kappa -> 0.53 (la mas critica) ; compensacion Stark incompleta 10 % -> 0.874 ; forma sech 0.885, exponenciales 0.44-0.45 (hay que reoptimizar G).
## Limites (importante)
Modelo efectivo con eliminacion adiabatica del brillante; NO validado contra el modelo Raman completo (G,B,D) con fuente en cascada; chirp ideal que compensa el corrimiento Stark; solo una excitacion; lectura (emision) no simulada: se asume simetrica por reciprocidad; sin tiempo de espera; fuga entre ranuras con razon g2/g1 aproximada; separacion entre paquetes dependiente de la forma. Los numeros F_h son estimaciones de la visibilidad de captura, no del canal completo.

## Actualizacion: validacion contra el modelo completo y canal completo (validar_completo.py, canal.py)
Punto 1, modelo Raman COMPLETO (exc 4 niveles x modo NC=3 x fuente en cascada, chirp ideal que compensa el corrimiento Stark, G(t) del modelo efectivo mapeado a Om(t)=G Delta_I/g1; validar.out): paquete gaussiano Tp=100 a D1: eta completo = 0.830 (escala G x1.0), 0.831 (x0.9), 0.793 (x1.1) frente a 0.892 del efectivo -> el efectivo sobreestima ~0.06 (7 %). No se reoptimizo el control en el modelo completo.
Errores corregidos en el camino: (a) en el modelo completo la fuente no tenia la energia w del modo (daba eta=0.000); (b) en el canal el acoplo del sumidero debe usar la energia ACUMULADA, gamma_c = w^2/(eps + int_0^t w^2) (la primera version usaba la restante y daba s~0.25); (c) la comparacion de dos etapas tenia ventanas solapadas (ya corregido arriba).
Punto 2, canal completo en el efectivo (captura directa de los dos paquetes, espera, lectura con G invertido, sumideros adaptados; canal.out), gauss Tp=100, ke=1e-2, ki=1e-3:
  separacion 450: s=(P0+P1)/2 = 0.683, V=0.867, F_h=(1+V)/2=0.934 ; separacion 890: s=0.754, V=0.725, F_h=0.862. Espera 0 o 2000: identico (chi=1, estado oscuro sin perdida en el modelo).
  Consistencia: V(890)=0.725 ~ V_c^2 = 0.851^2 (lectura simetrica).
  Con la correccion del modelo completo (~0.93 por etapa) s bajaria a ~0.59 y ~0.66 (estimacion, supone el mismo error en la lectura).
  Referencia (memoria, dos etapas, puertos reales, otras condiciones): s~0.7, F_h 0.83-0.88.
Limites: lectura y canal solo en el modelo efectivo; sin biexciton; sin decaimiento del estado oscuro durante la espera; control no reoptimizado en el modelo completo; F_h=(1+V)/2 supone P0~P1.

## Calculos en paralelo (mapa.py, validar_D2.py, canal2.py)
1. Mapa de captura (gauss, ki=1e-3, Gmax=0.012; mapa_a.out, mapa_b.out): la eficiencia optimizada converge a kappa_e/(kappa_e+kappa_i) para paquetes largos: ke=0.005: 0.83 (Tp=250); 0.01: 0.909; 0.02: 0.952; 0.04: 0.975 (Tp>=100). Con paquetes cortos (Tp=50) cae (0.59-0.93) porque G_max limita; con ke pequeno hace falta Tp grande (ke=0.005 -> 0.75 a Tp=100).
2. Segundo estado oscuro en el modelo completo (validar_D2_*.out; g2=0.0613, Delta_I=1.6, C=0.0282, Gmax=0.011): eta completo = 0.829/0.844/0.822 (escala G x0.9/1.0/1.1) frente a 0.892 efectivo -> sobreestimacion ~0.05.
3. Canal completo con la MISMA entrada y el mismo ruido (canal2.out; ke=1e-2, ki=1e-3, Tp=100; sep minima del dos-etapas): resumen con chi=1:
   entrada gauss : directa s=0.754 F_h=0.862 ; dos etapas (conmutable) s=0.198 F_h=0.857 ; pasivo s=0.096 (T=0), 0.013 (2000), 0 (8000).
   entrada exp_gro: directa s=0.586 F_h=0.830 ; dos etapas s=0.643 F_h=0.843 ; pasivo s=0.217 (T=0), 0.029 (2000), 0 (8000).
   La espera no cambia s ni F_h con chi=1 (estado oscuro sin perdida en el modelo); con chi<1 F_h cae igual en ambos protocolos: gauss directa chi=0.9: 0.843/0.821/0.763 (T=0/2000/8000); chi=0: 0.710/0.608/0.515.
   Lectura: frente al pasivo la memoria oscura conserva s a esperas largas; frente al dos-etapas la ventaja de la captura directa depende de la forma de la entrada: gana mucho con paquetes gaussianos y no gana (0.59 vs 0.64) con el paquete exponencial creciente que el dos-etapas necesita; con puerto fijo el dos-etapas falla.
Limites: modelo efectivo (sobreestima ~0.05-0.06 por captura segun el modelo completo; en el canal se aplicarian dos etapas); control no reoptimizado en el completo; sin transferencia de poblacion d1<->d2 por ruido local; el pasivo es analitico simple con primer bin esperando 'sep' mas.

## Reoptimizacion en el modelo completo y figuras
validar_opt.out: rejilla 27 puntos (escala, retraso, desintonia) para D1 gauss Tp=100: mejor eta=0.831 (escala 0.9, retraso 0, desintonia 0) vs 0.830 sin reoptimizar; desintonia +-0.004 -> 0.68-0.71. Sin mejora: el deficit de ~0.06 respecto del efectivo es una perdida real, no calibracion. Figuras: figs_C2_v2.py (fig1_limites_captura, fig2_canal_completo, fig3_dinamica_robustez); datos: mapa_parte0/1.npy, chi_V0.npy, canal2.out.
