# Funcion de onda conjunta del par emitido, psi(t1,t2), y numero de Schmidt (psi2.py, psi2.out)
Modelo efectivo (A=|f,0> con perdida kq, B=|g,2> con salto a la guia 2 km, acoplo V=sqrt2 G; G=1). psi(t1<t2)=sqrt2 km beta(t1) exp(-km (t2-t1)/2); estado de dos bosones simetrizado; norma = eficiencia.
Validacion: eficiencia por integral = formula cerrada (0.9656 vs 0.9656 a kq/G=0.05, km=sqrt2 G).
Resultado: en el optimo km*=sqrt2 G el par es casi un solo modo temporal: K=1.15, peso del modo dominante 0.93 (kq/G=0.05, eta=0.966); K=1.12, 0.94 (kq/G=0.5, eta=0.72). Fuera del optimo K crece (K=3.5 a km=0.1 G; 33 a km=20 G).
Lectura: es un |2> bosonico indistinguible en un solo paquete, NO un biphoton entrelazado tiempo-energia. Un |2> preparado en el modo y liberado por fuga lineal da tambien un solo modo exacto (K=1); la preparacion secuencial con el mismo modo da el mismo tipo de estado.
Limites: modelo efectivo (eliminacion adiabatica), sin otros canales de |f>, sin dispersion de la guia, sin ruido de fase.
