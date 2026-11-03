# Memoria de 3 puntos vs rectangulo de 4 puntos (A2): resultado negativo

`mem.py`: modelo reducido autocontenido (1 exciton + modo, carga/lectura ideales, propagadores exactos). Validado cualitativamente contra memoria/ (3 puntos).
T=2000, puertos ideales (F_h / s):
- 3 puntos: chi=1 0.94/0.79; chi=0 0.68/0.64
- 4 puntos (rectangulo, ranuras -0.34 y -0.16): chi=1 0.93/0.71; chi=0 0.70/0.49

A2 no elimina el requisito chi~1 en esta configuracion. Limites: pulsos optimizados solo en sistema cerrado, una eleccion de ranuras, una sola T. Pulsos en `pulses4.json`.
