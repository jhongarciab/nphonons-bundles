# Ronda 14 — esquema de Ma, Xie y Li (Tareas 39-43) y reconciliación con Gautier et al. (Tarea 44)

Carpeta `validacion_ma/` (QuTiP 5.3.1, Python 3.11, `requirements.txt`, venv propio `.venv`); Tarea 44 en `../validacion/` (QuTiP 4).
Tablas: `tarea39_40_resultados.md`, `tarea41_resultados.md`, `tarea42_resultados.md`, `tarea43_resultados.md`, `../validacion/tarea44_resultados.md`.
Figuras: `tarea39_curvas.png`, `tarea40_resonancia.png`, `tarea42_kappa1_kappa2.png`, `../validacion/tarea44_gamma_bf.png`.
Código: `ma_model.py`, `ma2.py`, `tarea39_worker.py`, `tarea40_worker.py`, `tarea41_delta1.py`, `tarea42_worker.py`, `tarea43_worker.py`, `run_ma.sh`, `run_new.sh`, `run_t43.sh`; `../validacion/modelo_gautier.py`, `gautier_worker.py`.

## Tarea 39 — estado final del modelo completo de Ma (N=22, Floquet sobre T=4π/w_p, potencias por cuadrados)
- Se reproducen los valores previos. A t=30/Γ exacto: w_p=12 → P_c=0.7957, paridad +0.729, P_e=0.125, |⟨a²⟩|=2.441; w_p=11.98 → 0.9961, +0.387, 0.0084, 3.985. Con muestreo estroboscópico (t=nT) P_e, paridad y |⟨a²⟩| difieren por el micromovimiento (a 11.98, P_e oscila 0.003–0.010 dentro del período); P_c no depende de la fase.
- w_p=12.0: P_c satura en 0.81 (0.8118 a Γt=300), sin ventana P_c>0.99. w_p=11.98: P_c=0.9975 a Γt≥67, F máx≈0.72 (Γt≈13) y luego F→0.50, paridad→0.002: el estado final es la mezcla de |±2i⟩ (P_c=0.9975), no el gato par.
- Tasa de decaimiento de la paridad (w_p=11.98, ventana Γt=26–152, P_c>0.99): 3.309e-4 frente a la predicción 3.333e-4 (razón 0.99).
- Convergencia N=22→26 (Γt=60): P_c 5e-6, |⟨a²⟩| 1e-5, P_e 7e-4, F 4e-3, paridad 2.4% (de un valor pequeño, 0.2). Validaciones de ρ: |Tr ρ−1|≤9e-12, ‖ρ−ρ†‖≤4e-13, mín autovalor −8e-16.
**Una línea:** el modelo completo reproduce las cifras de Ma y la predicción de la tasa de phase-flip al 1%; con w_p=12 no hay gato confinado y con 11.98 el estado final es una mezcla de |±2i⟩ de pureza P_c=0.9975.

## Tarea 40 — barrido de w_p (Γt≈60; 15 puntos pedidos + 5 extra hasta 12.06 para cerrar el ancho)
- Máximo de P_c=0.9975 (0.9980 por spline) en w_p=11.9806 frente a la predicción 11.9800 (dif. 6e-4, menor que el paso 4.3e-3). P_e mínimo 0.0026.
- FWHM (spline): 11.9688–12.0130, es decir 0.044; asimétrico (0.012 a la izquierda, 0.032 a la derecha).
**Una línea:** la resonancia dressed está en w_p=11.9806 (predicción 11.980) y tiene FWHM≈0.044.

## Tarea 41 — δ₁ frente a κ/ω_m (modelo_comun, g_x=6, ω_m=1000)
- δ₁/(−4g_x²/(3ω_m)) = 1 − (7/9)(κ/2ω_m)² (exacto a 4 cifras): 0.999999998 en κ/ω_m=1e-4 y 0.99806 en 1e-1; tiende a 1 cuando κ/ω_m→0. Con los parámetros de Ma (κ/w=0.005) da w_p*=11.980000. Convenciones documentadas en `tarea41_resultados.md` (mismo δ₁; resonancia Naseem δ_m=−δ₁, Ma w_p/2=w+δ₁).
**Una línea:** δ₁ converge al valor de Ma −4g_x²/(3ω) con corrección cuadrática (7/9)(κ/2ω)².

## Tarea 42 — figura de mérito κ₁/κ₂ (α²=4, N=22, re-sintonizado)
- **κ₁/κ₂ = (5/72)(κ/g_z)² se verifica:** razón medido/predicho entre 0.963 y 1.029 (media 1.004) en 21 puntos, con g_x∈{0.03…0.2}, w∈{4,6,8}, g_z/κ∈{2…12}; independiente de g_x y w (barrido (c): 0.991–1.016).
- Adiabaticidad (κ₂/κ<0.1): se cumple en (a) salvo g_z/κ=12 (0.16); en (b) solo g_z/κ=2; los puntos saturados (κ₂/κ hasta 2.6) siguen dentro del 4% de la predicción (0.963 en el peor: g_x=0.2, g_z/κ=12): la fórmula deja de ser exacta muy lentamente al saturar.
- Único punto sin medida: g_x=0.2, g_z/κ=2, donde P_c máx=0.989 (<0.99): con κ₁/κ₂≈1.7e-2 el gato no supera el 99%.
- κ₁/brecha física no satura: baja de 4.6e-3 (g_z/κ=2) a 5.8e-4 (12) en (a) y de 3.9e-2 a 7.8e-3 en (b); la brecha crece con g_z pero se estanca/decae para κ₂/κ≳0.25–1 (barrido (c), g_x=0.2,w=4: 1.9e-3 < 4.1e-3 con g_x=0.1).
- Convergencia N=22→28 (g_x=0.05, g_z/κ=12): tasa de paridad 5e-5, P_c 5e-8, brecha 2.7%. Validación: |Tr ρ−1| llega a 3.3e-10 (excede la tolerancia 1e-10 por acumulación en potencias de U); ‖ρ−ρ†‖≤2.4e-11 y autovalores ≥−1e-15 cumplen.
- Nota metodológica: el umbral de peso de borde de la Tarea 38 (1e-3) es demasiado estricto para |α|²=4 (los modos físicos llegan a ~2e-2); aquí se usó 0.1 (los espurios de Naseem pesaban ~1).
**Una línea:** κ₁/κ₂=(5/72)(κ/g_z)² es válida al 4% en todo el rango probado, independiente de g_x y w, y la brecha (no κ₂) es la que se satura.

## Tarea 43 — baño filtrado
(ver `tarea43_resultados.md`; sección añadida al terminar la corrida)

## Tarea 44 — reconciliación con Gautier et al. (buffer de dos niveles, baño térmico)
- Piso numérico: con n_th=0 y κ₁=0 el bit-flip es exactamente 0, pero el truncamiento deja γ_bf~5e-9…1e-4 (∝g₂²); los valores a <3× de ese piso están marcados † en la tabla.
- γ_bf es **lineal en n_th** (desviación <10% hasta n_th=0.1) con γ_bf/(n_th κ)=0.002 (g₂/κ=0.05), 0.009 (0.1), 0.13 (0.3), 0.68 (1), 0.95 (3): ∝(g₂/κ)^2.3 hasta ~0.3 y saturado ~1 por encima de 1.
- **Supresión exponencial con |α|²:** se mantiene en todo el rango pero con pendiente d ln γ_bf/dα²≈−1.0…−1.4 para g₂/κ≤0.1, ≈−0.55 en 0.3, −0.4…−0.55 en 1 y −0.1…−0.2 (α²=2→4) en 3: se debilita al subir g₂/κ.
- Compatibilidad: (i) Gautier (supresión mantenida con factor constante por debajo del punto de trabajo) vale para g₂/κ≲0.1, donde la pendiente es constante y γ_bf∝n_th; (ii) Tareas 36-37 (γ_bf≈0.05 n_q κ a Γ₂/κ=0.13, g₂/κ=0.18) coincide con la interpolación 0.038 entre g₂/κ=0.1 y 0.3; vale para g₂/κ≈0.2–0.3 y n_q≲0.1; (iii) para g₂/κ≳1 cada excitación térmica del buffer produce un bit-flip con probabilidad O(1) (γ_bf≈n_th κ) y la protección con α² es mucho más débil.
**Una línea:** la supresión exponencial y la linealidad en n_th valen para g₂/κ≲0.1–0.3; por encima, γ_bf≈n_thκ y la ventaja de α² se pierde.
