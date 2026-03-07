# Plan operativo — baseline 1QD (Bin + Muñoz Eq. 20)

## Objetivo inmediato
Construir un baseline confiable en 1QD para comparar:
1. **Pureza tipo Bin** (definición por poblaciones/MC, referencia objetivo).
2. **Aproximación tipo Muñoz Eq. 20** (indicador `T_N`) para evaluar cuándo aproxima bien.

## Estructura de trabajo
- `01_baseline_1QD/`:
  - script exacto/MC para pureza tipo Bin en 1QD.
  - versión rápida (grilla chica) + checkpoints.
- `02_validacion_TN_vs_PiN/`:
  - script de comparación `T_N` vs `Pi_N` en el mismo grid/parámetros.
  - métricas de error y mapas de diferencia.
- `03_2QD_extension/`:
  - se usa solo cuando 1QD esté validado.

## Criterio de salida de fase 1QD
- Reproducir cualitativamente mapa de Bin (región de alta pureza y su localización).
- Demostrar numéricamente en 1QD para N=2 y N=3:
  - correlación alta entre `T_N` y `Pi_N` en régimen de interés.
  - error controlado en región de operación (umbral definido por el equipo).

## Nota
No mezclar 1QD y 2QDs en la misma corrida ni en archivos de salida.
Etiquetar siempre modelo y métrica en nombre de archivo.
