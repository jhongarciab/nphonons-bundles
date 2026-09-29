# msc — paper de generalización (qubits gato con acoplamientos g_x, g_z)

Mapa de carpetas. Para el estado científico y los pendientes, ver `HANDOFF.md`.

```
msc/
├── README.md                 este mapa
├── HANDOFF.md                traspaso consolidado: modelo, derivaciones, resultados, pendientes (leer primero)
├── papers/                   PDFs de referencia (Ma 2019 y de Matos Filho 1996 solo en local, no se suben; Naseem 2026 y Dubovitskii 2024, arXiv)
├── docs/                     escritos LaTeX, uno por carpeta (vienen del chat)
│   ├── documento_completo/       documento v1 (.tex, .pdf, refs_doc.bib); NECESITA v2 (HANDOFF §12)
│   ├── brecha_punto_excepcional/ derivación exacta de la brecha (refs.bib reconstruida)
│   ├── lamb_shift_ma2019/        corrimientos y errores de Ma (refs_ma.bib)
│   ├── figura_merito/            κ₁/κ₂ = (5/72)(κ/g_z)² (refs_fm.bib)
│   ├── mapa_plataformas/         tabla de plataformas v1 (refs_mapa.bib reconstruida)
│   └── handoff_chat/             HANDOFF original del chat (referencia histórica)
└── Mechanical-Cat-State/
    ├── .venv/                QuTiP 4.7 (solo validacion/tareas_naseem)
    ├── .venv_qutip5/         QuTiP 5.3.1 (todo lo demás; usar este)
    ├── codigo_original/      código y figuras del repositorio original del artículo (fig1–5_git_*.py, figs/)
    ├── figuras_finales/      CÓDIGO OFICIAL del paper: figuras, datos y su documentación
    │   ├── README.md             métodos, parámetros, cifras y validaciones por figura
    │   ├── PENDIENTES_Y_HALLAZGOS.md
    │   ├── C6_polaron.md
    │   ├── figuras/              PDF y PNG finales (principal_*, figura_central*, figura_termica, apendice_*, fig2–4, p9_diagnostico)
    │   ├── codigo/               calc_* (cálculo con caché), run_* (lanzadores), *fig*.py (solo dibujan)
    │   └── data/                 cachés .npz y CSV
    └── validacion/           VALIDACIÓN (no es código del paper)
        ├── independiente/        verificación independiente V1–V6 y V4b de los resultados R1–R6 (reportes V*_reporte.md)
        ├── tareas_ma/            Tareas 39–47 del agente original (esquema de Ma; QuTiP 5)
        └── tareas_naseem/        Tareas 1–44 del agente original (esquema de Naseem; QuTiP 4)
```

## Qué sirve para qué
- **Para escribir el paper:** `HANDOFF.md`, `figuras_finales/README.md`, `figuras_finales/figuras/` y `docs/`.
- **Para reproducir una figura:** `cd Mechanical-Cat-State/figuras_finales && ../.venv_qutip5/bin/python codigo/<script>.py` (desde la caché, sin recálculo).
- **Para auditar resultados:** `validacion/independiente/` (verificación independiente) y `figuras_finales/PENDIENTES_Y_HALLAZGOS.md`.
- **Histórico** (solo consulta; varias conclusiones están superadas, ver HANDOFF §11): `validacion/tareas_ma/`, `validacion/tareas_naseem/` y `codigo_original/`.

## Notas de la reorganización (2026-09-29)
- `verificacion_independiente/` pasó a `validacion/independiente/`, `validacion_ma/` a `validacion/tareas_ma/` y `validacion/` a `validacion/tareas_naseem/`.
- Los scripts de figuras_finales ya apuntan a `.venv_qutip5`.
- Los scripts históricos de `tareas_*` conservan rutas relativas antiguas (por ejemplo `../.venv`, `../validacion`). Si se vuelven a ejecutar, hay que ajustarlas: el venv de QuTiP 4 queda ahora en `../../.venv`.
- `docs/documento_completo`: `fig2_val.pdf`, `fig3_val.pdf` y `fig2_wigner.pdf` son copias de `figuras_finales/figuras/` (`fig2`, `fig3`, `apendice_fig2_estados`). Hou 2024 no está disponible.
- Los PDFs de Ma 2019 y de Matos Filho 1996 no se suben al remoto (derechos de autor; están en `.gitignore`). Todos los docs compilan (pdflatex + bibtex, sin citas indefinidas).
