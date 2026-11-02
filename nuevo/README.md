# nuevo/ — línea de bundles de fonones (desarrollo autónomo, 2026-11-02)

Leer primero: **`DOSSIER_bundles.md`** (veredicto, resultados con su verificación, qué no afirmar, esquema del artículo).

| Carpeta | Contenido |
|---|---|
| `bundles/` | Biblioteca (`src/bundles_core.py`, `bundles_mc.py`), pruebas (`tests/run_tests.py`), datos (`data/*.npz`, `tablas.md`), informe de datos (`informe_WP1.md`) y **figuras** (`figs/fig1…fig6.png` con sus scripts) |
| `kit/` | `figstyle.py`: kit de figuras en estilo de revista (calibrado contra figuras de PRA/AQT de 2026) |
| `verif/` | Verificación independiente: `informe_WP2.md` (resonancia, flujo, cota, prueba con baño microscópico), `vf/informe_VF.md` (auditoría y revisión adversaria final) |
| `lit/` | `lit_bundles.md` y `refs_bundles.bib` (solo referencias verificadas) |
| `scoutA/B/C/` | Informes de los tres exploradores (molécula de QDs: NO-GO; bundles: GO condicionado; huecos de literatura) |

Reproducir (con `OMP_NUM_THREADS=1`): `python3 bundles/tests/run_tests.py`; datos con `bundles/src/run_*.py`; figuras con `bundles/figs/fig*.py`.
Los PDFs de terceros se eliminaron del repositorio (no se redistribuyen).
