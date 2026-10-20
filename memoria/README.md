# Memoria de fonones con estados oscuros de tres puntos cuánticos

Material del artículo *Memoria para un qubit fonónico basada en estados oscuros de tres puntos cuánticos acoplados* (REVTeX 4.2, PRA).
Esta carpeta contiene solo lo necesario para leer, regenerar y compilar el artículo actual. Lo demás se conserva en `archivo_no_usado/`.

## Estructura

| Carpeta | Contenido |
|---|---|
| `tex/` | `ingles/` (`manuscrito_aps.tex`, una columna) y `espanol/` (`manuscrito_aps_espanol.tex`, dos columnas), cada una con su PDF, `.bbl` y auxiliares; `referencias_intro.bib` común, la nota de teoría `memoria_teoria.tex` y las plantillas oficiales de APS (`plantilla_aps/`). |
| `figs/` | Figuras del artículo: `figM0`–`figM5` y `figM7`–`figM9` en PGF (fuente de LaTeX) y, cuando existen, PDF y PNG de revisión. |
| `src/` | Código Python y lanzadores `.sh`: simulaciones (para pcjhon), generadores de figuras (para el Mac) y estilo común `memstyle.py`. |
| `data/` | Datos guardados que leen los generadores de figuras (`.log`, `.json`, `.npz`); `memory_figs/` tiene los datos de las Figs. 2–6, `passive_sim/` la referencia pasiva simulada. |
| `docs/` | Notas y reportes de la línea de memoria vigentes; ver `docs/FIGURAS.md` para el mapa figura → script → datos. |
| `archivo_no_usado/` | Todo lo que no se usa en el artículo actual (no se borró nada); ver su `README.md` y `MANIFIESTO_MOVIMIENTOS.csv`. |

## Flujo de trabajo

1. **Simulaciones**: solo en `pcjhon`. Los lanzadores `src/launch_*.sh` escriben en `data/`. Sincronizar con la misma estructura (`src/`, `data/`).
2. **Figuras**: en el Mac, `cd src && python3 plot_memory_<nombre>.py`. Leen `data/` y escriben en `figs/`.
3. **Compilar**: `cd tex/ingles && pdflatex manuscrito_aps.tex && bibtex manuscrito_aps && pdflatex manuscrito_aps.tex && pdflatex manuscrito_aps.tex`; igual en `tex/espanol` con `manuscrito_aps_espanol`. Los `.aux`, `.log` y `.blg` se regeneran en cada carpeta. Las figuras se importan como `../../figs/...` y la bibliografía como `../referencias_intro`.

Dependencias de Python: `numpy`, `scipy`, `matplotlib`; las simulaciones además `qutip` (5.x).

## Antes de modificar la estructura

Los generadores usan rutas relativas (`src/` → `../data`, `../figs`) y las figuras del manuscrito se importan como `../../figs/...` y la bibliografía como `../referencias_intro`; si se mueve una carpeta, hay que ajustar esas rutas.
