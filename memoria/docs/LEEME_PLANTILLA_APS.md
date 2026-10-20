# Borrador APS (REVTeX 4.2)

- `tex/ingles/manuscrito_aps.tex`: versión en inglés, una columna (`preprint`), con la opción `pra`; para probar PRB, cambiar `pra` por `prb` en `\documentclass`. Las figuras se importan como PGF desde `../../figs/`. La bibliografía es `tex/referencias_intro.bib` (`\bibliography{../referencias_intro}`).
- `tex/espanol/manuscrito_aps_espanol.tex`: versión en español, doble columna, que comparte figuras y bibliografía; los rótulos internos de las figuras PGF siguen en inglés.
- Compilar desde `tex/ingles` o `tex/espanol`: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. Los `.bbl` están generados y versionados junto al `.tex`; sin `bibtex` previo la bibliografía solo carga si el `.bbl` está presente.
- Mapa de figuras, generadores y datos: `docs/FIGURAS.md`.
- Las simulaciones se corren en pcjhon; el Mac solo procesa los datos guardados y compila.
