# Ruta 3: ampliacion de la auditoria de novedad (segunda ronda)
Metodo: 7 busquedas web extendidas + 8 consultas a la API de arXiv (titulo/abstract) + lectura de abstracts por id. Solo [ABSTRACT] salvo donde se indica. NO es exhaustivo (sin Google Scholar ni acceso a bases con texto completo).
## Hallazgos nuevos (no estaban en informe_3.md)
| Trabajo | Que hace | Relevancia |
|---|---|---|
| 2509.24429 "Nonclassical phonon pair" (exp., optomecanica, silicio; viola Cauchy-Schwarz > 5 sigma, coherencia 132 ns) [ABSTRACT] | Par de fonones no clasicos por mezcla de cuatro ondas optomecanica | Ya existe un EXPERIMENTO de "par de fonones"; mecanismo distinto (laser, optomecanica). Hay que citarlo y diferenciar. |
| 2112.13342 "Dynamical emission of phonon pairs in optomechanical systems" (teoria, STIRAP) [ABSTRACT] | "Emisor de pares de fonones antiagrupado", extensible a n fonones | Reduce la novedad de "fuente de pares de fonones"; mecanismo distinto. |
| 1901.00414 "Two-Photon Resonance Fluorescence of a Ladder-Type Atomic System" (exp., transmon en guia, g-f a dos fotones, pico superagrupado) [ABSTRACT] | Emision de dos fotones del transmon por la transicion g-f a una guia, con bombeo; sugiere cavidad resonante para reforzarla | Antecedente experimental directo para fotones del mismo mecanismo (escalera del transmon -> par). |
| 1308.4094 (Pechal et al., fotones unicos con control por microondas) [ABSTRACT via busqueda] | Acoplo efectivo de 2o orden |f,0> -> |g,1> con emision de UN fotón | Mismo mecanismo en version de un cuanto; la version de pares es la extension natural. |
| 2504.08511 Mazhari & Rituraj, "Efficient Photon Pair Generation from an Emitter in a Cavity" (v5, jun 2026) [TEXTO: abstract + intro + seccion de parametros optimos] | Emisor con cavidad doblemente resonante (modos w0 y w0/2), bombeo incoherente; solucion analitica aproximada del Lindblad; eficiencia maxima ~35 %; optimo con kappa_2 ~ g_1 (tasa de salida ~ acoplo) | Ya existe un OPTIMO ANALITICO de eficiencia de dos cuantos con condicion de salida ~ acoplo. Modelo distinto (bombeo incoherente, dos modos) y eficiencia distinta, pero debilita la novedad de C2 (kappa_m* = sqrt2 G es del mismo tipo). |
| 2312.15753 (transmon + resonador, transiciones |g,2>-|f,1>) [ABSTRACT via busqueda] | observa una transicion asistida; dice que |g,0>->|f,0> a dos fotones no se observo | contexto |
| 2609.07454 "High-purity fluorescence photon bundle emission from two separate emitters" (2026) [ABSTRACT] | bundles de dos fotones desde dos emisores separados; espacio finito excluye excitaciones mayores | relacionado con la linea de dimeros/bundles; fotones |
| 2605.09914 "Two-Phonon Resonance Drives Multicomponent Mechanical Cat States" (2026) [ABSTRACT] | proceso fotón -> fotón + dos fonones por supermodos opticos | otro mecanismo de dos fonones (optomecanico), gatos |
## No encontrado (ausencia de hallazgo, no prueba)
- Emision de un par de fonones |f,0> -> |g,2> de un transmon a una guia/cristal fononico con formula de eficiencia cerrada.
- Medida de acoplo longitudinal g_z qubit-mecanico (solo longitudinal electromagnetico: 2407.02024, 2107.10269).
## Efecto sobre C1-C4
C1 (par de fonones por acoplo efectivo): parcial; ahora ademas hay pares de fonones optomecanicos (2509.24429, 2112.13342).
C2 (formulas cerradas): parcialmente conocida en espiritu (optimo de salida ~ acoplo en 2504.08511); lo propio es la forma concreta para el modelo qubit+modo+guia y la estadistica de cluster por renovacion.
C3 (instancia fononica desde la escalera): mecanismo conocido en fotones (1901.00414, Gao 2024) y en un cuanto (1308.4094); la instancia fononica con parametros de cristal fononico es la parte no encontrada.
C4 (comparacion con Bin): no encontrada en ninguna de las dos rondas; sigue siendo la parte mas defendible, pero es una comparacion, no un mecanismo.
