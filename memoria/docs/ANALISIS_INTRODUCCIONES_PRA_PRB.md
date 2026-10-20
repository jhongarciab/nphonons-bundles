# Análisis de introducciones en artículos afines (PRA/PRB) — 17 oct 2026

Fuente: texto completo de los preprints de arXiv (extraído con `pdftotext`), lectura de cada introducción. Palabras y referencias de la introducción se contaron en el texto extraído (rango de números `[a–b]` citados antes de la Sec. II); el número de párrafos se leyó a mano. Preprints, no la versión final de la revista: pueden diferir en detalles menores.

## Corpus
| Artículo | Revista | Palabras | Refs distintas | Párrafos | Cierre de la introducción |
|---|---|---:|---:|---:|---|
| Vargas-Calderón et al., *Dark exciton giant Rabi oscillations…* (2112.03679) | PRB 106, 035305 | ~730 | 45 | 5 | contribuciones + esquema + nota sobre parámetros |
| Cosacchi et al., *N-photon bundle statistics…* (2108.03967) | PRB 106, 115304 | ~710 | 32 | 5 | qué se hace en cada plataforma, resultados cualitativos |
| Hall et al., *Controlling dephasing of coupled qubits via shared-bath coherence* (2405.14685) | PRB 112, 045303 | ~760 | 23 | 6 | «In this paper, we demonstrate…» + resultado cualitativo |
| Ask et al., *Cavity-free vacuum-Rabi splitting in circuit quantum acoustodynamics* (1811.09421) | PRA 99, 013840 | ~920 | 22 | 5 | resultado con criterio cuantitativo (γT ≥ 1) + organización |
| Teh et al., *Creation, storage and retrieval of an optomechanical cat state* (1809.00608) | PRA 98, 063814 | ~1260 | 64 | 9 | resultado numérico (9 fonones, S=36) + esquema por secciones |
| Droenner et al., *Collective enhancements in many-emitter phonon lasing* (1706.07777) | PRA | ~430 | ~34 | 3 | esquema por secciones |
| Gorshkov et al., *Photon storage in Λ-type media I* (quant-ph/0612082) | PRA 76, 033804 | ~3700 | 53 | 9+ | artículo de serie (I–III); introducción atípica |
| Bin et al., *N-phonon bundle emission via the Stokes process* (1907.12714) | PRL 124, 053601 | ~770 | 41 | 4 | sin esquema (formato PRL) |
| *Filtering multiphoton emission…* (no PRA/PRB; referencia de estilo) | — | ~870 | 38 | 4 | esquema por secciones |

Mediana de los seis artículos regulares de PRA/PRB (sin Gorshkov): **~750 palabras, ~33 referencias, 5 párrafos**.
Nuestra introducción actual (`manuscrito_memoria.tex`): **~980 palabras, 40 referencias distintas, 8 párrafos**.

## Estructura recurrente (embudo)
1. **Contexto amplio**, 1 párrafo, referencias agrupadas en rangos (`[1–6]`, `[14–39]`): por qué importa el tema (tecnologías cuánticas, fonones, memoria).
2. **Plataforma concreta y sus ventajas** (fonones: sin pérdidas por radiación, escalas distintas de las ópticas, integración en chip).
3. **Trabajo más cercano, nombrado y descrito** en 3–4 frases («the recent work by Bin et al. [12] proposes…») y **qué no hace** (la brecha).
4. **«In this work…»**: qué se estudia y contribuciones concretas («the main contributions are: …»).
5. **Alcance y organización** («The paper is organized as follows…»), a veces con una nota de honestidad sobre parámetros (Vargas-Calderón).
La figura 1 aparece a veces dentro de la introducción (Cosacchi, Filtering). Ninguno usa subsecciones en la introducción.

## Cifras en la introducción
Solo Teh (memoria) y Ask dan un resultado cuantitativo en el último párrafo; el resto resume cualitativamente. PRL (Bin) y PRB (Vargas, Hall, Cosacchi) no ponen números de fidelidad en la introducción.

## Comparación con nuestra versión
**Bien:** nombra el trabajo más cercano (Bin 2020, Cosacchi 2022) y su límite; separa «oscuro por espín» de «oscuro colectivo»; el volumen de referencias (40) está dentro del rango.
**Para ajustar:**
- 3 párrafos antes de llegar a los puntos cuánticos (memoria → fonones → plataformas mecánicas); el corpus usa 1–2. Los párrafos 2 y 3 se pueden fundir (11 refs en uno solo es lo máximo del corpus fuera de Teh).
- Falta el párrafo de **organización** («The paper is organized as follows…»), presente en 6 de 7 artículos regulares.
- Las **contribuciones** no están enumeradas; el corpus las declara explícitamente.
- Ponemos **cifras de resultado** (F=0.618…) en la introducción; es poco habitual en PRA/PRB. Solución: una frase cualitativa («a probability of recovery of order 0.7 where the passive reference retains <10 %») o dejarlo para resultados y resumen.
- 8 párrafos frente a una mediana de 5: reducir a 5–6.
- Pasar al inglés, formato APS («Ref. [12]», «Bin et al. [12]», referencias en rangos).

## Esquema propuesto (6 párrafos, ~800 palabras, ~38 refs)
1. Memoria cuántica: función en comunicación y procesamiento; métricas (tiempo, eficiencia, fidelidad; fase en qubit de dos intervalos). [duan2001, fleischhauer2002, gorshkov2007, afzelius2009, hedges2010, ortu2022]
2. Fonones y plataformas mecánicas, fundidos: transporte entre componentes, control de fonones individuales, memorias mecánicas existentes; criterio común: comparar con almacenamiento pasivo incluyendo captura y lectura. [habraken2012…bozkurt2025]
3. Puntos cuánticos como interfaz luz–movimiento + trabajo más cercano (Bin 2020, Vargas-Calderón 2022, Cosacchi 2022): emiten fonones; no almacenan un qubit entrante. [kettler2021, spinnler2023/24, bin2020, cosacchi2022]
4. Moléculas de puntos y estados oscuros colectivos: Förster, acoplo no uniforme → elementos entre autoestados; protección por subespacios libres de decoherencia con ruido correlacionado; distinción con oscuros por espín. [ortner2003…, lidar1998, rozbicki2008, sirkina2025]
5. «In this work…»: protocolo (tres puntos, dos ranuras oscuras, Raman, puertos); comparación con memoria pasiva; contribuciones enumeradas; resultado cualitativo.
6. Alcance y organización; aclarar que es un resultado teórico condicional.

## Sección «Modelo» en los mismos artículos (17 oct 2026)
| Artículo | Palabras (Sec. II) | Ecuaciones | Estructura |
|---|---:|---:|---|
| Vargas-Calderón (PRB) | ~450 | 3 | «We consider… described by the Hamiltonian [12]»: H completo en Ec. (1), definición de estados y operadores, marco rotante, parámetros con su justificación experimental; figura de transiciones |
| Droenner (PRA) | ~600 | 3 | Sistema físico, H₀ + H_I, ecuación maestra de Lindblad con D[x]; figura del esquema |
| Cosacchi (PRB) | ~670 | 3 | Subsección «cQED model»: H en marco rotante, definición de operadores, disipadores de Lindblad, baño de fonones |
| Hall (PRB) | ~2250 | ~29 | H = H₀ + H_IB; subsecciones para el método (Trotter, expansión de clusters…) |
| Teh (PRA) | largo | muchas | subsecciones A–E: protocolo, Hamiltoniano, relaciones entrada-salida, función modal óptima |
| Filtering | ~3600 | ~51 | mecanismo analítico en subsecciones A–B |

Patrón: el Hamiltoniano completo aparece pronto como Ec. (1), con cada término definido justo después; el marco rotante y la ecuación maestra se declaran a continuación; se justifican los valores de parámetros en unidades de ω_b; una figura de esquema abre la sección. Las secciones cortas (Vargas, Droenner, Cosacchi) no usan subsecciones; las largas sí. Nuestra Sec. II (~800 palabras, 6 ecuaciones, 3 subsecciones: Hamiltoniano, estados colectivos, acoplo y subespacio de almacenamiento) queda entre las dos clases.

## Sección «Escritura y lectura Raman» (17 oct 2026)
Referencia de estilo: Gorshkov et al. (PRA 76, 033804) trata el esquema Raman con eliminación adiabática del estado excitado, discute el desplazamiento Stark de la resonancia de dos fotones y dedica una subsección a las condiciones de adiabaticidad; la recuperación se obtiene por inversión temporal del almacenamiento. Teh et al. presentan el protocolo (descarga, almacenamiento, recuperación) en subsecciones antes del Hamiltoniano. Nuestra Sec. III sigue ese patrón: protocolo en un párrafo, H_eff por eliminación del intermedio, resonancia con desplazamiento, eficiencia y condiciones, y tabla de comparación con pulsos numéricos; la lectura se describe como inversión del proceso, empezando por la última ranura.
