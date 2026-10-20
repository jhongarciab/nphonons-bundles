# Revisión bibliográfica para la memoria fonónica (16 octubre 2026)

## Dictamen

Hay una historia teórica publicable **si** se presenta como una memoria de un qubit fonónico itinerante de dos ranuras temporales, mapeado mediante Raman a dos excitones moleculares oscuros de tres puntos acoplados por Förster. La novedad está en la codificación en dos estados oscuros, la interferencia y la protección de la coherencia frente a ruido común, y el protocolo de entrada y salida analizado junto con el coste de los intervalos desprotegidos. No se puede reclamar la primera memoria de fonones, la primera memoria de estado oscuro, ni superioridad general frente a una cavidad fonónica reconfigurable.

La comparación cuantitativa actual es con **dos cavidades pasivas específicas** de pérdida intrínseca κ=0.001 ωb y las mismas eficiencias de carga/captura: a T=2000/ωb, F=0.607, s=0.715 y F/s=0.849 frente a F=0.087, s=0.090 y F/s=0.964. La mejora está en probabilidad de éxito y F sin condicionar, no en calidad del estado una vez recuperado. La dependencia de χ=1 y de los pulsos/puertos ideales limita el alcance del resultado.

## Antecedentes imprescindibles y uso exacto

| Referencia primaria | Qué establece | Cómo nos obliga a posicionarnos |
|---|---|---|
| [Bin et al., PRL 124, 053601 (2020)](https://doi.org/10.1103/PhysRevLett.124.053601) | Fuente teórica de paquetes de fonones | Contexto histórico del proyecto; no es comparador de memoria. |
| [Wang et al., Opt. Express 25, 30149 (2017)](https://doi.org/10.1364/OE.25.030149) | Propuesta de absorción, almacenamiento y reemisión de un fonón con conjunto NV y control adiabático | Prohíbe reivindicar primera memoria de fonón o primera transferencia a estado oscuro. Nuestra diferencia: molécula de puntos, dos ranuras excitónicas, qubit temporal y Förster. |
| [Taylor et al., npj Quantum Inf. 8, 19 (2022)](https://doi.org/10.1038/s41534-022-00526-2) | Memoria fonónica reconfigurable con cavidad de larga vida y acoplamiento dinámico; >90 % de fidelidad teórica de transferencia para pulso exponencial | Es el comparador más exigente. Nuestra ventaja sobre la referencia pasiva de κ=0.001 no implica superar su arquitectura. La conmutación de puertos tampoco es nueva. |
| [Zivari et al., Sci. Adv. 8, eadd2811 (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9674299/) | Experimento de fonones itinerantes con codificación time-bin | Motiva el tipo de qubit de entrada; no demuestra integración con nuestros puntos ni memoria. |
| [Cosacchi et al., Adv. Quantum Technol. 5, 2100131 (2022)](https://doi.org/10.1002/qute.202100131) | Memoria teórica de **fotón** en excitón oscuro de un punto, con escritura/lectura controlada y baño fonónico | Antecedente directo de almacenamiento óptico en excitón oscuro. Diferenciar claramente fonón entrante y dos estados oscuros moleculares. No confundir con `cosacchi2022` de paquetes. |
| [Kerfoot et al., Nat. Commun. 5, 3299 (2014)](https://www.nature.com/articles/ncomms4299) | Hibridación excitón-fonón e interferencia espectroscópica en molécula de dos puntos | Sustenta que existe la física de acoplo coherente; no demuestra memoria de tres puntos. |
| [Muljarov et al., PRL 95, 177405 (2005)](https://doi.org/10.1103/PhysRevLett.95.177405) | Interacción excitón-fonón diagonal y no diagonal en moléculas | Base del mecanismo transversal y del análisis de desfase. |
| [Sirkina et al., arXiv:2509.15418 (2025)](https://arxiv.org/abs/2509.15418) | Förster en baño acústico compartido tratado microscópicamente | Advierte que χ=1 y la relajación por regla de oro no constituyen validación microscópica. |
| [Bopp et al., PRB 107, 165426 (2023)](https://doi.org/10.1103/PhysRevB.107.165426) | Control coherente de excitones directos e indirectos en molécula de puntos | Apoya el control óptico/eléctrico del ingrediente molecular; no prueba los parámetros de tres puntos. |

## Novedad que sí se puede defender

La combinación concreta no apareció en esta búsqueda: un qubit fonónico time-bin itinerante escrito secuencialmente en **dos estados oscuros diferentes** de una molécula de **tres** puntos mediante una transición Raman asistida por el acoplo no uniforme al modo; almacenamiento sin resonancia residual y lectura temporal inversa. Este es un juicio de búsqueda, no una prueba de ausencia en toda la literatura. La justificación de tres puntos es operacional: con dos emisores hay solo un estado oscuro en el subespacio de una excitación, de modo que no se puede mantener el qubit de dos carriles como dos excitones con el mismo número de excitaciones. El mecanismo de Förster sigue presente como hopping entre los tres sitios y genera la base brillante/oscura y sus brechas.

## Reparos que un árbitro tendrá razón en plantear

1. **Referencia más fuerte:** comparar contra una cavidad reconfigurable de alta Q, motivada por Taylor, o acotar explícitamente el resultado a la cavidad pasiva de igual κ intrínseca. La actual razón de éxitos ≈8 no es una ventaja universal.
2. **Modelo conjunto:** la cifra F/s=0.849 proviene del modelo reducido con carga y lectura reales. En el modelo completo con biexcitón se ejecutó carga real y lectura ideal (F/s=0.920). No afirmar que los cuatro elementos ya se probaron simultáneamente.
3. **Protección:** la fidelidad ≈0.85 depende de χ=1, ruido blanco común y pausas cortas. Un cálculo del baño espacial o barrido χ y tiempos desprotegidos debe aparecer como condición central, no en nota marginal.
4. **Métrica:** separar probabilidad de recuperación s, fidelidad no condicionada F y F/s. Indicar que la fase de la coherencia se calibra al calcular F; verificar que el protocolo actúa sobre ambos estados lógicos y sus superposiciones, no solo una entrada favorable.
5. **Geometría:** el Hamiltoniano óptimo corresponde a acoplamientos de un triángulo isósceles; una pila lineal de tres puntos no materializa automáticamente esa matriz. Para un paper teórico abstracto se puede declarar la red de hopping como diseño, sin prometer la plataforma InAs/GaAs concreta.
6. **Entrada/salida:** la captura usa paquetes y conmutaciones ideales; pérdidas intrínsecas durante las ventanas, imperfección de forma y descarga completa aún requieren límites de sensibilidad.

## Cambios sugeridos al manuscrito

- Introducción: añadir Wang, Taylor y Cosacchi memoria **antes** de anunciar la contribución. Citar Zivari para justificar time-bins y Kerfoot/Muljarov/Bopp para los ingredientes físicos.
- Resumen: sustituir «en un modelo Lindblad de tres puntos con biexcitón, carga y lectura reales» por «en el modelo reducido con carga y lectura reales; por separado, el modelo completo confirma el efecto de biexcitón con carga real y lectura ideal».
- Figura de resultado: tres curvas de F, s y F/s contra T, junto con un segundo referente de cavidad reconfigurable o una declaración explícita de que la referencia pasiva tiene κ intrínseca fija.
- Titular propuesto: **Almacenamiento Raman de qubits fonónicos temporales en dos estados oscuros de una molécula de tres puntos cuánticos**.

## Conclusión editorial

El trabajo **sí tiene una pregunta y una respuesta científicas**: ¿pueden tres emisores transformar un qubit fonónico de dos tiempos en una coherencia entre dos estados moleculares oscuros y devolverlo después de que una cavidad dada ya haya perdido el fonón? En el modelo, sí, bajo ruido casi común y puertos rápidos. Esto justifica desarrollar un manuscrito teórico. El resultado todavía no sustenta una memoria superior a las mejores propuestas fonónicas ni una predicción experimental para InAs/GaAs. La prioridad previa a enviar es corregir el alcance del resumen, añadir los antecedentes directos y cerrar al menos un control de fidelidad del canal y otro de comparación con una memoria reconfigurable.
