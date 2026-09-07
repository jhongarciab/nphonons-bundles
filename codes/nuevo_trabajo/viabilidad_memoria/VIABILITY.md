# Viabilidad de una memoria fonónica asistida por estados oscuros

## Pregunta central

Determinar si dos emisores excitónicos acoplados pueden almacenar y recuperar un qubit fonónico con una ventaja cuantificable frente al almacenamiento directo en el modo mecánico.

El primer qubit de prueba es

\[
|\psi_b\rangle=\alpha|0\rangle_b+\beta|1\rangle_b.
\]

El estado oscuro \(|\Psi_-\rangle\) se considera un candidato a estado de almacenamiento, no una memoria garantizada. Para demostrar una memoria completa habrá que construir explícitamente los procesos de escritura y lectura.

## Hipótesis comprobables

1. La emisión colectiva puede proteger el estado oscuro, mientras que las pérdidas individuales limitan esa protección.
2. Una interacción de intercambio fonón–emisor y un control que mezcle los sectores brillante y oscuro permiten una transferencia reversible.
3. La arquitectura oscura supera al almacenamiento directo en fidelidad, tiempo de almacenamiento o eficiencia en una región robusta de parámetros.

## Modelo mínimo

- Dos puntos cuánticos y un modo fonónico.
- Acoplamiento de Förster \(J\).
- Acoplamiento Holstein \(\lambda N_e(b+b^\dagger)\).
- Pérdida fonónica \(\kappa\) y ocupación térmica \(n_\mathrm{th}\).
- Emisión radiativa colectiva \(\gamma_\mathrm{col}\).
- Emisión radiativa individual \(\gamma_\mathrm{ind}\).
- Desfasamiento individual y colectivo.
- Control diferencial y, como interfaz explícita, interacción de intercambio tipo sideband/Raman.

## Métricas

\[
F_\mathrm{write},\quad F_\mathrm{read},\quad F_\mathrm{process},\quad
\eta_\mathrm{memory},\quad T_1,\quad T_2,\quad P_\mathrm{leak}.
\]

El primer experimento numérico de esta carpeta no calcula todavía una memoria write–store–read. Calcula las líneas base de supervivencia para verificar si el estado oscuro ofrece alguna ventaja antes de introducir controles más complejos.

## Criterios de decisión

- **Continuar con memoria completa:** existe una mejora frente al almacenamiento directo y una transferencia write–read no trivial.
- **Reducir el alcance:** existe protección oscura, pero aún no una interfaz fonónica reversible; la tesis se enfoca en decoherencia colectiva.
- **Cambiar de arquitectura:** no existe ventaja frente a las líneas base bajo parámetros realistas.

## Literatura inicial

- Hall et al., *Controlling dephasing of coupled qubits via shared bath coherence*, 2025: https://journals.aps.org/prb/abstract/10.1103/ltk8-fpv3
- *A mechanical quantum memory for microwave photons*, Nature Physics, 2025: https://www.nature.com/articles/s41567-025-02975-w
- *Acoustic phonon phase gates with number-resolving phonon detection*, Nature Physics, 2025: https://www.nature.com/articles/s41567-025-03027-z
- *Millisecond coherence times in gigahertz-frequency mechanical oscillators*, Nature Physics, 2026: https://www.nature.com/articles/s41567-026-03314-3
