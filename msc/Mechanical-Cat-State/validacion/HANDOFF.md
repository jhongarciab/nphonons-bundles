# Handoff — Validación numérica de Naseem, PRA 113, 013732 (2026), arXiv:2508.10500v2

Este documento resume todo el trabajo hecho hasta ahora en `./validacion/`
y deja lista la siguiente tarea (24-25) para que otro agente (con más
cómputo) la ejecute. Léelo completo antes de tocar código — hay bugs
metodológicos ya encontrados y corregidos que no hay que repetir, y un
patrón de infraestructura obligatorio para evitar que las corridas se
cuelguen.

## EMPEZAR AQUÍ (mismo repo, sin clonar)

Ya estás en el repo `trabajo` (remoto `jhongarciab/nphonons-bundles`),
en esta misma máquina/filesystem — no hay que clonar nada. Pasos:

1. `cd msc/Mechanical-Cat-State` y verifica si existe `.venv/`. Si no
   existe (está en `.gitignore`, no viaja con git), créalo:
   ```bash
   # Linux (sin python3.11 en el sistema): uv baja un Python 3.11 standalone
   pip install --user uv && export PATH=$HOME/.local/bin:$PATH
   uv python install 3.11
   uv venv --python 3.11 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   # (macOS: brew install python@3.11; python3.11 -m venv .venv; pip install -r requirements.txt)
   ```
   Chequeo: `.venv/bin/python -c "import qutip;print(qutip.__version__)"` -> 4.7.6
   (si sale 5.x, el venv esta mal: QuTiP5 rompe los scripts).
2. Verifica que corre: `cd validacion && python tarea24_verificacion.py`
   — debe imprimir `VERIFICACION: OK` con diferencia ~4e-10.
3. Ve directo a la **sección 3** de este documento ("Tarea 24 — ESTADO")
   para el estado exacto (celda por celda) de dónde quedó la ejecución
   y el comando exacto para retomarla. Ve a la **sección 4** para el
   script de la Tarea 25 (`tarea25_ab_worker.py`, ya escrito, sin
   correr) y qué falta ejecutar.
4. Usa **siempre** el patrón de proceso fresco por celda (worker +
   bucle de shell, ver sección 6) — correr todo en un solo proceso
   largo cuelga esta clase de barridos (Tarea 22).
5. Commit por hito (no todo junto al final) y push a `origin main`
   cuando termines cada pieza (rejillas completas, análisis de
   resonancia, tabla (a), tabla (b), resumen final) — así si se corta
   de nuevo, queda igual de fácil retomar.

## 0. Entorno

- Repo del paper original clonado en `msc/Mechanical-Cat-State/`
  (remoto propio del autor, `MuhammadTahirNaseem/Mechanical-Cat-State` —
  **no tocar ni pushear ahí**; su `.git` está movido a `.git.bak` para
  que el repo `trabajo` (el que sí es nuestro, remoto
  `jhongarciab/nphonons-bundles`) pueda trackear los archivos como
  contenido normal).
- Scripts originales del paper (sin modificar, nunca tocar):
  `fig1_git_v1.py`, `fig2_git_v1.py`, `fig3_git_v4.py`, `fig4_git_v1.py`,
  `fig5_git_v1.py`. Usan QuTiP 4 (`Options`, API vieja).
- Entorno: `msc/Mechanical-Cat-State/.venv/` — Python 3.11 (via
  `brew install python@3.11`) + `qutip==4.7.6` + `cython<3.0` (Cython≥3
  no es compatible con QuTiP4). Activar con:
  `source msc/Mechanical-Cat-State/.venv/bin/activate`
- Todos los scripts de validación viven en `msc/Mechanical-Cat-State/validacion/`,
  numerados `tareaN_*.py`, con su `tareaN_resultados.md` (hallazgos) y
  `tareaN_resultados.npz` (datos). NO se modifican los scripts
  originales del paper — todo el trabajo nuevo es código separado.

### Patrón de infraestructura obligatorio: un proceso por celda

**Lección crítica de las Tareas 22-23**: correr un barrido con muchas
llamadas a `qutip.propagator`/`.eigenstates()` en un único proceso
Python largo produce un throttling severo e impredecible en esta
máquina (duty cycle de CPU tan bajo como 3-15%, sin relación clara con
batería/memoria/térmico — causa no diagnosticada). La solución que
funciona: un script "worker" que recibe los parámetros de UNA sola celda
por línea de comandos, hace el cómputo, y guarda un `.npz` individual;
un bucle de shell (`for` sobre los parámetros) lo invoca una vez por
celda, cada vez en un **proceso Python fresco**. Ver
`tarea22_worker.py`/`tarea23_worker.py` como plantilla. **Usar este
patrón para cualquier barrido de la Tarea 24-25** (son barridos de 9×9
y de varios Γ₂/κ × |α|², exactamente el tipo de caso que colgó antes).

Al lanzar cómputo largo con la herramienta Bash: usar `timeout` grande
y dejar que la herramienta lo pase a background automáticamente al
superar su propio límite (~590s) — esto funciona bien. **Evitar** el
parámetro explícito `run_in_background: true`, que en pruebas produjo
throttling mucho peor que dejar que la herramienta lo backgroundee sola.

## 1. Resumen de hallazgos por ronda (Tareas 1-23)

Documentos completos: `RESUMEN_FINAL.md` (Tareas 1-5),
`RESUMEN_FINAL_RONDA2.md` (6-8), `RONDA3.md` (9-12), `RONDA4.md` (13-15),
`RONDA5.md` (16-17), `RONDA6.md` (18-19), `RONDA7.md` (20-21),
`RONDA8.md` (22-23). Aquí solo la síntesis que importa para seguir:

### El factor 2: g_eff = 2g (RESUELTO, alta confianza)

- **Tareas 1-5**: reproducción de fig1/fig2/fig4/fig5 del paper;
  validación de traza/hermiticidad/positividad; bug menor confirmado
  (Re_S1_plus con Δ1_minus en vez de Δ1_plus) pero físicamente
  inconsecuente.
- **Tareas 2-3 (ronda 1)**: intentos iniciales de resolver si g_eff=g o
  2g dieron resultados **no concluyentes** (confundidos por acoplamiento
  fuerte y por comparar con drive fijo sin escanear). **No reusar esas
  conclusiones.**
- **Tareas 7-8 (ronda 2)**: dos métodos independientes y limpios
  confirman **g_eff=2g**: (Tarea 7) medición directa de Rabi de dos
  fotones sin disipación, coincide con 2√2·(2g) dentro de <5%; (Tarea 8)
  álgebra exacta de la transformación tipo polarón (verificada con
  commutadores Y exponencial matricial completa) reproduce exactamente
  el coeficiente de la Ec. (9) del paper, −2g, no el +g de la Ec. (11).
  **Conclusión: la Ec. (11) del manuscrito tiene un error de factor −2.**

### Saturación del qubit y estado oscuro

- **Tarea 6a**: la fórmula de saturación de un qubit libre falla por 2
  órdenes de magnitud aquí — el qubit está fuertemente hibridizado con
  la mecánica (g_x=6κ, g_z=60κ ≫ κ).
- **Tarea 6b/9b**: para ε≳0.4κ, g_eff=2g reproduce n̄_ss del completo
  dentro de 1-6%. Para ε≲0.2κ hay un "piso" no resuelto en ronda 1
  (ver Tarea 16 para su origen correcto).

### El "piso" y el aliasing (RESUELTO)

- **Tarea 9**: muestreo estroboscópico (t=nT_m, T_m=2π/ω_m) reduce la
  deriva espuria dn/dt en ~100-800× — confirma que gran parte de la
  "deriva" de la ronda 1 era aliasing del desplazamiento polarónico.
  Pero un piso residual en ⟨n⟩ persiste.
- **Tarea 13**: se descubrió que **todas las corridas previas salvo la
  Tarea 10 iniciaban el qubit en el estado FÍSICAMENTE EXCITADO**
  (`basis(Na,0)`, mal etiquetado "ground" en los comentarios del código
  original — confirmado numéricamente que `sigmam()` mapea
  `basis(Na,0)`→`basis(Na,1)`, o sea basis(Na,0) es el excitado).
  **Desde la Tarea 13 en adelante, el qubit SIEMPRE se inicia en
  `basis(Na,1)` (base física real).**
- **Tarea 16**: el piso es físico (no numérico — insensible a
  atol/rtol), y requiere el acoplamiento cruzado g_z·g_x, pero...
- **Tarea 19 (vía Floquet)**: ...el modo espectral dominante de
  relajación real escala **exactamente** como g_x² + γ_mecánico
  (R²=1.0000, intercepto=γ_m casi exacto) — es el canal Γ1 estándar, NO
  un efecto nuevo de "patada polarónica" en g_z (esa hipótesis, probada
  en la Tarea 17, dio resultados no concluyentes/ruidosos y se descarta).

### Floquet exacto: el gran cambio de método (Tareas 18+)

- Desde la Tarea 18, se abandona medir "estado estacionario" con
  `mesolve` a tiempo finito (ambiguo, lento) y se usa
  `qutip.propagator(H, T_m, c_ops)` para obtener el propagador exacto de
  un período, diagonalizarlo, y clasificar sus autovalores/autovectores
  por overlap con operadores físicos (paridad P, número n, a, a², σ_z).
  Es rápido (~20s-5min según Nb) y elimina la ambigüedad de convergencia.
- **Hallazgo mayor (Tarea 18-ii)**: la paridad *verdadera* del
  estacionario en ε=1.44 (punto del paper) es **0.052**, no 0.87 como
  medía la Tarea 12 con `mesolve` a τ_max=60 — existe un modo de
  phase-flip ultra-lento (λ≈6.9e-4κ, τ≈1447κ⁻¹) que ningún τ_max previo
  alcanzaba a resolver. Todo lo llamado "estacionario" en rondas 3-5 es
  en realidad un plateau transitorio (que de hecho es mejor cat state
  que el verdadero final).
- **Jerarquía de escalas (Tarea 20)**: confinamiento (τ≈10κ⁻¹) ≪
  phase-flip (τ≈1447κ⁻¹) ≲ bit-flip/coherencia lógica (τ≈2074κ⁻¹, la
  más lenta).
- **Tarea 21 — hallazgo central para la Tarea 24-25**: la **brecha de
  confinamiento del modelo efectivo diverge catastróficamente** frente
  al completo al aumentar Γ₂/κ (ratio efectivo/completo: 1.3× en
  Γ₂/κ=0.03 → **53.5×** en Γ₂/κ=2.07, el punto del paper). El phase-flip
  real es sistemáticamente **~4×** la fórmula ingenua 2|α|²(Γ₋+Γ₊)
  (factor estable). *(Nota: la Tarea 21 tenía un bug — el efectivo
  omitía el amortiguamiento intrínseco γ; corregido desde la Tarea 22 en
  adelante, siempre incluir `sqrt((n_th+1)γ)a`, `sqrt(n_th γ)a†` en el
  efectivo.)*
- **Tarea 22**: confirmada supresión exponencial del bit-flip con α²
  (pendiente ln(γ_bf)≈−1.3 a −1.6, más débil que el −2 ingenuo). Sesgo
  η=γ_pf/γ_bf crece hasta >1500× con α². **Hallazgo de alerta**: γ_bf del
  modelo efectivo es extremadamente frágil frente al término δ₁/Δ₂₋
  (colapsa 3-5 órdenes de magnitud al "compensar" la desintonía),
  mientras el completo apenas cambia (razón 0.63-1.13) — el efectivo NO
  es confiable para γ_bf cuantitativo.
- **Tarea 23**: el **Γ₂/κ óptimo (máxima brecha de confinamiento
  completa) ≈ 0.126**, brecha≈0.161 — muy por debajo del punto del paper
  (Γ₂/κ≈2.07). La brecha completa NO es monótona (sube, baja, vuelve a
  subir en Γ₂/κ=3) — el ajuste de un solo pico A·x/(1+Bx²) da R²=0.46
  (mediocre, no captura la subida final).

## 2. ERROR ENCONTRADO EN LAS TAREAS 22-23 ("compensar") — descartar esos resultados del completo

**Diagnóstico**: en el caso "compensar" (retonar ω_q=ω_d=2(ω_m+δ₁) en
el modelo completo), el drive queda en 2(ω_m+δ₁) mientras el acoplamiento
mecánico (términos g_x, g_z) sigue oscilando a ω_m. Como δ₁≠0, estas dos
frecuencias **no son conmensurables** con el mismo período — es decir,
`propagator(H, T_m)` con `T_m=2π/ω_m` **NO calcula un propagador de
Floquet válido** para ese Hamiltoniano (H(t) no es periódico con período
T_m si contiene un término oscilando a una frecuencia que no es múltiplo
entero de 2π/T_m). **Todos los resultados "compensado" del modelo
COMPLETO en las Tareas 22-23 son inválidos y deben descartarse** (los
resultados del modelo EFECTIVO compensado, que es time-independent, no
tienen este problema — esos siguen siendo válidos). Concretamente,
descartar: la fila "compensado" de la Tabla de la Tarea 22 (comparación
bf con/sin compensación, columna "completo"), y cualquier resultado de
`tarea22_cache/*_comp1.npz` con `modelo=full`.

## 3-4. ESTADO ACTUAL: Tareas 24 y 25 COMPLETAS (ronda 9)

Ver `RESUMEN_FINAL_RONDA9.md` (resultados) y `tarea25_resultados.md` (tablas).
Hecho: 4 rejillas de la Tarea 24 completas; resonancia vestida en (δ_m,Δ_q)=(0.048,0.144)
(pico agudo, signo opuesto a (δ₁,0)); Tarea 25 (a)/(b) 20/20 celdas.
Pendiente / abierto: revisar `clasificar()` en `tarea25_ab_worker.py` (modo de confinamiento
espurio en 4 celdas del completo, ver caveat en el resumen); no-suspender el PC en corridas largas
(scripts `run_t24_*.sh`, `run_t25.sh`: xargs -P 4-6, un proceso por celda).

## 3b. Ronda 10 (Tareas 26-27) COMPLETA

Ver `RESUMEN_FINAL_RONDA10.md`. Sin piso numérico en γ_bf; brecha robusta: máximo en 0.13 y subida en 3.0 persisten en el completo; el efectivo (Δ_2−=Δ_q) es monótono.

## 3c. Ronda 11 (Tareas 31-34) COMPLETA; 29-bis INCOMPLETA

Ver `RESUMEN_FINAL_RONDA11.md`. Conclusión: el máximo en Γ₂/κ≈0.13 solo aparece con Floquet (dependencia temporal), no con ningún ingrediente estático.
29-bis (buffer 2 niveles vs armónico, y compuerta Z A/B/C) quedó a medias: `modelo_buffer.py`, `tarea29bis_*worker.py`, `run_t29bis.sh`, `run_gate.sh`; caches parciales. Problema conocido: para la brecha de B (buffer armónico, dim ~22500) ni shift-invert cerca de 0 ni ARPACK 'LR' sirven; el método que validó contra denso es la unión de shift-invert complejos σ=0.05+iω (ω=0..12) quitando los 4 modos lógicos (código sin integrar en `modelo_buffer.py` aún).

## 3d. Ronda 12 (Tareas 35-36) COMPLETA

Ver `RESUMEN_FINAL_RONDA12.md`. **El cuádruplete de Floquet (Im 7-37) es un artefacto de truncamiento (|Im|∝N); el máximo de la brecha del completo no es físico.** Pendiente: recalcular la brecha física excluyendo modos de borde de Fock / con N mayor; revisar Tareas 21, 23, 27, 28, 33. Umbral térmico (Tarea 36): tabla de n_q* y f_min en el resumen.

## 3e. Ronda 13 (Tareas 37-38) COMPLETA

Ver `RESUMEN_FINAL_RONDA13.md`. La brecha física del Floquet completo es monótona y satura en ~0.23 (sin máximo en 0.13): se retiran los resultados de máximo/óptimo de Tareas 21, 23, 27, 28. Umbral térmico con baños consistentes: η≥100 requiere hf/kT≳4.8.

## 3f. Ronda 14 (Tareas 39-44)

Ver `../validacion_ma/RESUMEN_FINAL_RONDA14.md`. Entorno QuTiP 5 aparte en `../validacion_ma/.venv` (`requirements.txt`). Tarea 43 (baño filtrado) al final del resumen.

## 3g. Ronda 15 (Tareas 45-46)

Ver `../validacion_ma/RESUMEN_FINAL_RONDA15.md`. **Importante:** la brecha de Floquet a Γ₂/κ≳0.25 quedó NO determinada (rama interior sin converger hasta N=38); rebaja lo dicho en las Rondas 13 y 14. Origen de la caída de P_c en Ma: el acoplamiento g_z σ_z a. Ancho de la resonancia ∝ κ₂^0.54.

## 5. Convenciones y valores de referencia a reutilizar

```
r = 0.1  (g_x/g_z en la definición base)
gz_org = 2π·6e6, omega_m_org = 2π·100e6, Gamma_m_org = 2π·15, kappa_org = 2π·100e3
gz_baseline = gz_org/kappa_org = 60.0   (unidades de κ)
om_m = omega_m_org/kappa_org = 1000.0
Gam_m = Gamma_m_org/kappa_org = 1.5e-4
gx_fijo = r*gz_baseline = 6.0   (fijo en todos los barridos de g_z de las Tareas 15/17/19/21/22/23)
g = gz*gx/om_m ;  g_eff = 2*g   (Tareas 7-8)
Gamma2 = 4*g_eff**2/kappa
delta_1 = gx**2 * (Im S_{1-} + Im S_{1+}),  con D1m=om_m, D1p=3*om_m
qubit SIEMPRE en basis(Na=2, 1)  (estado base fisico; basis(Na,0) es el EXCITADO)
T_m = 2*pi/om_m  (periodo de muestreo estroboscopico / Floquet base)
```

QuTiP4 API relevante: `qutip.propagator(H, T, c_ops, options=Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000))`
devuelve un superoperador; `.eigenstates()` da autovalores/autovectores;
`qutip.vector_to_operator(vec)` reconstruye el operador desde el
autovector; overlaps vía `abs((op.dag()*Xk).tr())/Xk.norm()`.

## 6. Archivos de referencia clave para la Tarea 24-25

- `tarea18_floquet.py` — implementación base del propagador de Floquet
  (sin marco rotante alternativo), usar como plantilla para la
  verificación de conmensurabilidad.
- `tarea20_modos_lentos.py` — clasificación de modos por overlap.
- `tarea21_espectro_completo_vs_efectivo.py` — comparación completo vs
  efectivo con δ_1.
- `tarea22_worker.py`, `tarea23_worker.py` — **plantilla obligatoria**
  del patrón "un proceso por celda".
