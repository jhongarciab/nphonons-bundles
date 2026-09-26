# Verificación del error de signo del término de pares en `ma2.py`

Fecha: 2026-09-26. Alcance: el signo del término de intercambio de pares explícito (`pair=True`) que se corrigió durante la Tarea 45(d).

## 1. Qué cambió en `ma2.py`

- **Historial.** `ma2.py` tiene solo dos commits: `43ca141` (2026-09-25 23:08, versión de la Ronda 14 con el filtro) y `4eb9c27` (2026-09-26 14:12). No hay commits posteriores a `4eb9c27`.
- **El signo equivocado nunca se commiteó.** El término de pares se escribió primero con `+p['G']` en el árbol de trabajo; se corrigió con un `sed` a `-p['G']` antes de cualquier commit. Por eso no existe un "commit del arreglo": solo el código corregido llegó a `4eb9c27`.
- **Diff `43ca141 → 4eb9c27` (todo lo que cambió en `ma2.py`):**
  - `hamiltonian(p, N, filt=None)` pasó a `hamiltonian(p, N, filt=None, counter=True, gz_direct=True, pair=False)`.
  - Se añadió la única línea que toca el término de pares: `if pair: H0 = H0 - p['G'] * (sp * a * a + sm * a.dag() * a.dag())`.
  - Las seis líneas de acoplamientos se reescribieron como tres condicionales (`gx` rotante siempre; `gx` de 3ω si `counter`; `gz σ_z a` si `gz_direct`). Con `counter=True` y `gz_direct=True` dan exactamente las mismas seis líneas de antes.
  - `floquet(p, N, filt=None, atol=1e-13, rtol=1e-11)` pasó a aceptar `**hopts` y a reenviarlos a `hamiltonian`.
- **Confirmación:** el cambio del término de pares solo actúa con `pair=True`. Con `pair=False` (valor por defecto) el Hamiltoniano es idéntico al anterior.
- **Por qué el signo correcto es `-p['G']` (= +|G|).** Con `gx<0` y `gz>0` se tiene `G=2·gx·gz/w<0`. Un modelo estático (H = ±|G|(σ₊a²+h.c.) + Ω(σ₊+σ₋), κD[σ₋], N=22, g_x=−0.05, g_z=12κ) da: con `+|G|`, P_c(±2i)=1.0000 y ⟨a²⟩=−4.000; con `−|G|`, P_c(±2 real)=1.0000 y P_c(±2i)=0.0006. El gato de Ma es en ±2i, así que hace falta `+|G|`.
- **Síntoma del error.** El primer intento (con `+p['G']`, es decir G<0) dio P_c máx=0.0366 en las tres celdas con pares, el nivel del vacío, porque se formó un gato real en ±2 en vez de ±2i.

## 2. Quién usa `pair`

- **Scripts que importan `ma2`:** solo `tarea42_worker.py`, `tarea43_worker.py`, `tarea45d_worker.py` y `tarea45e_worker.py`.
- **`pair=True`:** solo `tarea45d_worker.py` (`opts = dict(counter='nocounter' not in var, gz_direct='nogz' not in var, pair='pair' in var)`), y solo en las variantes `nogz_pair` y `nocounter_nogz_pair`. Se lanzan desde `run_t45.sh` y `run_t45d2.sh`. No aparece en ningún otro script.
- Los otros tres llaman a `ma2.floquet` sin opciones: `tarea42_worker.py` (`ma2.floquet(p, N)`), `tarea43_worker.py` (`ma2.floquet(p, N, filt)`), `tarea45e_worker.py` (`ma2.floquet(p, N, atol=1e-12, rtol=1e-10)`), así que corren con `pair=False`.
- La variante `nocounter_retuned` (47(c)) y `nocounter`/`full` no contienen "pair" en el nombre, así que también corren con `pair=False`.
- **Corrección a la premisa de la pregunta:** las Tareas 39 y 40 no usan `ma2.py`. Usan `ma_model.py`, una implementación anterior e independiente (`tarea39_worker.py`, `tarea40_worker.py`, `tarea39_40_analisis.py`). Las Tareas 46 y 47(b) usan `tarea46.py` (Hamiltoniano propio, `tarea46_run.py`, `tarea47b_run.py`), la Tarea 47(b) Fig. 2 usa `tarea47b_fig2.py` (modelo efectivo propio) y la Tarea 44 usa `modelo_gautier.py`.

## 3. Recálculo con el `ma2.py` actual (una celda por tarea; mismas tolerancias que el original)

Las salidas se escribieron en un directorio temporal; no se tocaron las cachés del repositorio.

**Tarea 42, `a_gx0.05_gz5`** (`tarea42_worker.py 6 0.05 5 22`):

| Cantidad | Guardado | Recalculado | Dif. relativa |
|---|---|---|---|
| κ₁/κ₂ (tasa de ajuste / 8 / κ₂) | 2.810126e-3 | 2.810126e-3 | 0 |
| Tasa de paridad (ajuste) | 1.873417e-5 | 1.873417e-5 | 0 |
| Tasa de paridad (espectral) | 1.878532e-5 | 1.878532e-5 | 7.7e-10 |
| Brecha física (peso de borde ≤0.1) | 2.153352e-3 | 2.153352e-3 | 1.1e-12 |
| 5º modo Re (sin filtro) | 2.153352e-3 | 2.153352e-3 | 1.1e-12 |
| P_c máx | 0.9992951 | 0.9992951 | 0 |

**Tarea 43, κ_f=1, N=16** (`tarea43_worker.py 1 16`):

| Cantidad | Guardado | Recalculado | Dif. relativa |
|---|---|---|---|
| κ₁ espectral (rate_spec/4) | 2.594150e-7 | 2.594150e-7 | 7.7e-9 |
| κ₁ previsto (fórmula filtrada) | 2.618332e-7 | 2.618332e-7 | 0 |
| Brecha física (peso ≤0.1) | 3.105117e-3 | 3.105117e-3 | 0 |
| 5º modo Re (sin filtro) | 2.803388e-3 | 2.803388e-3 | 6.4e-13 |
| Tasa de paridad (temporal) | 8.349634e-7 | 8.349634e-7 | 1.0e-13 |
| P_c máx | 0.9988197 | 0.9988197 | 1.1e-16 |

**Tarea 40, w_p=11.98** (`tarea40_worker.py 11.98`, con `ma_model.py`):

| Cantidad | Guardado | Recalculado | Dif. relativa |
|---|---|---|---|
| P_c | 0.9975452 | 0.9975452 | 2.2e-16 |
| P_e | 2.648004e-3 | 2.648004e-3 | 2.2e-16 |
| ⟨a²⟩ | 3.973396 | 3.973396 | 2.2e-16 |
| paridad | 0.1982295 | 0.1982295 | 2.2e-16 |

**45(e), κ₂/κ=1, s=0** (`tarea45e_worker.py 1 0`, con `ma2.py`; equivale a w_p=11.98 con d=12): P_c, P_e, ⟨a²⟩ y paridad idénticos al guardado (dif. 0 a 5.6e-16). Esta celda coincide bit a bit con la de la Tarea 40: son dos implementaciones independientes (`ma_model.py` y `ma2.py`) y dan el mismo resultado.

**Límite:** se verificó una celda por tarea, no todas.

## 4. Conclusión

Ningún resultado distinto de 45(d) dependió del término de pares con el signo equivocado: solo `tarea45d_worker.py` activa `pair`, y las Tareas 39, 40, 42, 43, 45(b, c, e) y 47(c, d) corren con `pair=False` (ese código no se ejecuta), lo que confirman los recálculos, que reproducen lo guardado a precisión del integrador. En 45(d), las tres celdas con `pair=True` se repitieron con el signo corregido y solo esas cifras (las corregidas) están en las tablas.
