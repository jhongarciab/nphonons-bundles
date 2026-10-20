# Mapa de figuras del artículo

Numeración del manuscrito (la misma en inglés y en español). Los generadores están en `src/`, leen `data/` y escriben en `figs/`.

| Fig. | Archivo en `figs/` | Generador | Datos | Origen de los datos |
|---|---|---|---|---|
| 1 | `figM0_memory_scheme.pgf` | `plot_memory_scheme.py` | ninguno (esquema) | — (en la versión en español la Fig. 1 se dibuja con TikZ dentro del `.tex`) |
| 2 | `figM1_raman_response` | `plot_memory_raman_map.py` | `data/memory_figs/raman_write_map.npz` | `scan_memory_raman_map.py` (pcjhon) |
| 3 | `figM3_raman_dynamics` | `plot_memory_rabi.py` | mismo `raman_write_map.npz` | cortes del mapa |
| 4 | `figM5_dark_storage_decay` | `plot_dark_storage_decay.py` | `data/memory_figs/dark_storage_decay.npz` | `scan_dark_storage_decay.py` (pcjhon) |
| 5 | `figM4_protocol_dynamics` | `plot_memory_protocol_trace.py` | `data/memory_figs/protocol_trace.npz` | `trace_memory_protocol.py` (`launch_trace.sh`) |
| 6 | `figM2_operational_sweep` | `plot_memory_operational.py` | `data/memory_figs/operational_T*.json`; referencia pasiva `data/passive_sim/passive_T*.log` | `f17_real_load_reduced.py` (`launch_opsweep*.sh`); `passive_two_modes.py` |
| 7 | `figM7_ports_noise` | `plot_memory_ports.py` | `data/dn3_*`, `data/dn4_*`, `data/f18_lambda_S*.json` (y `data/f18_prev/`), `data/passive_sim/` | `launch_dense*.sh`, `launch_stark.sh`, `f18_lambda_scan.py` / `launch_f18tol.sh`, `launch_strain2.sh` |
| 8 | `figM9_requirements` | `plot_memory_requirements.py` | `data/dn4_chi*`, `data/stark_{A,B}_*`, `data/passive_sim/` | `launch_stark.sh` |
| 9 | `figM8_biexciton` | `plot_memory_biexciton.py` | `data/f19_full_dxxm*.log`, `data/f17tol9_ro_real10_kout10_T2000.log` | `f19_full_real_ports.py` (`launch_f19*.sh`, `launch_dense.sh`) |

Notas
- `memstyle.py` define el estilo, la función `curve` y `passive_ref` / `passive_sim` (referencia pasiva).
- La referencia pasiva de las Figs. 6–8 es la simulación de dos modos acústicos independientes con los mismos puertos (`passive_two_modes.py`, `tau12=100/ω_b`).
- Todas las simulaciones usan tolerancia relativa `1e-9` (`F9_TOL=1e-9`, `F14_TOL=1e-9`) y `N_cut=2`, salvo lo indicado en cada pie de figura.
