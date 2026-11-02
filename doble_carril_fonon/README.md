# Doble carril fonónico con chequeo de borrado (estudio de factibilidad)

Scripts de factibilidad (QuTiP 5, unidades µs y rad/µs). Modelo dispersivo efectivo, **no** el JC completo.
- `src/erasure_sim.py`: modelo y protocolo (almacenamiento + chequeo JPNS + lectura del transmon).
- `src/scan_check.py`: calidad del chequeo aislado frente a Δ, σ y T1q.
- `src/full.py`: silicio; compara chequeo vs pasivo. `src/extra.py`, `src/extra_hbar.py`: sensibilidad a χ y contraste HBAR.
- `src/feas.py`, `src/tlf.py`: umbral de correlación de ruido y modelo de fluctuadores (Monte Carlo).
Parámetros: Hitchcock et al. arXiv:2502.16426 (silicio) y Franzen et al. arXiv:2610.18778 (HBAR/transmon). T1q, T2q del transmon: los del dispositivo ETH (28/15 µs) o supuestos (100–300 µs).

---
**Actualización (2026-11-02):** los números de esta carpeta (ρ = 0.65, χ en lugar de 2χ, convención antigua de Tφ) quedan **superados** por `estudio/` (ver `estudio/DOSSIER.md`). Mantener solo como histórico.
