#!/usr/bin/env python3
"""
02_coherencia_polaron.py — Coherencia del estado oscuro en base de polarón
===========================================================================

Versión corregida de 01_coherencia_libre.py. El estado almacenado es
|Ψ_-⟩ ⊗ |ñ⟩  donde  |ñ⟩ = D(α)|n⟩  con α = λ/ω_b  (base de polarón).

Fundamento: del régimen II (ec. D_libre de la tesis), la transformación
unitaria D̃ = exp[λ/ω_b · n̂_e · (b†-b)] elimina exactamente el
acoplamiento e-ph. Los eigenstados del Hamiltoniano libre en ese marco
son los estados de Fock desplazados |ñ⟩ = D(λ/ω_b)|n⟩. Por tanto
C_polaron(t) = ⟨Ψ_-,ñ|ρ(t)|Ψ_-,ñ⟩ debe decaer en escala ~min(1/κ, 1/2γ_φ),
NO en la escala ~1/λ que se observó con el proyector en base de Fock.

Comparación directa:
  C_Fock(t)    = ⟨Ψ_-,n|ρ(t)|Ψ_-,n⟩      (base de Fock  — incorrecto)
  C_polaron(t) = ⟨Ψ_-,ñ|ρ(t)|Ψ_-,ñ⟩      (base de polarón — correcto)

Parámetros: idénticos a pureza_2qds.py (producción).
Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
from pathlib import Path
import qutip as qt
from scipy.optimize import curve_fit

# =============================================================================
# PARÁMETROS — idénticos a pureza_2qds.py
# =============================================================================
RERUN = False   # False: calcula y guarda; True: carga desde NPZ

omega_b   = 1.0
gamma     = 0.0002
gamma_phi = 0.0004
J         = 0.5
kappa     = 0.001
lam       = 0.22      # α = λ/ω_b = 0.22

n_list    = [2, 3, 4]
Ncut      = 16

# Tiempo de evolución: cubre varios 1/κ ~ 1000/ω_b y 1/(2γ_φ) ~ 1250/ω_b
t_max  = 8000.0 / omega_b
n_steps = 2000
tlist   = np.linspace(0.0, t_max, n_steps)

# =============================================================================
# CONSTRUCCIÓN DE OPERADORES — QD₁ ⊗ QD₂ ⊗ Fock(Ncut)
# =============================================================================
b   = qt.destroy(Ncut)
nb  = b.dag() * b
Ib  = qt.qeye(Ncut)
sm  = qt.sigmam()
Iq  = qt.qeye(2)

b_s  = qt.tensor(Iq, Iq, b)
nb_s = qt.tensor(Iq, Iq, nb)
sm1  = qt.tensor(sm, Iq, Ib)
sp1  = sm1.dag()
sm2  = qt.tensor(Iq, sm, Ib)
sp2  = sm2.dag()
pe1  = sp1 * sm1
pe2  = sp2 * sm2

# ─── Estados colectivos excitónicos ─────────────────────────────────────────
ket_cv     = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
ket_vc     = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
ket_dark   = (ket_cv - ket_vc) / np.sqrt(2.0)   # |Ψ_-⟩
ket_bright = (ket_cv + ket_vc) / np.sqrt(2.0)   # |Ψ_+⟩

proj_dark_exc   = ket_dark   * ket_dark.dag()
proj_bright_exc = ket_bright * ket_bright.dag()
proj_dark_sys   = qt.tensor(proj_dark_exc,   Ib)
proj_bright_sys = qt.tensor(proj_bright_exc, Ib)

# ─── Operador de desplazamiento fonónico D(α) con α = λ/ω_b ─────────────────
# Corresponde a D̃ del régimen II de la tesis (ec. D_eph).
# qt.displace(N, alpha) = exp(alpha*b† - alpha*b)
alpha  = lam / omega_b          # = 0.22
D_op   = qt.displace(Ncut, alpha)   # operador en Fock(Ncut)

# Verificación: D†D = I
DD = D_op.dag() * D_op
err_unitario = (DD - qt.qeye(Ncut)).norm()
assert err_unitario < 1e-10, f"D no es unitario: error = {err_unitario:.2e}"
print(f"Verificación D†D = I: error = {err_unitario:.2e}  ✓")

# ─── Hamiltoniano libre (Ω = 0, Δ = 0) ──────────────────────────────────────
# H_libre actúa en la base de laboratorio (Fock original), no en la de polarón.
# La transformación de polarón solo modifica el OBSERVABLE, no el Hamiltoniano.
ne_total  = pe1 + pe2
H_libre   = (omega_b * nb_s
             + lam * ne_total * (b_s + b_s.dag())
             + J * (sp1 * sm2 + sp2 * sm1))

# ─── Operadores de colapso ──────────────────────────────────────────────────
c_ops = [
    np.sqrt(kappa)     * b_s,
    np.sqrt(gamma)     * sm1,
    np.sqrt(gamma)     * sm2,
    np.sqrt(gamma_phi) * pe1,
    np.sqrt(gamma_phi) * pe2,
]

# =============================================================================
# AJUSTE EXPONENCIAL
# =============================================================================
def exp_offset(t, Pss, A, T):
    return Pss + A * np.exp(-t / T)


def fit_Tdec(tlist, ydata, label=""):
    """Ajuste Pss + A*exp(-t/T). Retorna T o nan."""
    try:
        # Estimación de T0 desde caída a 1/e del rango
        rango   = ydata[0] - ydata[-1]
        target  = ydata[-1] + rango * np.exp(-1.0)
        idx0    = np.argmin(np.abs(ydata - target))
        T0      = tlist[idx0] if idx0 > 0 else t_max / 4.0
        p0      = [ydata[-1], ydata[0] - ydata[-1], T0]
        popt, _ = curve_fit(exp_offset, tlist, ydata, p0=p0,
                            bounds=([0, 0, 10], [1.0, 2.0, t_max * 3]),
                            maxfev=20000)
        Pss, A, T = popt
        print(f"  {label:30s}  Pss={Pss:.4f}  A={A:.4f}  T={T:.1f}/ω_b")
        return T, Pss
    except Exception as e:
        print(f"  {label}: ajuste fallido — {e}")
        return np.nan, np.nan


# =============================================================================
# CÁLCULO O CARGA
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
CODES_DIR = BASE_DIR.parent.parent
RESULTS_DATA_DIR = CODES_DIR / "results" / "data"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
data_file = RESULTS_DATA_DIR / "02_coherencia_polaron_data.npz"

if not RERUN:
    resultados = {}

    for n in n_list:
        print(f"\n{'='*65}")
        print(f"n = {n} fonones")
        print(f"{'='*65}")

        # ── Estado inicial en base de polarón: |Ψ_-⟩ ⊗ D(α)|n⟩ ─────────────
        fock_n_lab    = qt.basis(Ncut, n)          # |n⟩ en base de Fock
        fock_n_polaron = D_op * fock_n_lab         # |ñ⟩ = D(α)|n⟩

        # Normalización (D es unitario, debe ser 1 exactamente)
        norma = fock_n_polaron.norm()
        assert abs(norma - 1.0) < 1e-10, f"|ñ⟩ no normalizado: {norma:.8f}"

        ket0 = qt.tensor(ket_dark, fock_n_polaron)
        rho0 = ket0 * ket0.dag()

        # Verificaciones
        assert abs(rho0.tr() - 1.0) < 1e-10, "Traza inicial ≠ 1"
        assert rho0.isherm, "ρ₀ no hermítica"

        p_dark_0 = qt.expect(proj_dark_sys, rho0)
        print(f"  P_-(t=0) = {p_dark_0:.6f}  (debe ser 1.0)")

        # ── Proyectores ──────────────────────────────────────────────────────
        # C_Fock:    |Ψ_-,n⟩⟨Ψ_-,n|  (base de Fock — referencia)
        # C_polaron: |Ψ_-,ñ⟩⟨Ψ_-,ñ|  (base de polarón — correcto)
        proj_Cfock    = qt.tensor(proj_dark_exc, qt.fock_dm(Ncut, n))
        proj_Cpolaron = qt.tensor(proj_dark_exc,
                                  fock_n_polaron * fock_n_polaron.dag())

        e_ops = [
            proj_dark_sys,    # P_-(t)
            proj_bright_sys,  # P_+(t)
            nb_s,             # ⟨n̂⟩(t)
            proj_Cfock,       # C_Fock(t)    — comparación
            proj_Cpolaron,    # C_polaron(t) — observable correcto
        ]

        resultado = qt.mesolve(
            H_libre, rho0, tlist, c_ops, e_ops,
            options=qt.Options(nsteps=10000, atol=1e-9, rtol=1e-7)
        )

        P_minus    = np.array(resultado.expect[0])
        P_plus     = np.array(resultado.expect[1])
        n_bar      = np.array(resultado.expect[2])
        C_fock     = np.array(resultado.expect[3])
        C_polaron  = np.array(resultado.expect[4])

        print(f"\n  Valores iniciales:")
        print(f"    C_Fock(0)    = {C_fock[0]:.6f}")
        print(f"    C_polaron(0) = {C_polaron[0]:.6f}  (debe ser 1.0)")
        print(f"    <n̂>(0)       = {n_bar[0]:.6f}")

        # Ajustes
        print(f"\n  Ajustes exponenciales:")
        T_P,  Pss_P  = fit_Tdec(tlist, P_minus,   label="P_-(t)")
        T_Cf, Pss_Cf = fit_Tdec(tlist, C_fock,    label="C_Fock(t)")
        T_Cp, Pss_Cp = fit_Tdec(tlist, C_polaron, label="C_polaron(t)")

        # Referencias analíticas
        T_ref_gphi = 1.0 / (2.0 * gamma_phi)
        T_ref_kap  = 1.0 / kappa
        print(f"\n  Referencias analíticas:")
        print(f"    1/(2γ_φ) = {T_ref_gphi:.1f}/ω_b")
        print(f"    1/κ      = {T_ref_kap:.1f}/ω_b")

        resultados[n] = {
            "P_minus":   P_minus,
            "P_plus":    P_plus,
            "n_bar":     n_bar,
            "C_fock":    C_fock,
            "C_polaron": C_polaron,
            "T_P":       T_P,
            "T_Cf":      T_Cf,
            "T_Cp":      T_Cp,
        }

    # Guardar
    save_dict = {"tlist": tlist, "n_list": np.array(n_list),
                 "alpha": np.array([alpha])}
    for n in n_list:
        for key in ["P_minus", "P_plus", "n_bar", "C_fock", "C_polaron"]:
            save_dict[f"{key}_{n}"] = resultados[n][key]
        for key in ["T_P", "T_Cf", "T_Cp"]:
            save_dict[f"{key}_{n}"] = np.array([resultados[n][key]])
    np.savez(data_file, **save_dict)
    print(f"\nDatos guardados en {data_file}")

else:
    data = np.load(data_file)
    tlist = data["tlist"]
    alpha = float(data["alpha"][0])
    resultados = {}
    for n in n_list:
        resultados[n] = {k: data[f"{k}_{n}"]
                         for k in ["P_minus", "P_plus", "n_bar",
                                   "C_fock", "C_polaron"]}
        for key in ["T_P", "T_Cf", "T_Cp"]:
            resultados[n][key] = float(data[f"{key}_{n}"][0])
    print(f"Datos cargados desde {data_file}")

# =============================================================================
# RESUMEN NUMÉRICO
# =============================================================================
print("\n" + "="*65)
print("RESUMEN — Tiempos de decaimiento")
print(f"  α = λ/ω_b = {alpha:.3f}")
print(f"{'n':>4}  {'T(P_-)':>10}  {'T(C_Fock)':>12}  "
      f"{'T(C_polaron)':>14}  {'1/(2γ_φ)':>10}  {'1/κ':>7}")
print("-"*65)
T_ref_gphi = 1.0 / (2.0 * gamma_phi)
T_ref_kap  = 1.0 / kappa
for n in n_list:
    r = resultados[n]
    print(f"{n:>4}  {r['T_P']:>10.1f}  {r['T_Cf']:>12.1f}  "
          f"{r['T_Cp']:>14.1f}  {T_ref_gphi:>10.1f}  {T_ref_kap:>7.1f}")
print("="*65)
print()
print("DIAGNÓSTICO:")
print("  Si T(C_polaron) >> T(C_Fock) -> base de polarón confirma")
print("  la plausibilidad de la memoria fonónica.")
print("  Si T(C_polaron) ~ min(1/(2γ_φ), 1/κ) -> resultado esperado.")
print()
print(f"Parámetros: κ={kappa}, γ={gamma}, γ_φ={gamma_phi}, "
      f"λ={lam}, J={J}, Ncut={Ncut}")
