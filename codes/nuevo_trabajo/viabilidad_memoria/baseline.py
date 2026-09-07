#!/usr/bin/env python3
"""Compara supervivencia directa, brillante y oscura.

Uso:
    python codes/nuevo_trabajo/viabilidad_memoria/baseline.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from metrics import expectation_curve, pure_state_fidelity
from model import Parameters, build_system
from noise import collapse_operators


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "data"
FIG_DIR = ROOT / "results" / "oficial"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_case(ket, H, c_ops, tlist):
    rho0 = ket * ket.dag()
    result = qt.mesolve(H, rho0, tlist, c_ops, options={"store_states": True})
    return np.asarray(expectation_curve(result, ket))


def main():
    p = Parameters(Ncut=8, lam=0.22, J=0.5, Omega=0.0)
    ops = build_system(p)
    tlist = np.linspace(0.0, 300.0, 601)

    phonon_qubit = (qt.basis(p.Ncut, 0) + qt.basis(p.Ncut, 1)).unit()
    direct = qt.tensor(qt.tensor(qt.basis(2, 1), qt.basis(2, 1)), phonon_qubit)

    cases = {
        "direct": direct,
        "bright": ops["bright"],
        "dark": ops["dark"],
    }
    noise_models = {
        "independent": dict(gamma_ind=2e-4, gamma_col=0.0),
        "collective": dict(gamma_ind=0.0, gamma_col=2e-4),
    }

    curves = {}
    for noise_name, noise_kwargs in noise_models.items():
        c_ops = collapse_operators(
            ops,
            kappa=1e-3,
            gamma_phi=4e-4,
            n_th=0.0,
            **noise_kwargs,
        )
        for case_name, ket in cases.items():
            curves[f"{noise_name}_{case_name}"] = run_case(
                ket, ops["H"], c_ops, tlist
            )

    np.savez(DATA_DIR / "viabilidad_baseline_data.npz", t=tlist, **curves)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, noise_name in zip(axes, noise_models):
        for case_name, color in zip(
            cases, ("tab:blue", "tab:orange", "tab:green")
        ):
            ax.plot(
                tlist,
                curves[f"{noise_name}_{case_name}"],
                label=case_name,
                color=color,
            )
        ax.set_title(noise_name)
        ax.set_xlabel(r"$\omega_b t$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Fidelidad de supervivencia")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "viabilidad_baseline.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "viabilidad_baseline.png", dpi=180, bbox_inches="tight")
    print(f"Datos: {DATA_DIR / 'viabilidad_baseline_data.npz'}")
    print(f"Figura: {FIG_DIR / 'viabilidad_baseline.pdf'}")
    print("Fidelidad inicial oscura:", pure_state_fidelity(ops["dark"] * ops["dark"].dag(), ops["dark"]))


if __name__ == "__main__":
    main()
