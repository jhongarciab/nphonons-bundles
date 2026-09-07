"""Canales de ruido para el modelo de viabilidad."""

import numpy as np


def collapse_operators(ops, *, kappa=1e-3, gamma_ind=2e-4,
                       gamma_col=0.0, gamma_phi=4e-4,
                       gamma_phi_col=0.0, n_th=0.0, phase=0.0):
    """Devuelve disipadores independientes, colectivos y térmicos."""
    c_ops = []

    b = ops["b"]
    if kappa > 0:
        c_ops.append(np.sqrt(kappa * (n_th + 1.0)) * b)
        if n_th > 0:
            c_ops.append(np.sqrt(kappa * n_th) * b.dag())

    if gamma_ind > 0:
        c_ops.extend([
            np.sqrt(gamma_ind) * ops["sm1"],
            np.sqrt(gamma_ind) * ops["sm2"],
        ])

    if gamma_col > 0:
        collective = ops["sm1"] + np.exp(1j * phase) * ops["sm2"]
        c_ops.append(np.sqrt(gamma_col) * collective)

    if gamma_phi > 0:
        c_ops.extend([
            np.sqrt(gamma_phi) * ops["ne1"],
            np.sqrt(gamma_phi) * ops["ne2"],
        ])

    if gamma_phi_col > 0:
        c_ops.append(np.sqrt(gamma_phi_col) * (ops["ne1"] + ops["ne2"]))

    return c_ops
