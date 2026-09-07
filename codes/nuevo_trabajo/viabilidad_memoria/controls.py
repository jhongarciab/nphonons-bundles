"""Controles candidatos; el intercambio se incluye explícitamente."""

import numpy as np


def differential_detuning(ops, amplitude):
    """Control que mezcla los sectores brillante y oscuro."""
    return amplitude * (ops["ne1"] - ops["ne2"])


def symmetric_sideband_exchange(ops, coupling):
    """Interfaz efectiva fonón-brillante tipo sideband/Raman.

    Este operador no se deduce automáticamente del acoplamiento Holstein;
    se incluye como hipótesis de control que debe justificarse físicamente.
    """
    bright_lowering = (ops["sm1"] + ops["sm2"]) / np.sqrt(2.0)
    return coupling * (
        ops["b"].dag() * bright_lowering
        + ops["b"] * bright_lowering.dag()
    )
