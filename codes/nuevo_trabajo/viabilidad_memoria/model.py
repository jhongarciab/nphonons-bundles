"""Modelo mínimo de dos emisores excitónicos y un modo fonónico."""

from dataclasses import dataclass

import numpy as np
import qutip as qt


@dataclass
class Parameters:
    omega_b: float = 1.0
    Delta: float = 0.0
    lam: float = 0.22
    J: float = 0.5
    Omega: float = 0.0
    Ncut: int = 8


def build_system(p: Parameters):
    """Construye operadores, estados colectivos y Hamiltoniano estático."""
    b = qt.destroy(p.Ncut)
    ib = qt.qeye(p.Ncut)
    iq = qt.qeye(2)

    ground_q = qt.basis(2, 1)
    excited_q = qt.basis(2, 0)
    lowering_q = ground_q * excited_q.dag()

    b_sys = qt.tensor(iq, iq, b)
    nb_sys = b_sys.dag() * b_sys
    sm1 = qt.tensor(lowering_q, iq, ib)
    sm2 = qt.tensor(iq, lowering_q, ib)
    sp1 = sm1.dag()
    sp2 = sm2.dag()
    ne1 = sp1 * sm1
    ne2 = sp2 * sm2
    ne = ne1 + ne2

    H = (
        p.omega_b * nb_sys
        + p.Delta * ne
        + p.lam * ne * (b_sys + b_sys.dag())
        + p.J * (sp1 * sm2 + sp2 * sm1)
        + p.Omega * (sp1 + sm1 + sp2 + sm2)
    )

    bright_q = (
        qt.tensor(excited_q, ground_q)
        + qt.tensor(ground_q, excited_q)
    ).unit()
    dark_q = (
        qt.tensor(excited_q, ground_q)
        - qt.tensor(ground_q, excited_q)
    ).unit()
    ground_qq = qt.tensor(ground_q, ground_q)
    vacuum_b = qt.basis(p.Ncut, 0)

    return {
        "H": H,
        "b": b_sys,
        "nb": nb_sys,
        "sm1": sm1,
        "sm2": sm2,
        "sp1": sp1,
        "sp2": sp2,
        "ne1": ne1,
        "ne2": ne2,
        "ne": ne,
        "bright": qt.tensor(bright_q, vacuum_b),
        "dark": qt.tensor(dark_q, vacuum_b),
        "ground": qt.tensor(ground_qq, vacuum_b),
        "vacuum": vacuum_b,
    }
