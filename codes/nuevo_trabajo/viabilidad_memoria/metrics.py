"""Métricas básicas para la evaluación de viabilidad."""

import qutip as qt


def pure_state_fidelity(rho, ket):
    """Fidelidad de población con un estado objetivo puro."""
    return float(qt.expect(ket * ket.dag(), rho).real)


def expectation_curve(result, ket):
    """Población del estado puro objetivo en una evolución de QuTiP."""
    projector = ket * ket.dag()
    return [float(qt.expect(projector, state).real) for state in result.states]
