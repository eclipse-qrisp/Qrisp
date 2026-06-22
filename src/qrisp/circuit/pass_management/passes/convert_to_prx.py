"""
Convert single-qubit gates to PRX (Phased-RX) decomposition.
"""

from __future__ import annotations

import numpy as np
from qrisp.circuit.operation import U3Gate


class PRXGate(U3Gate):
    r"""
    PRX (Phased-RX) gate.

    The PRX gate is a single-qubit gate of the form:

    .. math::

        \text{PRX}(\alpha, \beta) = R_Z(\beta) \cdot R_X(\alpha) \cdot R_Z(-\beta)

    Parameters
    ----------
    alpha : float
        The rotation angle.
    beta : float
        The phase parameter.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

        # PRX(alpha, beta) = RZ(beta) * RX(alpha) * RZ(-beta)
        #
        # Using RX(alpha) = RZ(-pi/2) * RY(alpha) * RZ(pi/2) and the U3
        # decomposition U3(theta, phi, lam) = RZ(phi) * RY(theta) * RZ(lam):
        #
        #   PRX(alpha, beta) = RZ(beta) * RZ(-pi/2) * RY(alpha) * RZ(pi/2) * RZ(-beta)
        #                    = RZ(beta - pi/2) * RY(alpha) * RZ(pi/2 - beta)
        #                    = U3(alpha, beta - pi/2, pi/2 - beta)
        #
        super().__init__(alpha, beta - np.pi / 2, np.pi / 2 - beta, name="prx")

    def inverse(self):
        """
        Returns the inverse of the PRX gate.

        The inverse of :math:`R_Z(\\beta) R_X(\\alpha) R_Z(-\\beta)` is
        :math:`R_Z(\\beta) R_X(-\\alpha) R_Z(-\\beta)`.

        Returns
        -------
        PRXGate
            The inverted PRX gate with parameters ``(-alpha, beta)``.
        """
        return PRXGate(-self.alpha, self.beta)
