"""Convert single-qubit gates to PRX (Phased-RX) decomposition."""

from __future__ import annotations

import numpy as np

from qrisp.circuit.operation import ClControlledOperation, U3Gate
from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.standard_operations import GPhaseGate


class PRXGate(U3Gate):
    r"""PRX (Phased-RX) gate.

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
        """Returns the inverse of the PRX gate.

        The inverse of :math:`R_Z(\\beta) R_X(\\alpha) R_Z(-\\beta)` is
        :math:`R_Z(\\beta) R_X(-\\alpha) R_Z(-\\beta)`.

        Returns
        -------
        PRXGate
            The inverted PRX gate with parameters ``(-alpha, beta)``.

        """
        return PRXGate(-self.alpha, self.beta)


def _get_phase_diff(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Return the global phase :math:`\\gamma` such that
    :math:`U_a = e^{i\\gamma} \\, U_b`.

    Computed as :math:`\\gamma = \\arg(\\operatorname{tr}(U_a U_b^\\dagger) / 2)`.
    """
    return float(np.angle(np.trace(U_a @ U_b.conj().T) / 2))


@CircuitPass
def convert_to_prx(qc: QuantumCircuit) -> QuantumCircuit:
    """Convert single-qubit gates to PRX (Phased-RX) gate decomposition.

    This pass converts arbitrary single-qubit gates to PRX gates.
    When a U3 gate is already in PRX form (:math:`\\lambda \\approx -\\phi`),
    it is replaced by a single :class:`PRXGate`. Otherwise it is decomposed
    into a sequence of two :class:`PRXGate` operations.

    Global phases introduced by the decompositions are accumulated and
    emitted as a :class:`~qrisp.circuit.GPhaseGate` on the zeroth qubit
    of the output circuit at the end of the pass.

    Parameters
    ----------
    qc : QuantumCircuit
        The input quantum circuit.

    Returns
    -------
    QuantumCircuit
        A new circuit with single-qubit gates decomposed into PRX gates
        and an optional trailing global-phase gate.

    Example
    -------
    >>> from qrisp import PassManager, convert_to_prx
    >>> pm = PassManager()
    >>> pm.add_pass(convert_to_prx)
    >>> transpiled_qc = pm.run(qc)

    """
    qc_new = qc.clearcopy()
    accumulated_phase = 0.0

    for i in range(len(qc.data)):
        op = qc.data[i].op

        if isinstance(op, ClControlledOperation):
            conversion_op = op.base_op
        else:
            conversion_op = op

        if isinstance(conversion_op, U3Gate):
            U_orig = conversion_op.get_unitary()

            # Single PRX case: lambda ≈ -phi (U3 is already in PRX form)
            if abs(conversion_op.lam + conversion_op.phi) < 1e-5:
                prx_0 = PRXGate(conversion_op.theta, conversion_op.phi + np.pi / 2)

                # Track global phase
                accumulated_phase += _get_phase_diff(U_orig, prx_0.get_unitary())

                # Append gate if not identity
                if abs(conversion_op.theta % (2 * np.pi)) >= 1e-5:
                    if isinstance(op, ClControlledOperation):
                        qc_new.append(prx_0.c_if(op.num_control, op.ctrl_state), qc.data[i].qubits)  # type: ignore[arg-type]
                    else:
                        qc_new.append(prx_0, qc.data[i].qubits)

            # Two PRX case: decompose arbitrary U3 into two PRX gates
            else:
                prx_0 = PRXGate(conversion_op.theta + np.pi, -conversion_op.lam + np.pi / 2)
                prx_1 = PRXGate(np.pi, (conversion_op.phi - conversion_op.lam) / 2 + np.pi / 2)

                # Combined unitary: PRX_1 · PRX_0  (prx_0 applied first in circuit)
                U_prx = prx_1.get_unitary() @ prx_0.get_unitary()
                accumulated_phase += _get_phase_diff(U_orig, U_prx)

                if not (abs(prx_0.alpha % (2 * np.pi)) < 1e-5):
                    if isinstance(op, ClControlledOperation):
                        qc_new.append(
                            prx_0.c_if(op.num_control, op.ctrl_state),  # type: ignore[arg-type]
                            qc.data[i].qubits,
                            qc.data[i].clbits,
                        )
                    else:
                        qc_new.append(prx_0, qc.data[i].qubits)

                if not (abs(prx_1.alpha % (2 * np.pi)) < 1e-5):
                    if isinstance(op, ClControlledOperation):
                        qc_new.append(
                            prx_1.c_if(op.num_control, op.ctrl_state),  # type: ignore[arg-type]
                            qc.data[i].qubits,
                            qc.data[i].clbits,
                        )
                    else:
                        qc_new.append(prx_1, qc.data[i].qubits)
        else:
            qc_new.append(qc.data[i])

    # Emit accumulated global phase on the zeroth qubit
    accumulated_phase = accumulated_phase % (2 * np.pi)
    if abs(accumulated_phase) > 1e-10 and len(qc_new.qubits) > 0:
        qc_new.append(GPhaseGate(accumulated_phase), [qc_new.qubits[0]])

    return qc_new
