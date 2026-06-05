"""
Convert single-qubit gates to PRY (Phased-RY) decomposition.
"""

from __future__ import annotations

import numpy as np
from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.operation import U3Gate, ClControlledOperation
from qrisp.circuit.quantum_circuit import QuantumCircuit

class PRYGate(U3Gate):
    def __init__(self, theta: float, phi: float) -> None:
        super().__init__(theta, phi, -phi)

@CircuitPass
def convert_to_pry(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Convert single-qubit gates to PRY (Phased-RY) gate decomposition.
    
    This pass converts arbitrary single-qubit gates to PRY gates, which are
    a specific form of U3 gates used in some quantum hardware implementations.
    The IQM backend requires PRX gates, but the conversion from PRY to PRX is
    trivial (a fixed phase conjugation) and happens inside the IQM conversion
    function.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The input quantum circuit.
        
    Returns
    -------
    QuantumCircuit
        A new circuit with single-qubit gates decomposed into PRY gates.
        
    Example
    -------
    >>> from qrisp import PassManager, convert_to_pry
    >>> pm = PassManager()
    >>> pm.add_pass(convert_to_pry)
    >>> transpiled_qc = pm.run(qc)
    """
    qc_new = qc.clearcopy()
    
    for i in range(len(qc.data)):
        op = qc.data[i].op

        if isinstance(op, ClControlledOperation):
            conversion_op = op.base_op
        else:
            conversion_op = op
        
        if isinstance(conversion_op, U3Gate):
            
            if abs(conversion_op.lam + conversion_op.phi) < 1E-5:
                if abs(conversion_op.theta % (2 * np.pi)) < 1E-5:
                    continue
                pry_0 = PRYGate(conversion_op.theta, conversion_op.phi)
                
                if isinstance(op, ClControlledOperation):
                    qc_new.append(pry_0.c_if(op.num_control, op.ctrl_state), qc.data[i].qubits)  # type: ignore[arg-type]
                else:
                    qc_new.append(pry_0, qc.data[i].qubits)
                
            else:
                pry_0 = PRYGate(conversion_op.theta + np.pi, -conversion_op.lam)
                pry_1 = PRYGate(np.pi, (conversion_op.phi - conversion_op.lam) / 2)
                
                if not (abs(pry_0.theta % (2 * np.pi)) < 1E-5 and abs(pry_0.phi % (2 * np.pi)) < 1E-5):
                    if isinstance(op, ClControlledOperation):
                        qc_new.append(
                            pry_0.c_if(op.num_control, op.ctrl_state),  # type: ignore[arg-type]
                            qc.data[i].qubits,
                            qc.data[i].clbits
                        )
                    else:
                        qc_new.append(pry_0, qc.data[i].qubits)
                        
                if not (abs(pry_1.theta % (2 * np.pi)) < 1E-5 and abs(pry_1.phi % (2 * np.pi)) < 1E-5):
                    if isinstance(op, ClControlledOperation):
                        qc_new.append(
                            pry_1.c_if(op.num_control, op.ctrl_state),  # type: ignore[arg-type]
                            qc.data[i].qubits,
                            qc.data[i].clbits
                        )
                    else:
                        qc_new.append(pry_1, qc.data[i].qubits)
        else:
            qc_new.append(qc.data[i])
            
    return qc_new
