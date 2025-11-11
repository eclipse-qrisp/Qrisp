from cirq import Circuit, LineQubit
from qrisp.circuit import ControlledOperation, ClControlledOperation

from cirq import CNOT, H, X, Y, Z

qrisp_cirq_ops_dict = {
    'cx': CNOT,
    'h': H,
    'x': X,
    'y': Y,
    'z': Z,
}


def convert_to_cirq(qrisp_circuit):
    """Function to convert a Qrisp circuit to a Cirq circuit."""
    # get data from Qrisp circuit
    qrisp_circ_num_qubits = qrisp_circuit.num_qubits()
    qrisp_circ_ops_data = qrisp_circuit.data
    circ_ops_list = []
    
    # create an empty Cirq circuit
    cirq_circuit = Circuit()
    cirq_qubits = [LineQubit(i) for i in range(qrisp_circ_num_qubits)]

    # create a mapping of Qrisp qubits to Cirq qubits
    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = cirq_qubits[i]

    for instr in qrisp_circ_ops_data:
        
        # get the gate name, qubits it is acting on 
        # and parameters if there are any
        op_i = instr.op.name
        op_qubits_i = instr.qubits
        # if hasattr(instr.op, 'params'):
        #    params = instr.op.params
        
        if op_i not in qrisp_cirq_ops_dict:
            raise ValueError(f"{op_i} gate is not supported by the Qrisp to Cirq converter.")
        # elif isinstance(op_i, ClControlledOperation):
        #    raise ValueError(f"{op_i} gate is not supported by the Qrisp to Cirq converter.")
        
        cirq_op_qubits = [qubit_map[q] for q in op_qubits_i]

        cirq_gate = qrisp_cirq_ops_dict[op_i]
        
        # map a controlled op from Qrisp to Cirq 
        # if isinstance(instr.op, ControlledOperation):
            
        cirq_gate = qrisp_cirq_ops_dict[op_i]
        cirq_circuit.append(cirq_gate(*cirq_op_qubits))
    
    return cirq_circuit 
