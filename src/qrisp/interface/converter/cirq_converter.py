from cirq import Circuit, LineQubit
from qrisp.circuit import ControlledOperation, ClControlledOperation

from cirq import CNOT, H, X, Y, Z, CZ, S, T, R, SWAP, rx, ry, rz, inverse, I, M, R, CZ, CCNOT

qrisp_cirq_ops_dict = {
    # all multi-qubit gates
    'cx': CNOT,
    'cz': CZ,
    'swap': SWAP,
    # toffoli gate
    '2cx': CCNOT,
    # all single qubit gates
    'h': H,
    'x': X,
    'y': Y,
    'z': Z,
    'rx': rx,
    'ry': ry,
    'rz': rz,
    's': S,
    't': T,
    's_dg': inverse(S),
    't_dg': inverse(T),
    'measure': M,
    'reset': R,
    'id': I,
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
        
        if hasattr(instr.op, 'params'):
            params = instr.op.params
        
        if op_i not in qrisp_cirq_ops_dict:
            raise ValueError(f"{op_i} gate is not supported by the Qrisp to Cirq converter.")
        # elif isinstance(op_i, ClControlledOperation):
        #    raise ValueError(f"{op_i} gate is not supported by the Qrisp to Cirq converter.")
        
        cirq_op_qubits = [qubit_map[q] for q in op_qubits_i]

        cirq_gate = qrisp_cirq_ops_dict[op_i]
        
        # map a controlled op from Qrisp to Cirq 
        # if isinstance(instr.op, ControlledOperation):
            
        cirq_gate = qrisp_cirq_ops_dict[op_i]

        if instr.op.params and not isinstance(instr.op, ControlledOperation):
            # for single qubit parametrized gates
            if op_i == 'id':
                # added becasue the identity gate has parameters (0, 0, 0)
                cirq_circuit.append(cirq_gate(*cirq_op_qubits))
            else:  
                gate_instance = cirq_gate(*params)
                cirq_circuit.append(gate_instance(*cirq_op_qubits))
        
        elif isinstance(instr.op, ControlledOperation):
            # control and target qubits from qrisp
            control_qubits = instr.qubits[:len(instr.op.ctrl_state)]
            target_qubits = instr.qubits[len(instr.op.ctrl_state):]
            

            cirq_ctrl_qubits = [qubit_map[q] for q in control_qubits]
            cirq_target_qubits = [qubit_map[q] for q in target_qubits]
            
            # verify the contorl and target qubit mapping worked
            assert cirq_op_qubits == cirq_ctrl_qubits + cirq_target_qubits
            assert op_qubits_i == control_qubits + target_qubits

            # if the controlled operation is also parametrized
            if instr.op.params:
                gate_instance = cirq_gate(*params)
                cirq_circuit.append(gate_instance(*cirq_ctrl_qubits, *cirq_target_qubits))
            else:
                cirq_circuit.append(cirq_gate(*cirq_ctrl_qubits, *cirq_target_qubits))
        
        else:
            # for simple single qubit gates
            cirq_circuit.append(cirq_gate(*cirq_op_qubits))
    
    return cirq_circuit
