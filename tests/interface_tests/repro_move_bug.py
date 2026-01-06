from qrisp import QuantumCircuit
from qrisp.misc.stim_tools import StimNoiseGate
import stim

def test_repro():
    qc = QuantumCircuit(1)
    
    # 1 parameter errors
    op = StimNoiseGate("DEPOLARIZE1", 0.1)
    print(f"Created op: {op.name}, num_qubits: {op.num_qubits}")
    qc.append(op, [qc.qubits[0]])
    
    print("Instructions before transpile:")
    for instr in qc.data:
        print(f"  {instr.op.name} (type: {type(instr.op)})")

    # Check if transpile preserves it
    qc_transpiled = qc.transpile()
    print("Transpiled instructions:")
    for instr in qc_transpiled.data:
        print(f"  {instr.op.name} (type: {type(instr.op)})")
        
    try:
        from qrisp.interface.converter.stim_converter import qrisp_to_stim
        stim_circuit = qrisp_to_stim(qc)
        print("Stim circuit:")
        print(stim_circuit)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()