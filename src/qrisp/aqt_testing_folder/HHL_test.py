from qrisp.interface import AQTBackend
from qiskit_aqt_provider import AQTProvider

provider = AQTProvider("ACCESS_TOKEN")
example_backend = AQTBackend(backend = provider.get_backend("offline_simulator_no_noise"))

import numpy as np
from numpy.linalg import solve
from scipy.linalg import expm
from qrisp import QuantumVariable, HHL, control, multi_measurement, ry, x


# Example based on https://arxiv.org/pdf/2108.09004.pdf
def hamiltonian_ev(qv):
    matrix = [[1, -1 / 3], [-1 / 3, 1]]

    t = 3 / 4 * np.pi

    matrix = expm(1j * t * (np.array(matrix)))

    # gate = UnitaryGate(matrix)
    # qc = gate.definition
    # qrisp_qc = QuantumCircuit.from_qiskit(qc)
    # qv.qs.append(qrisp_qc.to_gate("hamiltonian_ev"), qv)

    qv.qs.unitary(matrix, qv)


def ev_inversion(ev, ancilla):
    # Rotate the ancilla qubit, performed through controlled rotation on the
    # ancilla qubit
    ry_values = [np.pi, np.pi / 3]

    for i in range(ev.size):
        with control(ev[i]):
            ry(ry_values[i], ancilla)


qv = QuantumVariable(1)
x(qv)

ancilla = HHL(qv, hamiltonian_ev, ev_inversion, precision=2)

mes_res = multi_measurement([ancilla, qv], backend = example_backend)
print(qv.qs.num_qubits())
print(sum([val for val in qv.qs.transpile().count_ops().values()]))

print("Measurement result: ", mes_res)

print("Intensity ratio: ", mes_res[(True, "1")] / mes_res[(True, "0")])


matrix = [[1, -1 / 3], [-1 / 3, 1]]

matrix = np.array(matrix)

b = [0, 1]

b = np.array(b)

solution = solve(matrix, b)
print("Classical solution intensity ratio: ", solution[1] ** 2 / solution[0] ** 2)
