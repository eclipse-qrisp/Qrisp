import numpy as np
from numpy.linalg import norm
import time
from qrisp import QuantumFloat, transpile
from qrisp.simulator import statevector_sim
#from qiskit import Aer, execute
from qrisp.interface import AQTBackend
from qiskit_aqt_provider import AQTProvider
provider = AQTProvider("ACCESS_TOKEN")
example_backend = AQTBackend(backend = provider.get_backend("offline_simulator_no_noise"))

##arithmetic -- works / min_reqs are dependend on precision of calcs

n = 4
qf1 = QuantumFloat(n)
qf2 = QuantumFloat(n)

qf1[:] = 10
qf2[:] = 8

qf_res = qf1*qf2
qf_res.get_measurement(backend = example_backend)
print(qf_res)
print(qf_res.qs.num_qubits())
print(sum([val for val in qf_res.qs.transpile().count_ops().values()]))


# %%
# Testing statevector simulator performance
qc = transpile(qf1.qs.compile())

start_time = time.time()
res = statevector_sim(qc)
print("Qrisp simulator time: ", time.time() - start_time)

# %%

""" # Reverse qubits because of differing statevector convention
qc.qubits.reverse()
qiskit_qc = qc.to_qiskit()

start_time = time.time()
# simulator = Aer.get_backend('qasm_simulator')
simulator = Aer.get_backend("statevector_simulator")
result = execute(qiskit_qc, simulator).result()
qiskit_res = result.get_statevector(qiskit_qc).data
print("Qiskit simulator time: ", time.time() - start_time)

# adjust global phase
res = res / sum(qiskit_res) * np.abs(np.sum(qiskit_res))

print("Results equal:", norm(res - qiskit_res) < 1e-4)
 """