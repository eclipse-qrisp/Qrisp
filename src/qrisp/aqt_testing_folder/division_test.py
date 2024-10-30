import time
from qrisp.interface import AQTBackend
from qiskit_aqt_provider import AQTProvider

provider = AQTProvider("ACCESS_TOKEN")
example_backend = AQTBackend(backend = provider.get_backend("offline_simulator_no_noise"))

# works, but we are at min_reqs in this case 


from qrisp.alg_primitives.arithmetic import (
    QuantumFloat,
    q_div,
    q_divmod,
    qf_inversion,
)
from qrisp.misc import multi_measurement

numerator = QuantumFloat(2, signed=True)
divisor = QuantumFloat(2,  signed=True)

N = 3
D = 2
numerator.encode(N)
divisor.encode(D)

prec = 1
quotient, remainder = q_divmod(numerator, divisor, prec=prec, adder="thapliyal")

# Q, R = list(multi_measurement([quotient, remainder], backend = qasm_simulator))[0]
Q, R = list(multi_measurement([quotient, remainder],backend=example_backend))[0]
print(quotient.qs.num_qubits())
print(print(sum([val for val in remainder.qs.transpile().count_ops().values()])))

print("Q: ", Q)
print("delta_Q: ", abs(Q - N / D))
print("R: ", R)

# %%%

qf_0 = QuantumFloat(3, signed=False)
qf_1 = QuantumFloat(3, signed=False)

qf_0[:] = 4
qf_1[:] = 2

qf_2 = q_div(qf_0, qf_1, prec = 2)
print(qf_2.get_measurement(backend=example_backend))
print(qf_2.qs.num_qubits())
print(sum([val for val in qf_2.qs.transpile().count_ops().values()]))

# %%
qf = QuantumFloat(2, signed=False)
qf.encode(2)


inverted_float = qf_inversion(qf, prec=2)

start_time = time.time()
print(inverted_float.get_measurement(backend=example_backend))
print(inverted_float.qs.num_qubits())
print(sum([val for val in inverted_float.qs.transpile().count_ops().values()]))
print(0.75**-1)
print(time.time() - start_time)
