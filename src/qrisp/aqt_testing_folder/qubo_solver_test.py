from qrisp.interface import AQTBackend
from qiskit_aqt_provider import AQTProvider
from qrisp.algorithms.qaoa import * 

provider = AQTProvider("ACCESS_TOKEN")
example_backend = AQTBackend(backend = provider.get_backend("offline_simulator_no_noise"))

# QUBO tutorial: quadratic knapsack - here one has to find maximum of QUBO_obj aka max y=x.T@Q@x
Q = np.array(
    [
        [1922,-476,-397,-235],
        [-476,1565,-299,-177],
        [-397,-299,1352,-148],
        [-235,-177,-148,874],

    ]
)

qarg = solve_QUBO(Q, depth = 1, backend = example_backend)

print(qarg.get_measurement(backend = example_backend))

print(qarg.qs.num_qubits())
print(sum([val for val in qarg.qs.transpile().count_ops().values()]))