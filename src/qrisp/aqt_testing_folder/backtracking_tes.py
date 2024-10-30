
from qrisp.interface import AQTBackend
from qiskit_aqt_provider import AQTProvider
provider = AQTProvider("ACCESS_TOKEN")
example_backend = AQTBackend(backend = provider.get_backend("offline_simulator_no_noise"))


#works / min_reqs are dependend on on complexity and depth

from qrisp import *

@auto_uncompute    
def reject(tree):
    exclude_init = (tree.h < tree.max_depth -1 )
    alternation_condition = (tree.branch_qa[0] == tree.branch_qa[1])

    return exclude_init & alternation_condition

@auto_uncompute    
def accept(tree):
    height_condition = (tree.h == tree.max_depth - 3)
    path_condition = (tree.branch_qa[0] == 0)
    path_condition = path_condition & (tree.branch_qa[1] == 0)
    path_condition = path_condition & (tree.branch_qa[2] == 1)

    return height_condition & path_condition


from qrisp.algorithms.quantum_backtracking import QuantumBacktrackingTree

tree = QuantumBacktrackingTree(max_depth = 3,
                               branch_qv = QuantumFloat(1),
                               accept = accept,
                               reject = reject)

tree.init_node([])


print(tree.statevector())
import matplotlib.pyplot as plt
tree.visualize_statevector()
plt.show()


qpe_res = tree.estimate_phase(precision = 1)


mes_res = qpe_res.get_measurement(backend = example_backend )
print(mes_res[0])
print(qpe_res.qs.num_qubits())
print(sum([val for val in qpe_res.qs.transpile().count_ops().values()]))

""" def reject(tree):
    return QuantumBool()

tree = QuantumBacktrackingTree(max_depth = 3,
                               branch_qv = QuantumFloat(1),
                               accept = accept,
                               reject = reject)

tree.init_node([])

qpe_res = tree.estimate_phase(precision = 4) """


