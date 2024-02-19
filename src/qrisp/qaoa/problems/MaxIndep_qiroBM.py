import qrisp

from qrisp.qiroStuff.qiroProblem import QIROProblem

from qrisp.qiroStuff.qiroMaxIndepSetInfrastr import *

from qrisp.qaoa.qaoa_problem import QAOAProblem
from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetCostOp, maxIndepSetclCostfct
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
from qrisp.qaoa.qaoa_benchmark_data import approximation_ratio
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import itertools

# qiskit mps simulator

optimal_sols = "10000000000000000000"
num_nodes = 20

recursive_array = []
normal_array = []


for index in range(30):
    giraf = create_rdm_graph(num_nodes,0.3, seed =  index*3)
    qarg = QuantumVariable(giraf.number_of_nodes())

    maxindep_instance = QAOAProblem(maxIndepSetCostOp(giraf), RX_mixer, maxIndepSetclCostfct(giraf))
    the_recursive = QIROProblem(giraf, maxindep_instance, 
                                     replacement_routine=create_maxIndep_replacement_routine, 
                                     qiro_cost_operator= create_maxIndep_cost_operator_reduced,
                                     qiro_mixer= create_maxIndep_mixer_reduced,
                                     qiro_init_function= init_function_reduced
                                     )

    res, solutions,  exclusions, corr_vals = the_recursive.run_new_idea(qarg=qarg, depth = 3, n_recursions = 1)
    testCostFun = maxIndepSetclCostfct(giraf)

    approx_rat = approximation_ratio(res, optimal_sols,testCostFun)
    print(approx_rat)
    recursive_array.append(approx_rat)
    qarg.delete()


for index2 in range(30):

    giraf = create_rdm_graph(num_nodes,0.3, seed =  index2*3)

    qarg = QuantumVariable(giraf.number_of_nodes())

    maxindep_instance = QAOAProblem(maxIndepSetCostOp(giraf), RX_mixer, maxIndepSetclCostfct(giraf))
    res = maxindep_instance.run(qarg=qarg, depth = 5)
    testCostFun = maxIndepSetclCostfct(giraf)

    approx_rat = approximation_ratio(res, optimal_sols,testCostFun)
    print(approx_rat)
    normal_array.append(approx_rat)
    qarg.delete()

print("recursive")
print(sum(recursive_array)/len(recursive_array))

print("normal")
print(sum(normal_array)/len(normal_array))