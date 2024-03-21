
from qrisp.qaoa import QAOAProblem
from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct,maxCliqueCostOp,init_state
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
import networkx as nx
import matplotlib.pyplot as plt

giraf = create_rdm_graph(9,0.7, seed =  133)
#draw graph
""" nx.draw(giraf,with_labels = True)
plt.show()  """


qarg = QuantumVariable(giraf.number_of_nodes())
#Instanciate QAOA
#mixer gets Graph as argument
#cost function gets graph as argument 
QAOAinstance = QAOAProblem(cost_operator= maxCliqueCostOp(giraf), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(giraf),callback=True) 
QAOAinstance.set_init_function(init_function=init_state)
QAOAinstance.set_callback()
theNiceQAOA = QAOAinstance.run(qarg=qarg, depth= 2)

# print the cost values and optimization params for each optimizer iteration 
print(QAOAinstance.optimization_cost)
print(QAOAinstance.optimization_params)

QAOAinstance.visualize_cost()


""" import itertools
def aClcostFct(state, G ):
    # we assume solution is right
    temp = True
    energy = 0 
    #intlist = [int(s) for s in list(state)]
    intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
    # get all combinations of vertices in graph that are marked as |1> by the solution 
    #combinations = list(itertools.combinations(list(np.nonzero(intlist)[0]), 2))
    combinations = list(itertools.combinations(intlist, 2))
    # if any combination is found in the list of G.edges(), the solution is wrong, and energy == 0
    for combination in combinations:
        if combination in G.edges():
            temp = False
    # else we just add the number of marked as |1> nodes
    if temp: 
        energy = -len(intlist)
        #energy = -sum(intlist)
    return(energy)

    

#print the ideal solutions
print("5 most likely Solutions") 
maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
for name, age in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:
        print((name, age))
        print(aClcostFct(name, giraf))


print("NX solution")
print(nx.max_weight_clique(giraf, weight = None))

#draw graph
nx.draw(giraf,with_labels = True)
plt.show()  """