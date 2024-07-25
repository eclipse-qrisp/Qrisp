.. _QAOAExample:

QAOA Implementation for various problem instances
=================================================

The following problem instances, gathered in the table below with their correspondinghave already been successfully implemented using the Qrisp framework:

.. list-table::
   :widths: 45 45 10
   :header-rows: 1

   * - PROBLEM INSTANCE
     - MIXER TYPE
     - IMPLEMENTED IN QRISP
   * - :ref:`MaxCut <QAOAMaxCut>`
     - X mixer
     -    ✅
   * - :ref:`Max-$\\ell$-SAT <maxsatQAOA>`
     - X mixer
     -    ✅
   * - :ref:`QUBO (NEW since 0.4!) <QUBOQAOA>`
     - X mixer
     -    ✅ 
   * - :ref:`MaxIndependentSet <QAOAMaxIndependentSet>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MaxClique <maxcliqueQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MaxSetPacking <maxSetPackQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MinSetCover <minsetcoverQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`Max-$\\kappa$-Colorable Subgraph <QAOAMkCS>`
     - XY mixer
     -    ✅ 

Our voyage into :ref:`MaxCut <MaxCutQAOA>` and :ref:`Max-$\\kappa$-Colorable Subgraph <MkCSQAOA>` problems is detailed in our :ref:`tutorial <tutorial>` section. We decided to build the QAOA module with a focus on modularity, ensuring it can adapt to various problem instances while maintaining independence from the choice of decoding. This design approach makes it straightfowrads to formulate and solve other problem instances taking the steps as we did in the tutorial. 

Our team’s experience validates the adage “practice makes perfect”, demonstrating that familiarity with the process leads to more efficient and effective solutions. 

Some of the problem instances mentioned in the table above are condensed and presented below.

.. _QAOAMaxCut:

MaxCut
------

The MaxCut problem instance and its QAOA implementation is heavily discussed in the :ref:`MaxCut tutorial <MaxCutQAOA>`. All the necessary ingredients and required steps to run QAOA are elaborate in easy to grasp manner.

Here we instead provide the condensed implementation of the algorithm for MaxCut with using all of the predefined functions for this specific instance.

Problem Definition
^^^^^^^^^^^^^^^^^^

The MaxCut problem is defined as follows: 

    Given a graph $G=(V,E)$, find a subset $S\subset V$ such that the number of edgest between $S$ and $V\text{\\} S$ is the largest.

First we import the necessary functions and packages, create a graph ``G`` we will be cutting, define a quantum argument ``qarg`` we'll be acting on, as well as specify the depth of our algorithm.
::
  from qrisp.qaoa import QAOAProblem, maxcut_obj,create_maxcut_cl_cost_function,create_maxcut_cost_operator, RX_mixer
  from qrisp import QuantumArray, QuantumVariable
  import networkx as nx
  from operator import itemgetter

  G = nx.Graph()
  G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

  qarg = QuantumArray(qtype = QuantumVariable(1), shape = len(G))

  depth = 5

QAOA instanciation
^^^^^^^^^^^^^^^^^^
Next we follow the recipe to run the algorithm with ``QAOAProblem``, feeding it the ``cost_operator``, a ``mixer`` and a ``cl_cost_function``.
::
  import time
  maxcut_instance = QAOAProblem(create_maxcut_cost_operator(G), RX_mixer, create_maxcut_cl_cost_function(G))
  
  start_time = time.time()
  res = maxcut_instance.run(qarg, depth, max_iter = 50)
  print(time.time()-start_time)

Result analysis
^^^^^^^^^^^^^^^
After running our QAOA on the MaxCut problem instance we can now obtain the QAOA solution and draw the graph with optimally colored nodes.
::
  best_cut, best_solution = min([(maxcut_obj(x,G),x) for x in res.keys()], key=itemgetter(0))
  print(f"Best string: {best_solution} with cut: {-best_cut}")

  res_str = list(res.keys())[0]
  print("QAOA solution: ", res_str)
  best_cut, best_solution = (maxcut_obj(res_str,G),res_str)

  colors = ['r' if best_solution[node] == '0' else 'b' for node in G]
  nx.draw(G,node_color = colors, pos=nx.bipartite_layout(G, [0,1,2]))

.. _QAOAMaxIndependentSet:

MaxIndependentSet
-----------------
In the following example we will demonstrate how to solve the *maxIndependentSet* problem instance with the :ref:`QAOA module <QAOA>`. 

The problem is structured as follows: 

    Given a graph $G=(V,E)$ maximize the size of a clique, i.e. a subset $V' \subset V$ of mutually non-adjacent vertices.


The problem shows structural similarities to the MaxClique problem instance and may be implemented in analogy.
We will not stick to mathematical assignment of variable names.

Imports:
::
  from qrisp.qaoa import QAOAProblem
  from qrisp.qaoa import maxIndepSetCostOp, maxIndepSetclCostfct,  init_state
  from qrisp.qaoa import RX_mixer
  from qrisp import QuantumVariable
  import networkx as nx
  import matplotlib.pyplot as plt 

Problem Definition
^^^^^^^^^^^^^^^^^^
We begin by specifiying the graph considered for the problem, using the ``erdos_renyi_graph``-function . 

Additionally, we define the ``QuantumVariable`` to operate on.
::
  giraf = nx.erdos_renyi_graph(9,0.2, seed = 127)
  nx.draw(giraf,with_labels = True) #draw graph
  plt.show() 
  qarg = QuantumVariable(giraf.number_of_nodes())

QAOA instanciation
^^^^^^^^^^^^^^^^^^
Next we instanciate the ``QAOAProblem``, handing over a ``cost_operator``, a ``mixer`` and a ``cl_cost_function``. We then set the the ``init_function`` and run the instance.

``cost_operator``-generator and ``cl_cost_function``-generator have to be called with the problem graph ``giraf``.

The problem operator is based on the pennylane unconstrained maxClique QAOA (TODO: link) implementation, which defines the operator as follows: 
$$H_C = 3 \\sum _{(i,j) \\in E(G)} Z_i Z_j - Z_i - Z_j + \\sum _{i \\in V(G)} Z_i$$


where $V(G)$ is is the set of vertices of the input graph $G$, $E(G)$ is the set of edges of $G$, and $Z_i$ is the Pauli-$Z$ operator applied to the $i$-th vertex.
 
The mixer operator is a basic :ref:`X mixer <RXmixer>` applied to all qubits.
::
  QAOAinstance = QAOAProblem(cost_operator = maxIndepSetCostOp(giraf), mixer = RX_mixer, cl_cost_function = maxIndepSetclCostfct(giraf))
  QAOAinstance.set_init_function(init_function = init_state)
  theNiceQAOA = QAOAinstance.run(qarg = qarg, depth = 5)

Result analysis
^^^^^^^^^^^^^^^

Define the classical cost_function for analysis of singular result ``QuantumStates``  ::

    
    import itertools
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



Print the 5 most likely solutions and the associated energy/cost value 
::
  print("5 most likely Solutions") 
  maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
  for name, age in theNiceQAOA.items():  
    if name in maxfive:
      print((name, age))
      print(aClcostFct(name, giraf))

Print the solution as given by ``networkx`` 
::  
  print("NX solution")
  print(nx.max_weight_clique(giraf, weight = None))

.. _QAOAMkCS:

Max-$\\kappa$-Colorable Subgraph
--------------------------------

The Max-$\kappa$-Colorable Subgraph problem instance and its QAOA implementation is heavily discussed in the :ref:`Max-$\\kappa$-Colorable Subgraph tutorial <MkCSQAOA>`. All the necessary ingredients and required steps to run QAOA are elaborate in easy to grasp manner.

Here we instead provide the condensed implementation of the algorithm for M$\kappa$CS with using all of the predefined functions for this specific instance. 

Problem Definition
^^^^^^^^^^^^^^^^^^

The Max-$\kappa$-Colorable Subgraph problem is defined as follows: 

    Given a graph $G$ and $\kappa$ colors, maximize the size (number of edges) of a properly colored subgraph.

Similarly to the example of MaxCut above, we import the necessary functions and packages, create a graph ``G`` we will be cutting, define the colors we want to use, define a quantum argument ``qarg`` we'll be acting on (we provide options for one-hot and binary encoding schemes), as well as specify the depth of our algorithm.
::
  from qrisp.qaoa import QAOAProblem, mkcs_obj, apply_phase_if_eq, create_coloring_operator, create_coloring_cl_cost_function, QuantumColor, XY_mixer, apply_XY_mixer, RX_mixer
  from qrisp import QuantumArray
  import random
  import networkx as nx
  from operator import itemgetter

  G = nx.Graph()
  G.add_edges_from([[0,1],[0,4],[1,2],[1,3],[1,4],[2,3],[3,4]])
  num_nodes = len(G.nodes)

  color_list = ["red", "blue", "yellow", "green"]

  qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = True), shape = num_nodes) 
  #qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = False), shape = num_nodes) # use one_hot_enc = False if you use binary encoding

  depth = 3

QAOA instanciation
^^^^^^^^^^^^^^^^^^
Next we follow the recipe to run the algorithm with ``QAOAProblem``, feeding it the ``cost_operator``, a ``mixer`` and a ``cl_cost_function``. In case one prefers to use the binary encoding, adjust the `#` in the code block below.
::
  coloring_instance = QAOAProblem(create_coloring_operator(G), apply_XY_mixer, create_coloring_cl_cost_function(G))
  # coloring_instance = QAOAProblem(create_coloring_operator(G), RX_mixer, create_coloring_cl_cost_function(G)) # use RX mixer if you use binary encoding

  init_state = [random.choice(color_list) for _ in range(len(G))]
  coloring_instance.set_init_function(lambda x : x.encode(init_state))

  res = coloring_instance.run(qarg, depth, max_iter = 25)

Result analysis
^^^^^^^^^^^^^^^
After running our QAOA on the M$\kappa$CS problem instance we can now obtain the QAOA solution and draw the graph with optimally colored nodes.
::
  best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in res.keys()], key=itemgetter(0))
  print(f"Best string: {best_solution} with coloring: {-best_coloring}")

  best_coloring, res_str = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in list(res.keys())[:5]], key=itemgetter(0))
  print("QAOA solution: ", res_str)
  best_coloring, best_solution = (mkcs_obj(res_str,G),res_str)

  nx.draw(G, node_color=res_str, with_labels=True)
