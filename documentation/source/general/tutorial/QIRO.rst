.. _qiro_tutorial:

Quantum-Informed Recursive Optimization
=======================================

This tutorial will give you an overview of problem specific implementations for solving optimization problems with the :ref:`Quantum Approximate Optimization Algorithm (QAOA) <QAOA>`.

The Quantum Alternating Operator Algorithm (QAOA) is used to solve combinatorical optimization instances of NP-hard instances. For further information see our :ref:`tutorial on QAOA <QAOA101>`! 

The paper `Quantum-Informed Recursive Optimization Algorithms (2023) <https://arxiv.org/abs/2308.13607>`_ by J. Finzgar et. al. establishes a blueprint for developing alogrithms, that update the problem instance recursively based on the correlations in results obtained by QAOA.

While QAOA cost functions are designed with specific problem instances in mind, this research has shown promising improvements to this to approach by further leveraging the problem structure.

We have implemented this approach for different problem instances, namely [REF]. The explanation below tackles the `Maximum Indepent Set (MIS) <https://en.wikipedia.org/wiki/Maximal_independent_set>`_ instance, in analogy to the original paper by Finzgar et. al.

Starting point of the algorithm
----------------

The algorithm evaluates the result of a :ref:`QAOA <QAOA>` optimization procedure to establish correlations in the solution space, then recursively updates the problem structure. We will further assume that you are already mostly familiar with QAOA.  

The MIS is defined via a graph with a set of vertices :math:`V` and a set of edges :math:`E` . To solve this optimization problem, one has to find the maximum number of vertices, where none of the vertices share a common edge, i.e. we want to find

.. math:: 
     \max \left( |V'| , V' \subset V \right) , \, \, \text{where} \sum_{ i, j \in |V'| } \mathbb{1}_{(i,j) \in E} = 0 

The QAOA cost operator has the form 

.. math::
    H_C = - \sum_{i \in V} ( \textbf{1} - Z_i ) + 3  \sum_{i,j \in E} ( \textbf{1} - Z_i )( \textbf{1} - Z_j ),

and the mixer operator is

.. math::
    H_M = - \sum_{i \in V} Z_i .

Executing the optimization loop of our QAOA implementation of depth :math:`p`  will result in the state 

.. math::
    \ket{\psi} =  e^{i \gamma^{*}_{p-1} H_C} e^{i \beta^{*}_{p-1} H_M} ... e^{i \gamma^{*}_0 H_C} e^{i \beta^{*}_0 H_M} ( \ket{0} + \ket{1} )^{\otimes n},

with :math:`n` being the number of vertices in the graph. The :math:`\{\gamma^{*}_{p-1}, \beta^{*}_{p-1}, ...  \gamma^{*}_0 , \beta^{*}_0 \}` represent the optimized QAOA angles.
This state :math:`\ket{\psi}` is the starting point of the **Quantum-Informed Recursive Optimization** considerations. 

Establishing correlations 
-------------------------

For any problem specific QIRO implementation the next step is to evaluate the expected correlations in the solution space, i.e. computing the values of the matrix **M**, where
:math:`\text{M}_{ii} = \bra{\psi} Z_i \ket{\psi}` and :math:`\text{M}_{ij} = \bra{\psi} Z_i Z_j \ket{\psi}`.

We then need to find the **maximum absolute value** of M.

Reducing the problem 
--------------------

Based on the **maximum absolute entry** of M and its sign, one of the following replacements is employed:

* If :math:`\text{M}_{ii} \geq 0` is the maximum absolute value, then the :math:`i`-th vertex is set to be in the independent set (IS). In turn, we can remove all vertices that share an edge with this vertex can be removed from the graph, since including them in the solution would violate the problem constraints

* If :math:`\text{M}_{ii} < 0` is the maximum absolute value we remove :math:`i`-th vertex from the graph

* If :math:`\text{M}_{ij} > 0,  (i, j) ∈ E` was selected, we remove both nodes from the graph with the argument, that, since both of them would be in the same state in the final solution, including both as part of the solution would violate the constraint, as they share an edge. In turn, they can be removed from the graph. 

* If :math:`\text{M}_{ij} < 0,  (i, j) ∈ E` was selected, we remove all nodes that share an edge with both vertices :math:`i` and :math:`j`. Since one of the vertices :math:`i` and :math:`j` will be part of the final solution (but not both), any vertex that is connected to both :math:`i` and :math:`j` is guaranteed to violate the problem constraints, and can be removed from the graph. In this case it may be possible, that no vertex is found to be as a canditate for removing. We will then simple chose second biggest absolute value of **M** for the replacement routine.

These operations are undertaken directly on the ``networkx`` graph that has been fed to instance of the ``QIROProblem`` class, see the code example [REF] below. 

We then hand over the reduced problem graph to a new ``QAOAProblem`` instance, optimize the parameter, and reduce the problem again with the same subroutine as above. 

The final solution
--------------------

The after a specific number of recursions the final solution is returned as the result of a ``QAOAProblem`` optimization routine, 
where we consider the excluded and included vertices from the above steps in the ``cost_operator``, ``mixer`` and ``init_function`` of the ``QAOAProblem``.

The final result is therefore a the classic ``dictionary`` return from the ``QAOAProblem`` class and poses an optimized solution to the initial full problem instance. 

Try it out yourself with the example code below!

QIRO implementation
===================

The QIRO class
--------------

Upon instanciation, the :ref:`QIROProblem` class requires five arguments: 

* The ``problem`` to be solved, which not necessarly a graph, since QIRO is also implemented for [REF] MaxSat.
* The ``replacement_routine``, which has the job of performing the aforementioned specific reductions to the ``problem`` object.
* The ``cost_operator``, ``mixer``, ``init_function`` and ``cl_cost_function`` in analogy to :ref:`QAOAProblem` instanciation. 

Why the ``cost_operator``, ``mixer``, and ``init_function`` undergo some slight adjustements, will be made clear in the code example below, aswell as the necessity 
for directly assigning a ``cost_operator``, a ``mixer``, and an ``init_function``.

To run the instance and solve the optimization problem we use the [REF] ``run_qiro`` function, which takes the following arguments:
qarg, depth, n_recursions,  mes_kwargs = {}, max_iter = 50

* The :ref:`QuantumVariable` ``qarg``, which is the quantum argument the algorithm is evaluated on, in analogy to the QAOA module
* The integer ``depth``, which is the [REF?] depth of QAOA optimization circuit.
* The integer ``n_recursions``, representing the number of QIRO update steps.
* The dictionary ``mes_kwargs = {}``, empty by default, to define further specifications of the measurements, see :ref:`get_measurement`.
* The integer ``max_iter = 50``, set to 50 by default, which defines the maximum number of the classical optimization loop with the ``COBYLA`` optimizer as part of the QAOA optimization routine



Maximum independent set example
===============================

We now investigate a code example for the Maximum independent set problem instance.

Preliminaries
-------------

Before we get to the superficial code let us first do some explaining of the relevant aspects, starting with the ``replacment_routine``.

All in all, the function remains straight forward. We employ a ``find_max`` subroutine to find the entry and the sign of the maximum correlation value, to then adjust the graph. 

:: 

    def create_maxIndep_replacement_routine( res, Graph, solutions= [], exclusions= []):

        # For multi qubit correlations
        orig_edges = [list(item) for item in Graph.edges()]

        # For single qubit correlations
        orig_nodes = list(Graph.nodes())
        
        # find the max_item
        max_item, sign = find_max(orig_nodes, orig_edges , res, solutions)

        # create a copy of the graph to prevent unwanted side effects
        newGraph = copy.deepcopy(Graph)

        # we just directly remove vertices from the graph, as suggested by the replacement rules 
        # if the item is an int, its a single node, else its an edge
        if isinstance(max_item, int):
            if sign > 0:
            # remove all adjacent nodes
                newGraph.remove_nodes_from(Graph.adj[max_item])
                solutions.append(max_item)
                exclusions.append(max_item)
            elif sign < 0:
                # remove the nodes
                newGraph.remove_node(max_item)
                exclusions.append(max_item)
        else:
            if sign > 0:
                # remove both nodes
                newGraph.remove_nodes_from(max_item)
                exclusions += max_item
            elif sign < 0:
                # remove all nodes connected to both nodes
                intersect = list(set( list(Graph.adj[max_item[0]].keys()) ) & set( list(Graph.adj[max_item[0]].keys()) ))
                newGraph.remove_nodes_from(intersect)
                exclusions += intersect 

        return newGraph, solutions, sign, exclusions

As you might gave noticed in the code above, we add the nodes that are included into (respective excluded from) the solution to a list ``solutions`` (``exclusions``). 
This allows us to directly recycle the code for the [REF the function?] ``cost_operator``, ``mixer`` and ``init_function`` of the original QAOA implementation with minor adjustments.

Since we have to consider nodes that are already asigned to be in the solution set, or exluded from the algorithm, we do not want to apply these functions to said nodes. 
We therefore include some simple lines of code into the functions, for example in the [REF] ``qiro_RXMixer``:

::

    def qiro_RXMixer(Graph, solutions = []):

        def RX_mixer(qv, beta):
            for i in Graph.nodes():
                #DONT mix solution states
                if not i in solutions:
                    rx(2 * beta, qv[i])
    return RX_mixer

With the preliminaries out of the way, let us jump right into the code example:


Code execution
--------------

We start off by importing all the relevant code and defining the graph of our problem, aswell as the :ref:`QuantumVariable` to run our instance on:

:: 

    # imports 
    from qrisp.qiro.qiro_problem import QIROProblem
    from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
    from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetclCostfct
    from qrisp.qiro.qiroproblems.qiroMaxIndepSetInfrastr import * 
    from qrisp.qiro.qiro_mixers import qiro_init_function, qiro_RXMixer
    from qrisp import QuantumVariable
    import networkx as nx


    Define a graph via the number of nodes, and the QuantumVariable arguments
    num_nodes = 13
    G = create_rdm_graph(num_nodes, 0.4, seed =  107)
    qarg = QuantumVariable(G.number_of_nodes())



With this, we can directly throw everything thats relevant at the [REF] ``QIROProblem`` class and create an instance.

:: 

    # assign the correct new update functions for qiro from above imports
    qiro_instance = QIROProblem(G, 
                                replacement_routine=create_maxIndep_replacement_routine, 
                                cost_operator= create_maxIndep_cost_operator_reduced,
                                mixer= qiro_RXMixer,
                                cl_cost_function= maxIndepSetclCostfct,
                                init_function= qiro_init_function
                                )

We think of arguments for the ``run_qiro`` function, run the algorithm, et violà! 

:: 

    # We run the qiro instance and get the results!
    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2)

All done! We have solved the NP-hard MIS problem using Quantum-Informed Recursive Optimization! 

Results
-------

But of course we also want to investigate our results, so lets find out about the five most likely solutions the algorithm came up with:

::

    print("QIRO 5 best results")
    maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
    costFunc = maxIndepSetclCostfct(G)
    for key, val in res_qiro.items():  
        if key in maxfive:
            # print the result bitstring and value of the costfunction
            print(key)
            print(costFunc({key:1}))

We do not put example output here, since the algorithm is not deterministic, and the output you receive may differ from what we can put here as an example. So just go ahead and try it yourself!

We can further compare our results to the `NetworkX MIS algorithm <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.mis.maximal_independent_set.html>`_ for solving the MIS problem:

::

    print("Networkx solution")
    print(nx.approximation.maximum_independent_set(G))

Chances are, you will see a result in the QIRO implementation, that is better than the classical algorithm provided by Networkx!

We can also compare these results with the standard QAOA implementation.

::

    from qrisp.qaoa.qaoa_problem import QAOAProblem
    from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetCostOp
    from qrisp.qaoa.mixers import RX_mixer

    Gtwo = create_rdm_graph(num_nodes, 0.4, seed =  107)
    qarg2 = QuantumVariable(Gtwo.number_of_nodes())
    maxindep_instance = QAOAProblem(maxIndepSetCostOp(G), RX_mixer, maxIndepSetclCostfct(G))
    res_qaoa = maxindep_instance.run( qarg = qarg2, depth = 3)

    print("QAOA 5 best results")
    maxfive = sorted(res_qaoa, key=res_qaoa.get, reverse=True)[:5]
    for key, val in res_qaoa.items(): 
        if key in maxfive:
            print(key)
            print(costFunc({key:1}))

As expected, the improvements are drastic, but see for yourself!

As a final caveat, we can look at the graph we are left with after all reduction steps

::

    final_Graph = qiro_instance.problem

Congratulations, you have reached the end of the tutorial and are now capable of solving the MIS problem in Qrisp!
Should your appetite not be satisfied, we advise you to check out our other QIRO implementations:

* maxClique
[REF]

and of course all the other material in the tutorial section.




