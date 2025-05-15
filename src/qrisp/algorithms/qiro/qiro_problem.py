"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

from qrisp.algorithms.qaoa.qaoa_problem import QAOAProblem
import inspect
import numpy as np
import copy


class QIROProblem(QAOAProblem):
    r"""
    Central structure to run QIRO algorithms. A subclass of the :ref:`QAOAProblem` class.
    The idea is based on the paper by `J. Finzgar et al. <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327>`_.

    This class encapsulates the replacement routine, cost operator, mixer operator, classical cost function
    and initial state preparation function for a specific QIRO problem instance.

    For a quick demonstration, we compare QAOA and QIRO for solving a MaxClique problem instance:

    ::

        from qrisp import QuantumVariable
        from qrisp.qiro import QIROProblem, create_max_clique_replacement_routine, create_max_clique_cost_operator_reduced, qiro_RXMixer, qiro_init_function
        from qrisp.qaoa import max_clique_problem, create_max_clique_cl_cost_function
        import matplotlib.pyplot as plt
        import networkx as nx

        # Define a random graph via the number of nodes and the QuantumVariable arguments
        num_nodes = 15
        G = nx.erdos_renyi_graph(num_nodes, 0.7, seed = 99)
        Gtwo = G.copy()

        qarg = QuantumVariable(G.number_of_nodes())
        qarg2 = QuantumVariable(Gtwo.number_of_nodes())

        # QAOA
        qaoa_instance = max_clique_problem(G)
        res_qaoa = qaoa_instance.run(qarg=qarg, depth=3)

        # QIRO
        qiro_instance = QIROProblem(problem = Gtwo,
                                    replacement_routine = create_max_clique_replacement_routine,
                                    cost_operator = create_max_clique_cost_operator_reduced,
                                    mixer = qiro_RXMixer,
                                    cl_cost_function = create_max_clique_cl_cost_function,
                                    init_function = qiro_init_function
                                    )
        res_qiro = qiro_instance.run_qiro(qarg=qarg2, depth=3, n_recursions = 2)

        # The final graph that has been adjusted
        final_graph = qiro_instance.problem

        cl_cost = create_max_clique_cl_cost_function(G)

        print("5 most likely QAOA solutions")
        max_five_qaoa = sorted(res_qaoa, key=res_qaoa.get, reverse=True)[:5]
        for res in max_five_qaoa:
            print([index for index, value in enumerate(res) if value == '1'])
            print(cl_cost({res : 1}))

        print("5 most likely QIRO solutions")
        max_five_qiro = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
        for res in max_five_qiro:
            print([index for index, value in enumerate(res) if value == '1'])
            print(cl_cost({res : 1}))

        print("Networkx solution")
        print(nx.approximation.max_clique(G))

        # Draw the final graph and the original graph for comparison
        plt.figure(1)
        nx.draw(final_graph, with_labels = True, node_color='#ADD8E6', edge_color='#D3D3D3')
        plt.title('final QIRO graph')

        plt.figure(2)
        most_likely = [index for index, value in enumerate(max_five_qiro[0]) if value == '1']
        nx.draw(G, with_labels=True,
                node_color=['#FFCCCB' if node in most_likely else '#ADD8E6' for node in G.nodes()],
                edge_color=['#FFCCCB' if edge[0] in most_likely and edge[1] in most_likely else '#D3D3D3' for edge in G.edges()])
        plt.title('Original graph with most likely QIRO solution')
        plt.show()


    For an in-depth tutorial, make sure to check out :ref:`the QIRO tutorial <Qiro_tutorial>`!

    Parameters
    ----------
    problem : Any
        The problem structure to be considered for the algorithm. For example, in the case of MaxClique a graph, or in the case of MaxSat a list of clauses.
    replacement_routine  : function
        A routine for adjusting the problem after the highest correlation value was found.
    cost_operator  : function
        Prepares the new ``cost_operator`` for the updated :ref:`QAOAProblem` instance.
        A function that receives a ``problem`` and a list of ``solutions``, and returns a function
        that is applied to a :ref:`QuantumVariable` and a real parameter $\gamma$.
    mixer : function
        Prepares the new ``mixer`` for the updated :ref:`QAOAProblem` instance.
        A function that receives a list of ``solutions`` and a list of ``exclusions``, and returns a function
        that is applied to a :ref:`QuantumVariable` and a real parameter $\beta$.
    cl_cost_function : function
        The classical cost function for the problem instance, which takes a dictionary of measurement results as input.
    init_function  : function
        Prepares the new ``init_function`` for the updated :ref:`QAOAProblem` instance.
        A function that receives a list of ``solutions`` and a list of ``exclusions``, and returns a function
        that is applied to a :ref:`QuantumVariable`.

    """

    def __init__(
        self,
        problem,
        replacement_routine,
        cost_operator,
        mixer,
        cl_cost_function,
        init_function,
        revert=False,
    ):

        super().__init__(
            cost_operator([problem, [], []]),
            mixer([problem, [], []]),
            cl_cost_function(problem),
        )
        self.qiro_cost_operator = cost_operator
        self.qiro_mixer = mixer

        self.problem = copy.deepcopy(problem)
        self.replacement_routine = replacement_routine

        self.init_function = init_function()
        self.qiro_init_function = init_function

    def run_qiro(self, qarg, depth, n_recursions, mes_kwargs={}, max_iter=50):
        """
        Run the specific QIRO problem instance with given quantum argument, depth of QAOA circuit, number of recursions,
        measurement keyword arguments (mes_kwargs) and maximum iterations for optimization (max_iter).

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The quantum variable to which the QAOA circuit is applied.
        depth : int
            The amount of QAOA layers.
        n_recursions : int
            The number of QIRO replacement iterations.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.

        Returns
        -------
        opt_res : dict
            The optimal result after running QAOA problem for a specific problem instance. It contains the measurement results after applying the optimal QAOA circuit to the quantum variable.

        """

        from qrisp import QuantumVariable

        self.set_init_function(self.init_function)
        res = self.run(qarg, depth, mes_kwargs, max_iter)

        corr_vals = []
        solutions = []
        exclusions = []

        for index in range(n_recursions):

            new_problem, solutions, sign, exclusions = self.replacement_routine(
                res, [self.problem, solutions, exclusions]
            )

            corr_vals.append(sign)
            self.problem = new_problem

            self.cost_operator = self.qiro_cost_operator(
                [new_problem, solutions, exclusions]
            )
            self.mixer = self.qiro_mixer([new_problem, solutions, exclusions])
            self.init_function = self.qiro_init_function(  # problem = new_problem,
                solutions=solutions, exclusions=exclusions
            )

            new_qarg = QuantumVariable(len(qarg))
            res = self.run(new_qarg, depth, mes_kwargs, max_iter)

        return res
