"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

from qrisp.qaoa.qaoa_problem import QAOAProblem
import inspect
import numpy as np
import copy

class QIROProblem(QAOAProblem):
    r"""
    Structure to run QIRO algorithms. The idea is based on the paper by J. Finzgar et al. (https://arxiv.org/pdf/2308.13607.pdf).

    This class encapsulates the cost operator, mixer operator, and classical cost function for a specific QIRO problem instance. It also provides methods to set the initial state preparation function, classical cost post-processing function, and optimizer for the problem.

    For a quick demonstration, we import the relevant functions from already implemented problem instances:
        
    ::
        
        # imports 
        from qrisp.qaoa.qiro_problem import QIROProblem
        from qrisp.qaoa.qaoa_problem import QAOAProblem
        from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
        from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct, maxCliqueCostOp
        from qrisp.qaoa.qiroproblems.qiroMaxCliqueInfrastr import * 
        from qrisp.qaoa.mixers import RX_mixer
        from qrisp.qaoa.qiro_mixers import qiro_init_function, qiro_RXMixer
        from qrisp import QuantumVariable
        import matplotlib.pyplot as plt
        import networkx as nx



        # First we define a graph via the number of nodes and the QuantumVariable arguments
        num_nodes = 15
        G = create_rdm_graph(num_nodes,0.7, seed =  99)
        Gtwo = create_rdm_graph(num_nodes,0.7, seed =  99)
        qarg = QuantumVariable(G.number_of_nodes())
        qarg2 = QuantumVariable(Gtwo.number_of_nodes())

        # set simulator shots
        mes_kwargs = {
            #below should be 5k
            "shots" : 5000
            }

        #assign cost_function and maxclique_instance, normal QAOA
        testCostFun = maxCliqueCostfct(Gtwo)
        maxclique_instance = QAOAProblem(maxCliqueCostOp(G), RX_mixer, maxCliqueCostfct(G))

        # assign the correct new update functions for qiro from above imports
        qiro_instance = QIROProblem(problem = Gtwo,  
                                    replacement_routine = create_maxClique_replacement_routine, 
                                    cost_operator = create_maxClique_cost_operator_reduced,
                                    mixer = qiro_RXMixer,
                                    cl_cost_function = maxCliqueCostfct,
                                    init_function = qiro_init_function
                                    )


        # We run the qiro instance and get the results!
        res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, 
                                        #mes_kwargs = mes_kwargs
                                        )
        # and also the final graph, that has been adjusted
        final_Graph = qiro_instance.problem

        # get the normal QAOA results for a comparison
        res_qaoa = maxclique_instance.run( qarg = qarg2, depth = 3)


        # We can then also print the top 5 results for each...
        print("QAOA 5 best results")
        maxfive = sorted(res_qaoa, key=res_qaoa.get, reverse=True)[:5]
        for key, val in res_qaoa.items(): 
            if key in maxfive:
                print(key)
                print(testCostFun({key:1}))


        print("QIRO 5 best results")
        maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
        for key, val in res_qiro.items():  
            if key in maxfive:
                
                print(key)
                print(testCostFun({key:1}))

        # or compare it with the networkx result of the max_clique algorithm...
        print("Networkx solution")
        print(nx.approximation.max_clique(Gtwo))

        # and finally, we draw the final graph and the original graphs to compare them!
        plt.figure(1)
        nx.draw(final_Graph, with_labels = True)

        plt.figure(2)
        nx.draw(G, with_labels = True)
        plt.show()  
        

    For an in-depth tutorial, make sure to check out :ref:`QIROmaxClique`!
        
    Parameters
    ----------
    problem : Any
        The problem structure to be considered for the algorithm. For example in the case of MaxClique a graph, or MaxSat a list of clauses
    qaoaProblem : QAOAProblem
        A :ref:`QAOAProblem` instance to be used for establishing the correlations.
    replacement_routine  : function
        A routine for adjusting the problem after the highest correlation value has been found.
    qiro_cost_operator  : function
        New ``cost_operator`` for the :ref:`QAOAProblem` instance.
    qiro_mixer_operator  : function
        New ``mixer_operator`` for the :ref:`QAOAProblem` instance.
    qiro_init_function  : function
        New ``mixer_operator`` for the :ref:`QAOAProblem` instance.

    """
    
    #try without QAOAProblem
    """ def __init__(self, problem, qaoaProblem, replacement_routine , qiro_cost_operator, qiro_mixer, qiro_init_function, ):
        super().__init__(qaoaProblem.cost_operator, qaoaProblem.mixer, qaoaProblem.cl_cost_function)
        self.problem = problem
        self.replacement_routine = replacement_routine
        self.qiro_cost_operator = qiro_cost_operator
        self.qiro_mixer = qiro_mixer
        self.qiro_init_function = qiro_init_function """
    
    def __init__(self, problem, replacement_routine , cost_operator, mixer, cl_cost_function, init_function):
        super().__init__(cost_operator(problem), mixer, cl_cost_function(problem))
        self.problem = problem
        self.replacement_routine = replacement_routine
        self.qiro_cost_operator = cost_operator
        self.qiro_mixer = mixer
        self.init_function = init_function
        self.qiro_init_function = init_function
    
    def run_qiro(self, qarg, depth, n_recursions,  mes_kwargs = {}, max_iter = 50):
        """
        Run the specific QIRO problem instance with given quantum arguments, depth of QAOA circuit,
        measurement keyword arguments (mes_kwargs) and maximum iterations for optimization (max_iter).
        
        Parameters
        ----------
        qarg : QuantumVariable
            The quantum variable to which the QAOA circuit is applied.
        depth : int
            The depth of the QAOA circuit.
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
        res= QAOAProblem.run(self, qarg, depth,  mes_kwargs, max_iter)

        corr_vals = []
        solutions = []
        exclusions = []

        for index in range(n_recursions):
            
            new_problem, solutions , sign, exclusions = self.replacement_routine(res, self.problem, solutions , exclusions)
            corr_vals.append(sign)    
            self.problem = new_problem
            self.cost_operator = self.qiro_cost_operator(new_problem, solutions=solutions)
            self.mixer = self.qiro_mixer(new_problem, solutions=solutions)
            self.init_function = self.qiro_init_function(new_problem, solutions=solutions, exclusions = exclusions)

            new_qarg = QuantumVariable(len(qarg))
            res = QAOAProblem.run(self, new_qarg, depth,  mes_kwargs, max_iter)

        return res







