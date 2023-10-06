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

import time

import numpy as np
from scipy.optimize import minimize
from sympy import Symbol

from qrisp import h, QuantumArray
from qrisp.qaoa.qaoa_benchmark_data import QAOABenchmark


class QAOAProblem:
    r"""
    Central structure to facilitate treatment of QAOA problems.

    This class encapsulates the cost operator, mixer operator, and classical cost function for a specific QAOA problem instance. It also provides methods to set the initial state preparation function, classical cost post-processing function, and optimizer for the problem.

    For a quick demonstration, we import the relevant functions from already implemented problem instances:
        
    ::
        
        from networkx import Graph

        G = Graph()

        G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

        from qrisp.qaoa import (QAOAProblem, 
                                create_maxcut_cost_operator, 
                                create_maxcut_cl_cost_function,
                                RX_mixer)

        maxcut_instance = QAOAProblem(cost_operator = create_maxcut_cost_operator(G),
                                       mixer = RX_mixer,
                                       cl_cost_function = create_maxcut_cl_cost_function(G))
        
        
        from qrisp import QuantumVariable
        
        res = maxcut_instance.run(qarg = QuantumVariable(5),
                                  depth = 4, 
                                  max_iter = 25)
        
        print(res)
        #Yields: {'11100': 0.2847, '00011': 0.2847, '10000': 0.0219, '01000': 0.0219, '00100': 0.0219, '11011': 0.0219, '10111': 0.0219, '01111': 0.0219, '00010': 0.02, '11110': 0.02, '00001': 0.02, '11101': 0.02, '00000': 0.0173, '11111': 0.0173, '10010': 0.0143, '01010': 0.0143, '11010': 0.0143, '00110': 0.0143, '10110': 0.0143, '01110': 0.0143, '10001': 0.0143, '01001': 0.0143, '11001': 0.0143, '00101': 0.0143, '10101': 0.0143, '01101': 0.0143, '11000': 0.0021, '10100': 0.0021, '01100': 0.0021, '10011': 0.0021, '01011': 0.0021, '00111': 0.0021}
        
    For an in-depth tutorial, make sure to check out :ref:`MaxCutQAOA`!
        
    Parameters
    ----------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` or :ref:`QuantumArray` and parameter $\gamma$. This function should perform the application of the cost operator.
    mixer : function
        A function receiving a :ref:`QuantumVariable` or :ref:`QuantumArray` and parameter $\beta$. This function should perform the application mixing operator.
    cl_cost_function : function
        The classical cost function for the specific QAOA problem instance.

    """
    
    def __init__(self, cost_operator, mixer, cl_cost_function):
        self.cost_operator = cost_operator
        self.mixer = mixer
        self.cl_cost_function = cl_cost_function
        
        self.init_function = None
        self.cl_post_processor = None

    def set_init_function(self, init_function):
        """
        Set the initial state preparation function for the QAOA problem.

        Parameters
        ----------
        init_function : function
            The initial state preparation function for the specific QAOA problem instance.
        """
        self.init_function = init_function
        
    
    
    def run(self, qarg, depth, mes_kwargs = {}, max_iter = 50):
        """
        Run the specific QAOA problem instance with given quantum arguments, depth of QAOA circuit,
        measurement keyword arguments (mes_kwargs) and maximum iterations for optimization (max_iter).
        
        Parameters
        ----------
        qarg : QuantumVariable
            The quantum variable to which the QAOA circuit is applied.
        depth : int
            The depth of the QAOA circuit.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.

        Returns
        -------
        opt_res : dict
            The optimal result after running QAOA problem for a specific problem instance. It contains the measurement results after applying the optimal QAOA circuit to the quantum variable.
        """       
        # Define QAOA angle parameters gamma and beta for QAOA circuit
        gamma = [Symbol("gamma_" + str(i)) for i in range(depth)]
        beta = [Symbol("beta_" + str(i)) for i in range(depth)]
        
        # Duplicate quantum arguments
        if isinstance(qarg, QuantumArray):
            qarg_dupl = QuantumArray(qtype = qarg.qtype, shape = qarg.shape)
        else:
            qarg_dupl = qarg.duplicate()
        
        # Prepare the initial state for particular problem instance (MaxCut: superposition, GraphColoring: any node coloring)
        if self.init_function is not None:
            self.init_function(qarg_dupl)
        else:
            h(qarg_dupl)
            
        # Apply p layers of phase separators and mixers
        for i in range(depth):                           
            self.cost_operator(qarg_dupl, gamma[i])
            self.mixer(qarg_dupl, beta[i])

        # Compile quantum circuit with intended measurements    
        if isinstance(qarg, QuantumArray):
            intended_measurement_qubits = sum([list(qv) for qv in qarg_dupl.flatten()], [])
        else:
            intended_measurement_qubits = list(qarg_dupl)
        
        compiled_qc = qarg_dupl.qs.compile(intended_measurements=intended_measurement_qubits)
        
        # Set initial random values for optimization parameters 
        init_point = np.pi * np.random.rand(2 * depth)/2
        
        # Define optimization wrapper function to be minimized using QAOA
        def optimization_wrapper(theta, qc, symbols, qarg_dupl, mes_kwargs):
            """
            Wrapper function for the optimization method used in QAOA.

            This function calculates the value of the classical cost function after post-processing if a post-processing function is set, otherwise it calculates the value of the classical cost function.

            Parameters
            ----------
            theta : list
                The list of angle parameters gamma and beta for the QAOA circuit.
            qc : QuantumCircuit
                The compiled quantum circuit.
            symbols : list
                The list of symbols used in the quantum circuit.
            qarg_dupl : QuantumVariable
                The duplicated quantum variable to which the quantum circuit is applied.
            mes_kwargs : dict
                The keyword arguments for the measurement function.

            Returns
            -------
            float
                The value of the classical cost function.
            """         
            subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}
            
            res_dic = qarg_dupl.get_measurement(subs_dic = subs_dic, precompiled_qc = qc, **mes_kwargs)
            
            cl_cost = self.cl_cost_function(res_dic)
            
            if self.cl_post_processor is not None:
                return self.cl_post_processor(cl_cost)
            else:
                return cl_cost
            
        
        # Perform optimization using COBYLA method
        res_sample = minimize(optimization_wrapper,
                              init_point, 
                              method='COBYLA', 
                              options={'maxiter':max_iter}, 
                              args = (compiled_qc, gamma+beta, qarg_dupl, mes_kwargs))
        
        
        optimal_theta = res_sample['x']
        
        # Prepare initial state - in case this is not called, prepare superposition state
        if self.init_function is not None:
            self.init_function(qarg)
        else:
            h(qarg)

        # Apply p layers of phase separators and mixers    
        for i in range(depth):                          
            self.cost_operator(qarg, optimal_theta[i])
            self.mixer(qarg, optimal_theta[i+depth])
        opt_res = qarg.get_measurement(**mes_kwargs)
        
        return opt_res
    
    def benchmark(self, qarg, depth_range, shot_range, iter_range, optimal_solution, repetitions = 1, mes_kwargs = {}):
        """
        This method enables convenient data collection regarding performance of the implementation.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The quantum argument, the benchmark is executed on. Compare to the :meth:`.run <qrisp.qaoa.QAOAProblem.run>` method.
        depth_range : list[int]
            A list of integers indicating, which depth parameters should be explored. Depth means the amount of QAOA layers.
        shot_range : list[int]
            A list of integers indicating, which shots parameters should be explored. Shots means the amount of repetitions, the backend performs per iteration.
        iter_range : list[int]
            A list of integers indicating, what iterations parameter should be explored. Iterations means the amount of backend calls, the optimizer is allowed to do.
        optimal_solution : -
            The optimal solution to the problem. Should have the same type as the keys of the result of ``qarg.get_measurement()``.
        repetitions : int, optional
            The amount of repetitions, each parameter constellation should go though. Can be used to get a better statistical significance. The default is 1.
        mes_kwargs : dict, optional
            The keyword arguments, that are used for the ``qarg.get_measurement``. The default is {}.

        Returns
        -------
        :ref:`QAOABenchmark`
            The results of the benchmark.
            
        Examples
        --------
        
        We create a MaxCut instance and benchmark several parameters
        
        ::
            
            from qrisp import *
            from networkx import Graph
            G = Graph()

            G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

            from qrisp.qaoa import maxcut_problem

            max_cut_instance = maxcut_problem(G)

            benchmark_data = max_cut_instance.benchmark(qarg = QuantumVariable(5),
                                       depth_range = [3,4,5],
                                       shot_range = [5000, 10000],
                                       iter_range = [25, 50],
                                       optimal_solution = "11100",
                                       repetitions = 2
                                       )
        
        We can investigate the data by calling ``visualize``:
        
        ::
            
            benchmark_data.visualize()
        
        .. image:: benchmark_plot.png
            
        The :ref:`QAOABenchmark` class contains a variety of methods to help 
        you drawing conclusions from the collected data. Make sure to check them out!

        """
        
        data_dict = {"layer_depth" : [],
                     "circuit_depth" : [],
                     "qubit_amount" : [],
                     "shots" : [],
                     "iterations" : [],
                     "counts" : [],
                     "runtime" : [],
                     "cl_cost" : []
                     }
        
        for p in depth_range:
            for s in shot_range:
                for it in iter_range:
                    for k in range(repetitions):
                        
                        if isinstance(qarg, QuantumArray):
                            qarg_dupl = QuantumArray(qtype = qarg.qtype, shape = qarg.shape)
                            mes_qubits = sum([qv.reg for qv in qarg_dupl.flatten()], [])
                        else:
                            qarg_dupl = qarg.duplicate()
                            mes_qubits = list(qarg_dupl)
                            
                        start_time = time.time()
                        
                        temp_mes_kwargs = dict(mes_kwargs)
                        temp_mes_kwargs["shots"] = s
                        counts = self.run(qarg=qarg_dupl, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs)
                        final_time = time.time() - start_time
                        
                        compiled_qc = qarg_dupl.qs.compile(intended_measurements=mes_qubits)
                        
                        data_dict["layer_depth"].append(p)
                        data_dict["circuit_depth"].append(compiled_qc.depth())
                        data_dict["qubit_amount"].append(compiled_qc.num_qubits())
                        data_dict["shots"].append(s)
                        data_dict["iterations"].append(it)
                        data_dict["counts"].append(counts)
                        data_dict["runtime"].append(final_time)
                        
                        
        return QAOABenchmark(data_dict, optimal_solution, self.cl_cost_function)
    
    def visualize_cost(self):
        """
        TODO

        Returns
        -------
        None.

        """
        pass