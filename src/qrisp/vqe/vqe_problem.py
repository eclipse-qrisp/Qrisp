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

from qrisp import h, QuantumArray, parallelize_qc
from qrisp.vqe.vqe_benchmark_data import VQEBenchmark


class VQEProblem:
    r"""
    Central structure to facilitate treatment of VQE problems.

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
    
    def __init__(self, spin_operator, ansatz_function, num_params, init_function = None, init_type='random', callback=False):
        self.spin_operator = spin_operator
        self.ansatz_function = ansatz_function
        self.num_params = num_params
        
        self.init_function = init_function
        self.cl_post_processor = None
        self.init_type = init_type
        # Fourier heuristic parameterization
        self.fourier_depth = None
        # should be set in the 
        self.init_params = None
        # parameters for callback
        self.callback = callback
        self.optimization_params = []
        self.optimization_costs = []

    def set_callback(self):
        self.callback = True

    def set_init_function(self, init_function):
        """
        Set the initial state preparation function for the VQE problem.

        Parameters
        ----------
        init_function : function
            The initial state preparation function for the specific VQE problem instance.
        """
        self.init_function = init_function
    
    def computeParams(self,p,t_max):
        """
        Compute the angle parameters gamma and beta based on the given inputs. Used for the TQA warm starting the initial state for QAOA.

        Parameters
        ----------
        p : int
            The number of partitions for the time interval.
        t_max : float
            The maximum time value.

        Returns
        -------
        np.array
            A concatenated numpy array of gamma and beta values.
        """
        dt = t_max / p
        t = dt * (np.arange(1, p + 1) - 0.5)
        gamma = (t / t_max) * dt
        beta = (1 - (t / t_max)) * dt
        return  np.concatenate((gamma,beta)) 

    def set_fourier_depth(self, fourier_depth, init_params = None):
        """
        Set the FOURIER heuristic for a QAOA problem.

        Parameters
        ----------
        fourier_depth : int
            Number of Fourier parameters.
        init_params : np.array
            (Optional) NumPy array of initial Fourier Parameters.
        """
        
        self.fourier_depth = fourier_depth
        self.init_params = init_params

      
    def compile_circuit(self, qarg, depth):
        """
        Compiles the circuit that is evaluated by the :meth:`run <qrisp.vqe.VQEProblem.run>` method.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument the cost function is called on.
        depth : int
            The amount of VQE layers.

        Returns
        -------
        compiled_qc : QuantumCircuit
            The parametrized, compiled QuantumCircuit without measurements.
        list[sympy.Symbol]
            A list of the parameters that appear in ``compiled_qc``.

        """
        
        temp = list(qarg.qs.data)
        
        # Define VQE angle parameters theta for VQE circuit
        theta = [Symbol("theta_" + str(i) + str(j)) for i in range(depth) for j in range(self.num_params)]

        # Prepare the initial state for particular problem instance
        if self.init_function is not None:
            self.init_function(qarg)
        else:
            h(qarg)
            
        # Apply p layers of the ansatz
        for i in range(depth):                           
            self.ansatz_function(qarg,[theta[self.num_params*i+j] for j in range(self.num_params)])

        # Compile quantum circuit with intended measurements    
        if isinstance(qarg, QuantumArray):
            intended_measurement_qubits = sum([list(qv) for qv in qarg.flatten()], [])
        else:
            intended_measurement_qubits = list(qarg)
        
        compiled_qc = qarg.qs.compile(intended_measurements=intended_measurement_qubits)
        
        qarg.qs.data = temp
        
        return compiled_qc, theta

    # Set initial random values for optimization parameters 
    #def optimization_routine(self, qarg, compiled_qc, symbols , depth, mes_kwargs, max_iter):    
    def optimization_routine(self, qarg, depth, mes_kwargs, max_iter): 
        """
        Wrapper subroutine for the optimization method used in QAOA. The initial values are set and the optimization via ``COBYLA`` is conducted here.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument the cost function is called on.
        complied_qc : QuantumCircuit
            The compiled quantum circuit.
        depth : int
            The amont of QAOA layers.
        symbols : list
            The list of symbols used in the quantum circuit.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary, as defined in previous functions.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50, as defined in previous functions.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available are ``random`` and TQA. The default is ``random``.

        Returns
        -------
        res_sample
            The optimized parameters of the problem instance.
        """
        
        compiled_qc, symbols = self.compile_circuit(qarg, depth)
        
        # Set initial random values for optimization parameters 
    #    init_point = np.pi * np.random.rand(2 * depth)/2
        
        # initial point is set here, potentially subject to change
        if not isinstance(self.fourier_depth, int):
            init_point = np.pi * np.random.rand(2 * depth)/2
        elif not isinstance(self.init_params, list):
            init_point = np.pi * np.random.rand(2 * self.fourier_depth)/2
        else:

            if len(self.init_params)/2 != self.fourier_depth:
                raise Exception("Fourier-depth does not match match length of init_params! (should be half the length)")
            init_point = self.init_params

        # Define optimization wrapper function to be minimized using VQE
        def optimization_wrapper(theta, qc, symbols, qarg, mes_kwargs):
            """
            Wrapper function for the optimization method used in VQE.

            This function calculates the expected value of the spin operator after post-processing if a post-processing function is set, otherwise it calculates the expexted value of the spin operator.

            Parameters
            ----------
            theta : list
                The list of angle parameters for the VQE circuit.
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
                The expected value of the spin operator.
            """         
            subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}
            
            expectation = qarg.get_spin_measurement(self.spin_operator, subs_dic = subs_dic, precompiled_qc = qc, **mes_kwargs)
            
            if self.callback:
                self.optimization_costs.append(expectation)

            if self.cl_post_processor is not None:
                return self.cl_post_processor(expectation)
            else:
                return expectation
            
        def tqa_angles(p,qc, symbols, qarg_dupl, mes_kwargs,steps=10): #qarg only before
            """
            Compute the optimal parameters for the Time-dependent Quantum Adiabatic (TQA) algorithm.

            The function first creates a linspace array `time` from 0.1 to 10 with `steps` steps. 
            Then for each `t_max` in `time`, it computes the parameters `x` using the `computeParams` 
            function and calculates the energy `qcut` using the `optimization_wrapper` function. 
            The energy values are stored in the `energy` list. The `t_max` corresponding to the 
            minimum energy is found and used to compute the optimal parameters which are returned.

            Parameters
            ----------
            p : int
                The number of partitions for the time interval.
            qc : QuantumCircuit
                The quantum circuit for the specific problem instance.
            symbols : list
                The list of symbols in the quantum circuit.
            qarg_dupl : list
                The list of quantum arguments.
            mes_kwargs : dict
                The measurement keyword arguments.
            steps : int, optional
                The number of steps for the linspace function, default is 10.

            Returns
            -------
            np.array
                A concatenated numpy array of optimal gamma and beta values.
            """
            time = np.linspace(0.1, 10, steps)
            energy = []
            for t_max in time:      
                x=self.computeParams(p,t_max)
                qcut=optimization_wrapper(x,qc,symbols,qarg_dupl,mes_kwargs)
                energy.append(qcut)
            
            idx = np.argmin(energy)
            t_max = time[idx]
            return self.computeParams(p,t_max)

        if self.init_type=='random':
            # Set initial random values for optimization parameters
            init_point = np.pi * np.random.rand(2 * depth)/2

        elif self.init_type=='tqa':
            # TQA initialization
            init_point = tqa_angles(depth,compiled_qc, symbols, qarg, mes_kwargs)

        def optimization_cb(x):
            self.optimization_params.append(x)

        # Perform optimization using COBYLA method
        if isinstance(self.fourier_depth, int):
            from qrisp.qaoa.optimization_wrappers.fourier_wrapper import fourier_optimization_wrapper
            for index_p in range(1, depth + 1):
                compiled_qc, symbols = self.compile_circuit(qarg, index_p)
                res_sample =  minimize(fourier_optimization_wrapper,
                                    init_point, 
                                    method='COBYLA', 
                                    options={'maxiter':max_iter}, 
                                    callback=optimization_cb,
                                    args = (compiled_qc, symbols, qarg, self.cl_cost_function,self.fourier_depth, index_p , mes_kwargs))
                init_point = res_sample['x']
                
        else:
            compiled_qc, symbols = self.compile_circuit(qarg, depth)
            # Perform optimization using COBYLA method
            res_sample = minimize(optimization_wrapper,
                                init_point, 
                                method='COBYLA', 
                                options={'maxiter':max_iter}, 
                                callback=optimization_cb,
                                args = (compiled_qc, symbols, qarg, mes_kwargs))
            
        return res_sample['x']
    
        

    def run(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random"):
        """
        Run the specific VQE problem instance with given quantum arguments, depth of VQE circuit,
        measurement keyword arguments (mes_kwargs) and maximum iterations for optimization (max_iter).
        
        Parameters
        ----------
        qarg : QuantumVariable
            The quantum variable to which the VQE circuit is applied.
        depth : int
            The depth of the VQE circuit.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available are ``random`` and TQA. The default is ``random``.

        Returns
        -------
        opt_res : dict
            The optimal result after running VQE problem for a specific problem instance. It contains the expectation of the spin operator after applying the optimal VQE circuit to the quantum variable.
        """

        #init_point = np.pi * np.random.rand(2 * depth)/2

        #alternative to everything below:
        #bound_qc = self.train_circuit(qarg, depth)
        #opt_res = bound_qc(qarg).get_measurement(**mes_kwargs)
        #return opt_res
        if not "shots" in mes_kwargs:
            mes_kwargs["shots"] = 5000
        
        #res_sample = self.optimization_routine(qarg, compiled_qc, symbols , depth,  mes_kwargs, max_iter)
        optimal_theta = self.optimization_routine(qarg, depth, mes_kwargs, max_iter)
        #if isinstance(self.fourier_depth, int):
        #    from qrisp.qaoa.parameters.fourier_params import fourier_params_helper
        #    optimal_theta = fourier_params_helper(optimal_theta, self.fourier_depth , depth) 
        
        # Prepare initial state - in case this is not called, prepare superposition state
        if self.init_function is not None:
            self.init_function(qarg)
        else:
            h(qarg)

        # Apply p layers of the ansatz    
        for i in range(depth):                          
            self.ansatz_function(qarg,[optimal_theta[self.num_params*i+j] for j in range(self.num_params)])

        opt_res = qarg.get_spin_measurement(self.spin_operator,**mes_kwargs)
        
        return opt_res
    
    def train_function(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random"):
        """
        This function allows for training of a circuit with a given instance of a ``VQEProblem``. It will then return a function that can be applied to a ``QuantumVariable``,
        s.t. that it is a solution to the problem instance. The function therefore acts as a circuit for the problem instance with optimized parameters.

        Parameters
        ----------
        qarg : QuantumVariable
            The quantum variable to which the VQE circuit is applied.
        depth : int
            The depth of the VQE circuit.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.

        Returns
        -------
        circuit_generator : function
            A function that can be applied to a ```QuantumVariable`` , with optimized parameters for the problem instance. The ``QuantumVariable`` then represent a solution of the problem.

        """

        compiled_qc, symbols = self.compile_circuit(qarg, depth)
        optimal_theta = self.optimization_routine(qarg, depth, mes_kwargs, max_iter)
        
        def circuit_generator(qarg_gen):
            if self.init_function is not None:
                self.init_function(qarg_gen)
            else:
                h(qarg_gen)
            for i in range(depth): 
                self.ansatz_function(qarg,[optimal_theta[self.num_params*i+j] for j in range(self.num_params)])
            
        return circuit_generator
    
    def benchmark(self, qarg, depth_range, shot_range, iter_range, optimal_solution, repetitions = 1, mes_kwargs = {}, init_type = "random"):
        """
        This method enables convenient data collection regarding performance of the implementation.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The quantum argument, the benchmark is executed on. Compare to the :meth:`.run <qrisp.vqe.VQEProblem.run>` method.
        depth_range : list[int]
            A list of integers indicating, which depth parameters should be explored. Depth means the amount of VQE layers.
        shot_range : list[int]
            A list of integers indicating, which shots parameters should be explored. Shots means the amount of repetitions, the backend performs per iteration.
        iter_range : list[int]
            A list of integers indicating, what iterations parameter should be explored. Iterations means the amount of backend calls, the optimizer is allowed to do.
        optimal_solution : Float
            The optimal solution to the problem. 
        repetitions : int, optional
            The amount of repetitions, each parameter constellation should go though. Can be used to get a better statistical significance. The default is 1.
        mes_kwargs : dict, optional
            The keyword arguments, that are used for the ``qarg.get_measurement``. The default is {}.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available are ``random`` and TQA. The default is ``random``.

        Returns
        -------
        :ref:`VQEBenchmark`
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
                     "energy" : [],
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
                        if init_type=='random':
                            energy = self.run(qarg=qarg_dupl, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs, init_type='random')
                        elif init_type=='tqa':
                            energy = self.run(qarg=qarg_dupl, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs, init_type='tqa')
                        final_time = time.time() - start_time
                        
                        compiled_qc = qarg_dupl.qs.compile(intended_measurements=mes_qubits)
                        
                        data_dict["layer_depth"].append(p)
                        data_dict["circuit_depth"].append(compiled_qc.depth())
                        data_dict["qubit_amount"].append(compiled_qc.num_qubits())
                        data_dict["shots"].append(s)
                        data_dict["iterations"].append(it)
                        #data_dict["counts"].append(counts)
                        data_dict["energy"].append(energy)
                        data_dict["runtime"].append(final_time)
                        
                        
        return VQEBenchmark(data_dict, optimal_solution, self.spin_operator)
    

    def visualize_cost(self):
        """
        todo

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        if not self.callback:
            raise Exception("Visualization can only be perform on a VQE instance with self.callback=True")
        
        x = list(range(len(self.optimization_costs)))
        y = self.optimization_costs

        plt.scatter(x, y)
        plt.xlabel("Iterations")
        plt.ylabel("Expectation")
        plt.show()