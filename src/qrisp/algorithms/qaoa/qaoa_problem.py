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
from qrisp.algorithms.qaoa.qaoa_benchmark_data import QAOABenchmark


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
    
    def __init__(self, cost_operator, mixer, cl_cost_function, init_function = None, init_type='random'):
        self.cost_operator = cost_operator
        self.mixer = mixer
        self.cl_cost_function = cl_cost_function
        
        self.init_function = init_function
        self.cl_post_processor = None
        self.init_type = init_type
        # Fourier heuristic parameterization
        self.fourier_depth = None
        # should be set in the 
        self.init_params = None

    def set_init_function(self, init_function):
        """
        Set the initial state preparation function for the QAOA problem.

        Parameters
        ----------
        init_function : function
            The initial state preparation function for the specific QAOA problem instance.
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

    def computeParams_dt(self,p,dt):
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
        t = (np.arange(1, p + 1) - 0.5)/p
        gamma = t * dt
        beta = (1 - t) * dt
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
        Compiles the circuit that is evaluated by the :meth:`run <qrisp.qaoa.QAOAProblem.run>` method.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument the cost function is called on.
        depth : int
            The amont of QAOA layers.


        Returns
        -------
        compiled_qc : QuantumCircuit
            The parametrized, compiled QuantumCircuit without measurements.
        list[sympy.Symbol]
            A list of the parameters that appear in ``compiled_qc``.

        Examples
        --------
        
        We create a MaxCut instance and compile the circuit:
        
            
        >>> from networkx import Graph
        >>> G = Graph([(0,1),(1,2),(2,0)])
        >>> from qrisp.qaoa import maxcut_problem
        >>> from qrisp import QuantumVariable
        >>> p = 5
        >>> qaoa_instance = maxcut_problem(G)
        >>> qarg = QuantumVariable(len(G))
        >>> qrisp_qc, symbols = qaoa_instance.compile_circuit(qarg, p)
        >>> print(qrisp_qc)
                     ┌───┐┌────────┐┌────────┐                          »
        qarg_dupl.0: ┤ H ├┤ gphase ├┤ gphase ├──■────────────────────■──»
                     ├───┤└────────┘└────────┘┌─┴─┐┌──────────────┐  │  »
        qarg_dupl.1: ┤ H ├────────────────────┤ X ├┤ P(2*gamma_0) ├──┼──»
                     ├───┤                    └───┘└──────────────┘┌─┴─┐»
        qarg_dupl.2: ┤ H ├─────────────────────────────────────────┤ X ├»
                     └───┘                                         └───┘»
        «                                            ┌──────────────┐   ┌────────┐   »
        «qarg_dupl.0: ───────■────────────────────■──┤ Rx(2*beta_0) ├───┤ gphase ├───»
        «                  ┌─┴─┐      ┌────────┐  │  └──────────────┘   └────────┘   »
        «qarg_dupl.1: ─────┤ X ├──────┤ gphase ├──┼─────────■────────────────────────»
        «             ┌────┴───┴─────┐└────────┘┌─┴─┐     ┌─┴─┐      ┌──────────────┐»
        «qarg_dupl.2: ┤ P(2*gamma_0) ├──────────┤ X ├─────┤ X ├──────┤ P(2*gamma_0) ├»
        «             └──────────────┘          └───┘     └───┘      └──────────────┘»
        «             ┌────────┐                                          »
        «qarg_dupl.0: ┤ gphase ├──────────────────■────────────────────■──»
        «             └────────┘┌──────────────┐┌─┴─┐┌──────────────┐  │  »
        «qarg_dupl.1: ────■─────┤ Rx(2*beta_0) ├┤ X ├┤ P(2*gamma_1) ├──┼──»
        «               ┌─┴─┐   ├──────────────┤└───┘└──────────────┘┌─┴─┐»
        «qarg_dupl.2: ──┤ X ├───┤ Rx(2*beta_0) ├─────────────────────┤ X ├»
        «               └───┘   └──────────────┘                     └───┘»
        «                                            ┌──────────────┐   ┌────────┐   »
        «qarg_dupl.0: ───────■────────────────────■──┤ Rx(2*beta_1) ├───┤ gphase ├───»
        «                  ┌─┴─┐      ┌────────┐  │  └──────────────┘   └────────┘   »
        «qarg_dupl.1: ─────┤ X ├──────┤ gphase ├──┼─────────■────────────────────────»
        «             ┌────┴───┴─────┐└────────┘┌─┴─┐     ┌─┴─┐      ┌──────────────┐»
        «qarg_dupl.2: ┤ P(2*gamma_1) ├──────────┤ X ├─────┤ X ├──────┤ P(2*gamma_1) ├»
        «             └──────────────┘          └───┘     └───┘      └──────────────┘»
        «             ┌────────┐                                          »
        «qarg_dupl.0: ┤ gphase ├──────────────────■────────────────────■──»
        «             └────────┘┌──────────────┐┌─┴─┐┌──────────────┐  │  »
        «qarg_dupl.1: ────■─────┤ Rx(2*beta_1) ├┤ X ├┤ P(2*gamma_2) ├──┼──»
        «               ┌─┴─┐   ├──────────────┤└───┘└──────────────┘┌─┴─┐»
        «qarg_dupl.2: ──┤ X ├───┤ Rx(2*beta_1) ├─────────────────────┤ X ├»
        «               └───┘   └──────────────┘                     └───┘»
        «                                            ┌──────────────┐   ┌────────┐   »
        «qarg_dupl.0: ───────■────────────────────■──┤ Rx(2*beta_2) ├───┤ gphase ├───»
        «                  ┌─┴─┐      ┌────────┐  │  └──────────────┘   └────────┘   »
        «qarg_dupl.1: ─────┤ X ├──────┤ gphase ├──┼─────────■────────────────────────»
        «             ┌────┴───┴─────┐└────────┘┌─┴─┐     ┌─┴─┐      ┌──────────────┐»
        «qarg_dupl.2: ┤ P(2*gamma_2) ├──────────┤ X ├─────┤ X ├──────┤ P(2*gamma_2) ├»
        «             └──────────────┘          └───┘     └───┘      └──────────────┘»
        «             ┌────────┐                                          »
        «qarg_dupl.0: ┤ gphase ├──────────────────■────────────────────■──»
        «             └────────┘┌──────────────┐┌─┴─┐┌──────────────┐  │  »
        «qarg_dupl.1: ────■─────┤ Rx(2*beta_2) ├┤ X ├┤ P(2*gamma_3) ├──┼──»
        «               ┌─┴─┐   ├──────────────┤└───┘└──────────────┘┌─┴─┐»
        «qarg_dupl.2: ──┤ X ├───┤ Rx(2*beta_2) ├─────────────────────┤ X ├»
        «               └───┘   └──────────────┘                     └───┘»
        «                                            ┌──────────────┐   ┌────────┐   »
        «qarg_dupl.0: ───────■────────────────────■──┤ Rx(2*beta_3) ├───┤ gphase ├───»
        «                  ┌─┴─┐      ┌────────┐  │  └──────────────┘   └────────┘   »
        «qarg_dupl.1: ─────┤ X ├──────┤ gphase ├──┼─────────■────────────────────────»
        «             ┌────┴───┴─────┐└────────┘┌─┴─┐     ┌─┴─┐      ┌──────────────┐»
        «qarg_dupl.2: ┤ P(2*gamma_3) ├──────────┤ X ├─────┤ X ├──────┤ P(2*gamma_3) ├»
        «             └──────────────┘          └───┘     └───┘      └──────────────┘»
        «             ┌────────┐                                          »
        «qarg_dupl.0: ┤ gphase ├──────────────────■────────────────────■──»
        «             └────────┘┌──────────────┐┌─┴─┐┌──────────────┐  │  »
        «qarg_dupl.1: ────■─────┤ Rx(2*beta_3) ├┤ X ├┤ P(2*gamma_4) ├──┼──»
        «               ┌─┴─┐   ├──────────────┤└───┘└──────────────┘┌─┴─┐»
        «qarg_dupl.2: ──┤ X ├───┤ Rx(2*beta_3) ├─────────────────────┤ X ├»
        «               └───┘   └──────────────┘                     └───┘»
        «                                            ┌──────────────┐                »
        «qarg_dupl.0: ───────■────────────────────■──┤ Rx(2*beta_4) ├────────────────»
        «                  ┌─┴─┐      ┌────────┐  │  └──────────────┘                »
        «qarg_dupl.1: ─────┤ X ├──────┤ gphase ├──┼─────────■────────────────────────»
        «             ┌────┴───┴─────┐└────────┘┌─┴─┐     ┌─┴─┐      ┌──────────────┐»
        «qarg_dupl.2: ┤ P(2*gamma_4) ├──────────┤ X ├─────┤ X ├──────┤ P(2*gamma_4) ├»
        «             └──────────────┘          └───┘     └───┘      └──────────────┘»
        «                                  
        «qarg_dupl.0: ─────────────────────
        «                  ┌──────────────┐
        «qarg_dupl.1: ──■──┤ Rx(2*beta_4) ├
        «             ┌─┴─┐├──────────────┤
        «qarg_dupl.2: ┤ X ├┤ Rx(2*beta_4) ├
        «             └───┘└──────────────┘
        """
        
        temp = list(qarg.qs.data)
        
        # Define QAOA angle parameters gamma and beta for QAOA circuit
        
        gamma = [Symbol("gamma_" + str(i)) for i in range(depth)]
        beta = [Symbol("beta_" + str(i)) for i in range(depth)]
        

        # Prepare the initial state for particular problem instance (MaxCut: superposition, GraphColoring: any node coloring)
        if self.init_function is not None:
            self.init_function(qarg)
        else:
            h(qarg)
            
        # Apply p layers of phase separators and mixers
        for i in range(depth):                           
            self.cost_operator(qarg, gamma[i])
            self.mixer(qarg, beta[i])

        # Compile quantum circuit with intended measurements    
        if isinstance(qarg, QuantumArray):
            intended_measurement_qubits = sum([list(qv) for qv in qarg.flatten()], [])
        else:
            intended_measurement_qubits = list(qarg)
        
        compiled_qc = qarg.qs.compile(intended_measurements=intended_measurement_qubits)
        
        qarg.qs.data = temp
        
        return compiled_qc, gamma + beta

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

        # Define optimization wrapper function to be minimized using QAOA
        def optimization_wrapper(theta, qc, symbols, qarg, mes_kwargs):
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
            
            res_dic = qarg.get_measurement(subs_dic = subs_dic, precompiled_qc = qc, **mes_kwargs)
            
            cl_cost = self.cl_cost_function(res_dic)
            
            if self.cl_post_processor is not None:
                return self.cl_post_processor(cl_cost)
            else:
                return cl_cost
            
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
            #time = np.linspace(0.1, 10, steps)
            dt = np.linspace(0.1, 2, 20)
            print(dt)
            energy = []
            for dt_ in dt:      
                x=self.computeParams_dt(p,dt_)
                qcut=optimization_wrapper(x,qc,symbols,qarg_dupl,mes_kwargs)
                energy.append(qcut)
            
            idx = np.argmin(energy)
            dt_max = dt[idx]
            print(dt_max)
            return self.computeParams_dt(p,dt_max)

        if self.init_type=='random':
            # Set initial random values for optimization parameters
            init_point = np.pi * np.random.rand(2 * depth)/2

        elif self.init_type=='tqa':
            # TQA initialization
            init_point = tqa_angles(depth,compiled_qc, symbols, qarg, mes_kwargs)



        # Perform optimization using COBYLA method
        if isinstance(self.fourier_depth, int):
            from qrisp.qaoa.optimization_wrappers.fourier_wrapper import fourier_optimization_wrapper
            for index_p in range(1, depth + 1):
                compiled_qc, symbols = self.compile_circuit(qarg, index_p)
                res_sample =  minimize(fourier_optimization_wrapper,
                                    init_point, 
                                    method='COBYLA', 
                                    options={'maxiter':max_iter}, 
                                    args = (compiled_qc, symbols, qarg, self.cl_cost_function,self.fourier_depth, index_p , mes_kwargs))
                init_point = res_sample['x']
                
        else:
            compiled_qc, symbols = self.compile_circuit(qarg, depth)
            # Perform optimization using COBYLA method
            res_sample = minimize(optimization_wrapper,
                                init_point, 
                                method='COBYLA', 
                                options={'maxiter':max_iter}, 
                                args = (compiled_qc, symbols, qarg, mes_kwargs))
            
        return res_sample['x']
    
        

    def run(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random"):
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
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available are ``random`` and TQA. The default is ``random``.

        Returns
        -------
        opt_res : dict
            The optimal result after running QAOA problem for a specific problem instance. It contains the measurement results after applying the optimal QAOA circuit to the quantum variable.
        """

        #init_point = np.pi * np.random.rand(2 * depth)/2

        #alternative to everything below:
        #bound_qc = self.train_circuit(qarg, depth)
        #opt_res = bound_qc(qarg).get_measurement(**mes_kwargs)
        #return opt_res
        if not "shots" in mes_kwargs:
            mes_kwargs["shots"] = 5000
        
        #res_sample = self.optimization_routine(qarg, compiled_qc, symbols , depth,  mes_kwargs, max_iter)
        res_sample = self.optimization_routine(qarg, depth, mes_kwargs, max_iter)
        optimal_theta = res_sample 
        if isinstance(self.fourier_depth, int):
            from qrisp.qaoa.parameters.fourier_params import fourier_params_helper
            optimal_theta = fourier_params_helper(optimal_theta, self.fourier_depth , depth) 
        
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
    
    def train_function(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random"):
        """
        This function allows for training of a circuit with a given instance of a ``QAOAProblem``. It will then return a function that can be applied to a ``QuantumVariable``,
        s.t. that it is a solution to the problem instance. The function therefore acts as a circuit for the problem instance with optimized parameters.

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
        circuit_generator : function
            A function that can be applied to a ```QuantumVariable`` , with optimized parameters for the problem instance. The ``QuantumVariable`` then represent a solution of the problem.

        Examples
        --------

        We create a MaxClique instance and train the ``QAOAProblem`` instance
        
        ::
            
            from qrisp.qaoa import QAOAProblem
            from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct,maxCliqueCostOp,init_state
            from qrisp.qaoa.mixers import RX_mixer
            from qrisp import QuantumVariable
            import networkx as nx
            
	        #create QAOAinstance
            G = nx.erdos_renyi_graph(9,0.7, seed =  133)
	        QAOAinstance = QAOAProblem(maxCliqueCostOp(G), RX_mixer, maxCliqueCostfct(G))
	        QAOAinstance.set_init_function(init_function=init_state)

            # create a blueprint-qv to train the problem instance on and train it
            qarg_new = QuantumVariable(G.number_of_nodes())
            training_func = QAOAinstance.train_circuit( qarg=qarg_new, depth=5 )

            # apply the trained function to a new qv 
            qarg_trained = QuantumVariable(G.number_of_nodes())
            training_func(qarg_trained)

            # get the results in a nice format
            opt_res = qarg_trained.get_measurement()
            aClCostFct = maxCliqueCostfct(G)

            print("5 most likely Solutions") 
            maxfive = sorted(opt_res, key=opt_res.get, reverse=True)[:5]
            for res, val in opt_res.items():  
                if res in maxfive:

                    print((res, val))
                    print(aClCostFct({res : 1})) 

        """

        compiled_qc, symbols = self.compile_circuit(qarg, depth)
        res_sample = self.optimization_routine(qarg, depth, mes_kwargs, max_iter)
        
        def circuit_generator(qarg_gen):
            if self.init_function is not None:
                self.init_function(qarg_gen)
            else:
                h(qarg_gen)
            for i in range(depth): 
                
                self.cost_operator(qarg_gen, res_sample[i])
                self.mixer(qarg_gen, res_sample[i+depth])
            
        return circuit_generator
    
    def benchmark(self, qarg, depth_range, shot_range, iter_range, optimal_solution, repetitions = 1, mes_kwargs = {}, init_type = "random"):
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
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available are ``random`` and TQA. The default is ``random``.

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
                        if init_type=='random':
                            counts = self.run(qarg=qarg_dupl, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs, init_type='random')
                        elif init_type=='tqa':
                            counts = self.run(qarg=qarg_dupl, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs, init_type='tqa')
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
