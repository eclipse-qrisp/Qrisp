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
from sympy import Symbol, Basic

from qrisp import h, QuantumArray, parallelize_qc
from qrisp.vqe.vqe_benchmark_data import VQEBenchmark
from qrisp.operators.pauli_operator import PauliOperator


class VQEProblem:
    r"""
    Central structure to facilitate treatment of VQE problems.
    This class encapsulates the Hamiltonian, the ansatz, and the initial state preparation function for a specific VQE problem instance. 
    
    Parameters
    ----------
    hamiltonian : PauliOperator
        The quantum Hamiltonian.
    ansatz_function : function
        A function receiving a :ref:`QuantumVariable` or :ref:`QuantumArray` and a parameter list. This function implements the unitary 
        corresponding to one layer of the ansatz.
    num_params : int
        The number of parameters per layer.
    init_function : function, optional
        A function preparing the initial state.
        By default, the inital state is the $\ket{0}$ state.
    callback : bool, optional
        If ``True``, intermediate results are stored. The default is ``False``.

    Examples
    --------

    For a quick demonstration, we show how to calculate the ground sate energy of the $H_2$ molecule using VQE, as explained `here <https://arxiv.org/abs/2305.07092>`_.
    Note that in the aforementioned paper, the energy is calculated setting the constant term in the Hamiltonian to 0.

    ::

        from qrisp import *
        from qrisp.operators import X,Y,Z

        # Problem Hamiltonian
        c = [-0.81054, 0.16614, 0.16892, 0.17218, -0.22573, 0.12091, 0.166145, 0.04523]
        H = c[0] \
            + c[1]*Z(0)*Z(2) \
            + c[2]*Z(1)*Z(3) \
            + c[3]*(Z(3) + Z(1)) \
            + c[4]*(Z(2) + Z(0)) \
            + c[5]*(Z(2)*Z(3) + Z(0)*Z(1)) \
            + c[6]*(Z(0)*Z(3) + Z(1)*Z(2)) \
            + c[7]*(Y(0)*Y(1)*Y(2)*Y(3) + X(0)*X(1)*Y(2)*Y(3) + Y(0)*Y(1)*X(2)*X(3) + X(0)*X(1)*X(2)*X(3))

        # Ansatz
        def ansatz(qv,theta):
            for i in range(4):
                ry(theta[i],qv[i])
            for i in range(3):
                cx(qv[i],qv[i+1])
            cx(qv[3],qv[0])

        from qrisp.vqe.vqe_problem import *

        vqe = VQEProblem(hamiltonian = H,
                         ansatz_function = ansatz,
                         num_params=4,
                         callback=True)

        energy = vqe.run(qarg = QuantumVariable(4),
                      depth = 1,
                      max_iter=50)
        print(energy)
        # Yields -1.864179046

    We visualize the optimization process:

    >>> vqe.visualize_energy(exact=True)

    .. figure:: /_static/vqeH2.png
        :alt: VQEH2
        :scale: 80%
        :align: center

    """
    
    def __init__(self, hamiltonian, ansatz_function, num_params, init_function = None, callback=False):
        
        self.hamiltonian = hamiltonian
        self.ansatz_function = ansatz_function
        self.num_params = num_params
        self.init_function = init_function
        self.cl_post_processor = None

        # parameters for callback
        self.callback = callback
        self.optimization_params = []
        self.optimization_costs = []

    def set_callback(self):
        """
        Sets ``callback=True`` for saving intermediate results.

        """

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
      
    def compile_circuit(self, qarg, depth):
        """
        Compiles the circuit that is evaluated by the :meth:`run <qrisp.vqe.VQEProblem.run>` method.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument to which the VQE circuit is applied.
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
        
        # Define parameters theta for VQE circuit
        theta = [Symbol("theta_" + str(i) + str(j)) for i in range(depth) for j in range(self.num_params)]

        # Prepare the initial state for particular problem instance, the default is the \ket{0} state
        if self.init_function is not None:
            self.init_function(qarg)
            
        # Apply p layers of the ansatz
        for i in range(depth):                           
            self.ansatz_function(qarg,[theta[self.num_params*i+j] for j in range(self.num_params)])

        # Compile quantum circuit
        compiled_qc = qarg.qs.compile()
        
        qarg.qs.data = temp
        
        return compiled_qc, theta

    def optimization_routine(self, qarg, depth, mes_kwargs, mes_settings, max_iter, init_type="random", init_point=None, optimizer="COBYLA"): 
        """
        Wrapper subroutine for the optimization method used in QAOA. The initial values are set and the optimization via ``COBYLA`` is conducted here.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument the cost function is called on.
        depth : int
            The amont of VQE layers.
        mes_kwargs : dict
            The keyword arguments for the measurement function.
        mes_settings : list
            The measurement setttings for the measurement function.
        max_iter : int
            The maximum number of iterations for the optimization method.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.
        init_point : list[float], optional
            Specifies the initial optimization parameters. 
        optimizer : str, optional
            Specifies the optimization routine. Available are ``COBYLA``, ``COBYQA``, ``Nelder-Mead``. 
            The Default is "COBYLA". 

        Returns
        -------
        res_sample
            The optimized parameters of the problem instance.
        """

        # Define optimization wrapper function to be minimized using VQE
        def optimization_wrapper(theta, qc, symbols, qarg, mes_kwargs, mes_settings):
            """
            Wrapper function for the optimization method used in VQE.

            This function calculates the expected value of the Hamiltonian after post-processing if a post-processing function is set, otherwise it calculates the expected value of the Hamiltonian.

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
            mes_settings : list
                The measurement setttings for the measurement function.

            Returns
            -------
            float
                The expected value of the Hamiltonian.
            """         

            subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}

            expectation = qarg.get_spin_measurement(self.hamiltonian, subs_dic = subs_dic, precompiled_qc = qc, mes_settings=mes_settings, **mes_kwargs)

            if self.callback:
                self.optimization_costs.append(expectation)

            if self.cl_post_processor is not None:
                return self.cl_post_processor(expectation)
            else:
                return expectation

        if init_point is None:
            # Set initial random values for optimization parameters 
            if init_type=='random':
                init_point = np.pi * np.random.rand(depth * self.num_params)/2

        #def optimization_cb(x):
        #    self.optimization_params.append(x)

        # Perform optimization using specified method
        compiled_qc, symbols = self.compile_circuit(qarg, depth)
        res_sample = minimize(optimization_wrapper,
                                init_point, 
                                method=optimizer,
                                options={'maxiter':max_iter}, 
                                args = (compiled_qc, symbols, qarg, mes_kwargs, mes_settings))
            
        return res_sample['x']

    def run(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random", init_point=None, optimizer="COBYLA"):
        """
        Run the specific VQE problem instance with given quantum arguments, depth of VQE circuit,
        measurement keyword arguments (mes_kwargs) and maximum iterations for optimization (max_iter).
        
        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The argument to which the VQE circuit is applied.
        depth : int
            The amount of VQE layers.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
            By default, 5000 shots are executed per measurement setting.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.
        init_point : list[float], optional
            Specifies the initial optimization parameters. 
        optimizer : str, optional
            Specifies the optimization routine. Available are ``COBYLA``, ``COBYQA``, ``Nelder-Mead``. 
            The Default is ``COBYLA``. 

        Returns
        -------
        energy : float
            The expected value of the Hamiltonian after applying the optimal VQE circuit to the quantum variable.
        """

        # Delete callback
        self.optimization_params = []
        self.optimization_costs = []

        if not "shots" in mes_kwargs:
            mes_kwargs["shots"] = 5000

        # Measurement settings
        meas_circs, meas_ops, meas_coeffs, constant_term = self.hamiltonian.get_measurement_settings(qarg, method=mes_kwargs.get("method",None)) 
        mes_settings = [meas_circs, meas_ops, meas_coeffs, constant_term]
        
        optimal_theta = self.optimization_routine(qarg, depth, mes_kwargs, mes_settings, max_iter, init_type, init_point, optimizer)
        
        # Prepare the initial state for particular problem instance, the default is the \ket{0} state
        if self.init_function is not None:
            self.init_function(qarg)

        # Apply p layers of the ansatz    
        for i in range(depth):                          
            self.ansatz_function(qarg,[optimal_theta[self.num_params*i+j] for j in range(self.num_params)])

        opt_res = qarg.get_spin_measurement(self.hamiltonian,mes_settings=mes_settings,**mes_kwargs)
        
        return opt_res
    
    def train_function(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random", init_point=None, optimizer="COBYLA"):
        """
        This function allows for training of a circuit with a given instance of a ``VQEProblem``. It will then return a function that can be applied to a ``QuantumVariable``,
        such that it prepares the ground state of the problem Hamiltonian. The function therefore applies a circuit for the problem instance with optimized parameters.

        Parameters
        ----------
        qarg : QuantumVariable
            The argument to which the VQE circuit is applied.
        depth : int
            The amount of VQE layers.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
            By default, 5000 shots are executed per measurement setting.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.
        init_point : list[float], optional
            Specifies the initial optimization parameters. 
        optimizer : str, optional
            Specifies the optimization routine. Available are ``COBYLA``, ``COBYQA``, ``Nelder-Mead``.
            The Default is "COBYLA". 

        Returns
        -------
        circuit_generator : function
            A function that can be applied to a ``QuantumVariable``, with optimized parameters for the problem instance. 
            The ``QuantumVariable`` then represents the ground state of the problem Hamiltonian.

        """

        if not "shots" in mes_kwargs:
            mes_kwargs["shots"] = 5000

        # Measurement settings
        meas_circs, meas_ops, meas_coeffs, constant_term = self.hamiltonian.get_measurement_settings(qarg, method=mes_kwargs.get("method",None)) 
        mes_settings = [meas_circs, meas_ops, meas_coeffs, constant_term]

        optimal_theta = self.optimization_routine(qarg, depth, mes_kwargs, mes_settings, max_iter, init_type, init_point, optimizer)
        
        def circuit_generator(qarg_gen):
            # Prepare the initial state for particular problem instance, the default is the \ket{0} state
            if self.init_function is not None:
                self.init_function(qarg_gen)

            for i in range(depth): 
                self.ansatz_function(qarg_gen,[optimal_theta[self.num_params*i+j] for j in range(self.num_params)])
            
        return circuit_generator
    
    def benchmark(self, qarg, depth_range, shot_range, iter_range, optimal_solution, repetitions = 1, mes_kwargs = {}, init_type = "random"):
        """
        This method enables convenient data collection regarding performance of the implementation.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The quantum argument the benchmark is executed on. Compare to the :meth:`.run <qrisp.vqe.VQEProblem.run>` method.
        depth_range : list[int]
            A list of integers indicating, which depth parameters should be explored. Depth means the amount of VQE layers.
        shot_range : list[int]
            A list of integers indicating, which shots parameters should be explored. Shots means the amount of repetitions, the backend performs per iteration and per measurement setting.
        iter_range : list[int]
            A list of integers indicating, what iterations parameter should be explored. Iterations means the amount of backend calls, the optimizer is allowed to do.
        optimal_solution : float
            The optimal solution to the problem. 
        repetitions : int, optional
            The amount of repetitions, each parameter constellation should go though. Can be used to get a better statistical significance. The default is 1.
        mes_kwargs : dict, optional
            The keyword arguments, that are used for the ``qarg.get_spin_measurement``. The default is {}.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.

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
            
        The :ref:`VQEBenchmark` class contains a variety of methods to help 
        you drawing conclusions from the collected data. Make sure to check them out!

        """
        
        data_dict = {"layer_depth" : [],
                     "circuit_depth" : [],
                     "qubit_amount" : [],
                     "shots" : [],
                     "iterations" : [],
                     "runtime" : [],
                     "energy" : []
                     }
        
        for p in depth_range:
            for s in shot_range:
                for it in iter_range:
                    for k in range(repetitions):
                        
                        if isinstance(qarg, QuantumArray):
                            qarg_dupl = QuantumArray(qtype = qarg.qtype, shape = qarg.shape)
                        else:
                            qarg_dupl = qarg.duplicate()
                            
                        start_time = time.time()
                        
                        temp_mes_kwargs = dict(mes_kwargs)
                        temp_mes_kwargs["shots"] = s

                        energy = self.run(qarg=qarg_dupl, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs, init_type=init_type)

                        final_time = time.time() - start_time
                    
                        compiled_qc = qarg_dupl.qs.compile()
                        
                        data_dict["layer_depth"].append(p)
                        data_dict["circuit_depth"].append(compiled_qc.depth())
                        data_dict["qubit_amount"].append(compiled_qc.num_qubits())
                        data_dict["shots"].append(s)
                        data_dict["iterations"].append(it)
                        data_dict["energy"].append(energy)
                        data_dict["runtime"].append(final_time)
                        
                        
        return VQEBenchmark(data_dict, optimal_solution, self.hamiltonian)
    

    def visualize_energy(self,exact=False):
        """
        Visualizes the energy during the optimization process.

        Parameters
        ----------
        exact : Boolean
            If ``True``, the exact ground state energy of the spin operator is computed classically, and compared to the energy in the optimization process.
            The default is ``False``.

        """

        import matplotlib.pyplot as plt

        if not self.callback:
            raise Exception("Visualization can only be performed for a VQE instance with callback=True")
        
        if exact:
            exact_solution = self.hamiltonian.ground_state_energy()
            plt.axhline(y=exact_solution, color="#6929C4", linestyle='--', linewidth=2, label='Exact energy')
        
        x = list(range(len(self.optimization_costs)))
        y = self.optimization_costs
        plt.scatter(x, y, color='#20306f',marker="o", linestyle='solid', linewidth=1, label='VQE energy')
        plt.xlabel("Iterations", fontsize=15, color="#444444")
        plt.ylabel("Energy", fontsize=15, color="#444444")
        plt.grid()
        plt.legend()

        plt.show()
