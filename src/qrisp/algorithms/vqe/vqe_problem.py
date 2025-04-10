"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
import warnings

import numpy as np
from scipy.optimize import minimize
from sympy import Symbol

from qrisp.algorithms.vqe.vqe_benchmark_data import VQEBenchmark
from qrisp.operators.qubit.measurement import QubitOperatorMeasurement
from qrisp.operators.fermionic import FermionicOperator

import jax
import jax.numpy as jnp
from qrisp.jasp import check_for_tracing_mode
from qrisp.jasp.optimization_tools.optimize import minimize as jasp_minimize


class VQEProblem:
    r"""
    Central structure to facilitate treatment of VQE problems.
    This class encapsulates the Hamiltonian, the ansatz, and the initial state preparation function for a specific VQE problem instance. 
    
    Parameters
    ----------
    hamiltonian : :ref:`QubitOperator` or :ref:`FermionicOperator`
        The problem Hamiltonian.
    ansatz_function : callable
        A function receiving a :ref:`QuantumVariable`, and an array of real parameters of size ``(n,)``. 
        This function implements the unitary corresponding to one layer of the ansatz.
    num_params : int
        The number of parameters $n$ per layer of the ansatz.
    init_function : callable, optional
        A function receiving a :ref:`QuantumVariable` and preparing the initial state from the $\ket{0}$ state.
        By default, the inital state is the $\ket{0}$ state.
    callback : bool, optional
        If ``True``, intermediate results are stored. The default is ``False``.

    Examples
    --------

    For a quick demonstration, we show how to calculate the ground state energy of the $H_2$ molecule using VQE. 
    The :meth:`electronic_structure_problem <qrisp.vqe.problems.electronic_structure.electronic_structure_problem>` method generates 
    a VQEProblem instance with the Hamiltonian and a chemistry-inspired Qubit Coupled Cluster Single Double `(QCCSD) ansatz <https://arxiv.org/abs/2005.08451>`_.

    ::

        from pyscf import gto
        from qrisp import QuantumVariable
        from qrisp.vqe.problems.electronic_structure import *

        mol = gto.M(
            atom = '''H 0 0 0; H 0 0 0.74''',
            basis = 'sto-3g')

        vqe = electronic_structure_problem(mol)

        energy = vqe.run(lambda : QuantumVariable(4), depth=1, max_iter=50)
        print(energy)
        #Yields -1.8461290172512965

    You can also specify a custom Hamiltonian and a custom hardware-efficient ansatz, as explained `here <https://arxiv.org/abs/2305.07092>`_.

    ::

        from qrisp import *
        from qrisp.operators.qubit import X,Y,Z

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

        energy = vqe.run(lambda : QuantumVariable(4),
                      depth = 1,
                      max_iter=50)
        print(energy)
        # Yields -1.864179046
    
    Note that for comparing to the results in the aforementioned paper, we have to add the nuclear repulsion energy $E_{\text{nuc}}=0.72$ to the calculated electronic energy $E_{\text{el}}$.

    We visualize the optimization process:

    >>> vqe.visualize_energy(exact=True)

    .. figure:: /_static/vqeH2.png
        :alt: VQEH2
        :scale: 80%
        :align: center

    """
    
    def __init__(self, hamiltonian, ansatz_function, num_params, init_function = None, callback=False):
        
        if isinstance(hamiltonian, FermionicOperator):
            hamiltonian = hamiltonian.to_qubit_operator()
        
        hamiltonian = hamiltonian.hermitize()
        hamiltonian = hamiltonian.eliminate_ladder_conjugates()
        hamiltonian = hamiltonian.apply_threshold(0)
        
        self.hamiltonian = hamiltonian
        self.ansatz_function = ansatz_function
        self.num_params = num_params
        self.init_function = init_function

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
        init_function : callable
            The initial state preparation function for the specific VQE problem instance.
        """
        self.init_function = init_function


    def compile_circuit(self, qarg, depth):
        """
        Compiles the circuit that is evaluated by the :meth:`run <qrisp.vqe.VQEProblem.run>` method.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The argument to which the VQE circuit is applied.
        depth : int
            The amount of VQE layers.

        Returns
        -------
        compiled_qc : :ref:`QuantumCircuit`
            The parametrized, compiled quantum circuit without measurements.
        list[sympy.Symbol]
            A list of the parameters that appear in ``compiled_qc``.

        """

        if callable(qarg):
            qarg = qarg()
        
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


    def optimization_routine(self, qarg, depth, mes_kwargs, init_type, init_point, optimizer, options, measurement_data): 
        """
        Wrapper subroutine for the optimization method used in QAOA. The initial values are set and the optimization via ``COBYLA`` is conducted here.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable` or callable
            The argument to which the VQE circuit is applied, 
            or a function returning a :ref:`QuantumVariable` to which the VQE circuit is applied. 
        depth : int
            The amount of VQE layers.
        mes_kwargs : dict
            The keyword arguments for the measurement function.
        init_type : string
            Specifies the way the initial optimization parameters are chosen. 
        init_point : ndarray, shape (n,)
            Specifies the initial optimization parameters. 
        optimizer : str
            Specifies the optimization routine. 
        options : dict
            A dictionary of solver options.
        measurement_data : QubitOperatorMeasurement
            Cached data to accelerate the measurement procedure. Automatically generated by default.

        Returns
        -------
        ndarray, shape (n,)
            The optimized parameters of the problem instance.
        float or jax.Array
            The expectation value of the Hamiltonian for the optimized parameters.    
        """

        if check_for_tracing_mode():

            # Define optimization wrapper function to be minimized using VQE
            def optimization_wrapper(theta, state_prep, mes_kwargs):
                """
                Wrapper function for the optimization method used in VQE.

                This function calculates the expectation value of the Hamiltonian.

                Parameters
                ----------
                theta : jax.Array
                    The array of angle parameters for the VQE circuit.
                state_prep : callable
                    A function returning a QuantumVariable. 
                    The expectation of the Hamiltonian for the state of this QuantumVariable will be measured. 
                    The state preparation function can only take classical values as arguments. 
                    This is because a quantum value would need to be copied for each sampling iteration, which is prohibited by the no-cloning theorem.
                mes_kwargs : dict
                    The keyword arguments for the measurement function.

                Returns
                -------
                jax.Array
                    The expectation value of the Hamiltonian.
                """       
   
                expectation = self.hamiltonian.expectation_value(state_prep,
                                                                **mes_kwargs)(theta)
            
                return expectation

        else:

            # Define optimization wrapper function to be minimized using VQE
            def optimization_wrapper(theta, state_prep, mes_kwargs, compiled_qc, symbols, measurement_data):
                """
                Wrapper function for the optimization method used in VQE.

                This function calculates the expectation value of the Hamiltonian.

                Parameters
                ----------
                theta : list
                    The list of angle parameters for the VQE circuit.
                state_prep : callable
                    A function returning a QuantumVariable. 
                mes_kwargs : dict
                    The keyword arguments for the measurement function.
                compiled_qc : QuantumCircuit
                    The compiled quantum circuit.
                symbols : list
                    The list of symbols used in the quantum circuit.
                measurement_data : QubitOperatorMeasurement
                    Cached data to accelerate the measurement procedure. Automatically generated by default.

                Returns
                -------
                float
                    The expectation value of the Hamiltonian.
                """         

                subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}

                expectation = self.hamiltonian.expectation_value(state_prep, 
                                                                **mes_kwargs,
                                                                precompiled_qc = compiled_qc,
                                                                subs_dic = subs_dic, 
                                                                measurement_data = measurement_data)(theta)
            
                if self.callback:
                    self.optimization_costs.append(expectation)

                return expectation
            

        if init_point is None:
            # Set initial random values for optimization parameters 
            if init_type=='random':

                if check_for_tracing_mode():
                    key = jax.random.key(11)
                    init_point = jax.random.uniform(key=key, shape=(self.num_params*depth,))*jnp.pi/2
                else:
                    init_point = np.random.rand(self.num_params*depth)*np.pi/2

            else:
                raise Exception(f'Parameter initialization method {init_type} is not available.')


        if callable(qarg):
            qarg_prep = qarg
        else:
            template = qarg.template()
            def qarg_prep():
                return template.construct()

        def state_prep(theta):

            qarg = qarg_prep()

            # Prepare the initial state, the default is the \ket{0} state
            if self.init_function is not None:
                self.init_function(qarg)

            for i in range(depth):
                self.ansatz_function(qarg, theta[self.num_params*i : self.num_params*(i+1)+1])

            return qarg
            
        # Perform optimization using specified method
        if check_for_tracing_mode():

            res_sample = jasp_minimize(optimization_wrapper,
                                    init_point, 
                                    method = optimizer,
                                    options = options, 
                                    args = (state_prep, mes_kwargs,))
            
            return res_sample.x, res_sample.fun
        
        else:

            compiled_qc, symbols = self.compile_circuit(qarg() if callable(qarg) else qarg, depth)

            res_sample = minimize(optimization_wrapper,
                                    init_point, 
                                    method = optimizer,
                                    options = options, 
                                    args = (state_prep, mes_kwargs, compiled_qc, symbols, measurement_data,))
            
            return res_sample.x, res_sample.fun


    def run(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random", init_point = None, optimizer = None, options = {}):
        """
        Run VQE for the specific problem instance.
        
        Parameters
        ----------
        qarg : :ref:`QuantumVariable` or callable
            The argument to which the VQE circuit is applied, 
            or a function returning a :ref:`QuantumVariable` to which the VQE circuit is applied. 
        depth : int
            The amount of VQE ansatz layers.
        mes_kwargs : dict, optional
            The keyword arguments for the :meth:`expectation_value <qrisp.operators.qubit.QubitOperator.expectation_value>` function. Default is an empty dictionary.
            By default, the target ``precision`` is set to 0.01. Precision refers to how accurately the Hamiltonian is evaluated.
            The number of shots the backend performs per iteration scales quadratically with the inverse precision.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.
        init_point : ndarray, shape (n,), optional
            Specifies the initial optimization parameters. 
        optimizer : str, optional
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            Available are, e.g., ``COBYLA``, ``COBYQA``, ``Nelder-Mead``. The Default is ``COBYLA``.    
            In tracing mode (i.e. Jasp) Jax-traceable :ref:`optimization routines <optimization_tools>` must be utilized.
            Available are ``SPSA``. The Default is ``SPSA``. 
        options : dict
            A dictionary of solver options.

        Returns
        -------
        float or jax.Array
            The expectation value of the Hamiltonian after applying the optimal VQE circuit to the quantum argument.

        """

        if not "precision" in mes_kwargs:
            mes_kwargs["precision"] = 0.01

        if not "diagonalisation_method" in mes_kwargs:
            mes_kwargs["diagonalisation_method"] = "commuting_qw"

        options["maxiter"] = max_iter

        if check_for_tracing_mode():
            
            if optimizer is None:
                optimizer = "SPSA"

            measurement_data = None

        else: 

            if optimizer is None:
                optimizer = "COBYLA"

            measurement_data = QubitOperatorMeasurement(self.hamiltonian, diagonalisation_method = mes_kwargs["diagonalisation_method"])

        
        opt_theta, opt_res = self.optimization_routine(qarg,
                                                    depth,
                                                    mes_kwargs, 
                                                    init_type, 
                                                    init_point, 
                                                    optimizer,
                                                    options,
                                                    measurement_data = measurement_data)
    
        return opt_res
    

    def train_function(self, qarg, depth, mes_kwargs = {}, max_iter = 50, init_type = "random", init_point = None, optimizer = None, options = {}):
        """
        This function allows for training of a circuit with a given instance of a ``VQEProblem``. It will then return a function that can be applied to a :ref:`QuantumVariable`,
        such that it prepares the ground state of the problem Hamiltonian. The function therefore applies a circuit for the problem instance with optimized parameters.
        
        Parameters
        ----------
        qarg : :ref:`QuantumVariable` or callable
            The argument to which the VQE circuit is applied, 
            or a function returning a :ref:`QuantumVariable` to which the VQE circuit is applied. 
        depth : int
            The amount of VQE ansatz layers.
        mes_kwargs : dict, optional
            The keyword arguments for the :meth:`expectation_value <qrisp.operators.qubit.QubitOperator.expectation_value>` function. Default is an empty dictionary.
            By default, the target ``precision`` is set to 0.01. Precision refers to how accurately the Hamiltonian is evaluated.
            The number of shots the backend performs per iteration scales quadratically with the inverse precision.
        max_iter : int, optional
            The maximum number of iterations for the optimization method. Default is 50.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.
        init_point : ndarray, shape (n,), optional
            Specifies the initial optimization parameters. 
        optimizer : str, optional
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            Available are, e.g., ``COBYLA``, ``COBYQA``, ``Nelder-Mead``. The Default is ``COBYLA``.    
            In tracing mode (i.e. Jasp) Jax-traceable :ref:`optimization routines <optimization_tools>` must be utilized.
            The Default is ``SPSA``. 
        options : dict
            A dictionary of solver options.

        Returns
        -------
        callable
            A function that can be applied to a :ref:`QuantumVariable`, with optimized parameters for the problem instance. 
            The :ref:`QuantumVariable` then represents the ground state of the problem Hamiltonian.

        """

        if not "precision" in mes_kwargs:
            mes_kwargs["precision"] = 0.01

        if not "diagonalisation_method" in mes_kwargs:
            mes_kwargs["diagonalisation_method"] = "commuting_qw"

        options["maxiter"] = max_iter

        if check_for_tracing_mode():
            
            if optimizer is None:
                optimizer = "SPSA"

            measurement_data = None

        else: 

            if optimizer is None:
                optimizer = "COBYLA"

            measurement_data = QubitOperatorMeasurement(self.hamiltonian, diagonalisation_method = mes_kwargs["diagonalisation_method"])

        
        opt_theta, opt_res = self.optimization_routine(qarg,
                                                    depth,
                                                    mes_kwargs, 
                                                    init_type, 
                                                    init_point, 
                                                    optimizer,
                                                    options,
                                                    measurement_data = measurement_data)
            
        def circuit_generator(qarg):
                
                if self.init_function is not None:
                    self.init_function(qarg)

                for i in range(depth):
                    self.ansatz_function(qarg, opt_theta[self.num_params*i : self.num_params*(i+1)+1])

        return circuit_generator
            

    def benchmark(self, qarg, depth_range, precision_range, iter_range, optimal_energy, repetitions = 1, mes_kwargs = {}, init_type = "random", optimizer = None, options = {}):
        """
        This method enables convenient data collection regarding performance of the implementation.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable` or callable
            The argument to which the VQE circuit is applied, 
            or a function returning a :ref:`QuantumVariable` to which the VQE circuit is applied. 
        depth_range : list[int]
            A list of integers indicating, which depth parameters should be explored. Depth means the amount of VQE ansatz layers.
        precision_range : list[float]
            A list of floats indicating, which precision parameters should be explored. Precision refers to how accurately the Hamiltonian is evaluated.
            The number of shots the backend performs per iteration scales quadratically with the inverse precision.
        iter_range : list[int]
            A list of integers indicating, what iterations parameter should be explored. Iterations means the amount of backend calls, the optimizer is allowed to do.
        optimal_energy : float
            The exact ground state energy of the problem Hamiltonian. 
        repetitions : int, optional
            The amount of repetitions, each parameter constellation should go though. Can be used to get a better statistical significance. The default is 1.
        mes_kwargs : dict, optional
            The keyword arguments for the :meth:`expectation_value <qrisp.operators.qubit.QubitOperator.expectation_value>` function. Default is an empty dictionary.
        init_type : str, optional
            Specifies the way the initial optimization parameters are chosen. Available is ``random``. 
            The default is ``random``: Parameters are initialized uniformly at random in the interval $[0,\pi/2)]$.
        optimizer : str, optional
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_. 
            Available are, e.g., ``COBYLA``, ``COBYQA``, ``Nelder-Mead``. The Default is ``COBYLA``.
        options : dict
            A dictionary of solver options.

        Returns
        -------
        :ref:`VQEBenchmark`
            The results of the benchmark.
            
        Examples
        --------
        
        We create a Heisenberg problem instance and benchmark several parameters:
        
        ::

            from qrisp import QuantumVariable
            from qrisp.vqe.problems.heisenberg import *
            from networkx import Graph

            G = Graph()
            G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])

            vqe = heisenberg_problem(G,1,0)
            H = create_heisenberg_hamiltonian(G,1,0)

            benchmark_data = vqe.benchmark(QuantumVariable(5),
                                depth_range = [1,2,3],
                                precision_range = [0.02,0.01],
                                iter_range = [25,50],
                                optimal_energy = H.ground_state_energy(),
                                repetitions = 2
                                )
        
        We can investigate the data by calling ``visualize``:
        
        ::
            
            benchmark_data.visualize()
        
        .. image:: vqe_benchmark_plot.png
            
        The :ref:`VQEBenchmark` class contains a variety of methods to help 
        you drawing conclusions from the collected data. Make sure to check them out!

        """
    
        data_dict = {"layer_depth" : [],
                     "circuit_depth" : [],
                     "qubit_amount" : [],
                     "precision" : [],
                     "iterations" : [],
                     "runtime" : [],
                     "energy" : []
                     }
        
        for p in depth_range:
            for s in precision_range:
                for it in iter_range:
                    for k in range(repetitions):
                            
                        start_time = time.time()
                        
                        temp_mes_kwargs = dict(mes_kwargs)
                        temp_mes_kwargs["precision"] = s

                        energy = self.run(qarg, depth = p, max_iter = it, mes_kwargs = temp_mes_kwargs, init_type = init_type, optimizer = optimizer, options = options)

                        final_time = time.time() - start_time
                    
                        compiled_qc, _ = self.compile_circuit(qarg() if callable(qarg) else qarg, depth = p)
                        
                        data_dict["layer_depth"].append(p)
                        data_dict["circuit_depth"].append(compiled_qc.depth())
                        data_dict["qubit_amount"].append(compiled_qc.num_qubits())
                        data_dict["precision"].append(s)
                        data_dict["iterations"].append(it)
                        data_dict["energy"].append(energy)
                        data_dict["runtime"].append(final_time)
                        
                        
        return VQEBenchmark(data_dict, optimal_energy, self.hamiltonian)
    

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
        plt.legend(fontsize=12, labelcolor="#444444")
        plt.tick_params(axis='both', labelsize=12)
        plt.grid()

        plt.show()
