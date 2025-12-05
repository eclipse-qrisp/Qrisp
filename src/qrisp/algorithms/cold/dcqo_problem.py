"""
********************************************************************************
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
********************************************************************************
"""

import numpy as np
from scipy.optimize import minimize
import sympy as sp
from qrisp.algorithms.cold.crab import CRABObjective
from qrisp import h
# from qrisp.algorithms.cold.AGP_param_opt import solve_params_at_lambda, build_Hg

class DCQOProblem:
    """
    General structure to formulate Digitized Counterdiabaric Quantum Optimization problems.
    This class is used to solve DCQO problems with the algorithms COLD, LCD and/or a nonlocal AGP,
    depending on the parameters given by the user.

    Parameters
    ----------
    qarg_prep : callable
        A function receiving a :ref:`QuantumVariable` for preparing the inital state.
        By default, the uniform superposition state $\ket{+}^n$ is prepared.
    sympy_lambda : callable
        A function $\lambda(t, T)$ mapping $t \in$ [0, T] to $\lambda \in$ [0, 1]. This function needs to return
        a `sympy <https://docs.sympy.org/>`_ expression with $t$ and $T$ as `sympy.Symbols <https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol>`_.
    alpha : callable
        The parameters for the adiabatic gauge potential (AGP). If the COLD method is being used,
        alpha must depend on the optimization pulses in ``H_control``.
    H_init : :ref:`QubitOperator`
        Hamiltonian, the system is at the time t=0.
    H_prob : :ref:`QubitOperator`
        Hamiltonian, the system evolves to for t=T.
    A_lam : :ref:`QubitOperator`
        Operator holding an appoximation for the adiabatic gauge potential (AGP).
    H_control : :ref:`QubitOperator`, optional
        Hamiltonian specifying the control pulses for the COLD method. If not given, the LCD method is used automatically.
    callback : bool, optional
        If ``True``, intermediate results are stored. The default is ``False``.
    """

    def __init__(
        self, 
        sympy_lambda, 
        sympy_g,
        alpha,
        H_init,
        H_prob,
        A_lam,
        H_control=None,
        qarg_prep=None, 
        callback=False

    ):
        
        self.sympy_lambda = sympy_lambda
        self.sympy_g = sympy_g
        self.alpha = alpha
        self.H_init = H_init
        self.H_prob = H_prob
        self.A_lam = A_lam
        self.H_control = H_control
        self.qarg_prep = qarg_prep
        
        # Parameters for callback
        self.callback = callback
        self.optimization_params = []
        self.optimization_costs = []

    def set_callback(self):
        """
        Sets ``callback=True`` for saving intermediate results.

        """

        self.callback = True

    def __precompute_timegrid(self, N_steps, T, method):
        """
        Compute lambda(t, T) and the time-derivative lambdadot(t, T) 
        for each timestep.

        Parameters
        ----------
        N_steps : int
            Number of timesteps.
        T : float
            Evolution time for the simulation.
        
        Returns
        -------
        lam : array
            The parametrized timefunction, specified by ``sympy_lambda`` for t in [0, T].
        lamdot : array
            The time derivative of ``sympy_lambda`` for t in [0, T].
        """

        # Sympy symbols for t and T
        t_sym, T_sym = sp.symbols('t T', real=True)
        # Array for t values
        dt = T/N_steps
        t_list = np.linspace(dt, T, N_steps)

        # Sympy functions for lam and lamdot
        lam_func = sp.lambdify((t_sym, T_sym), self.sympy_lambda(), 'numpy')
        lamdot_expr = sp.diff(self.sympy_lambda(), t_sym)
        lamdot_func = sp.lambdify((t_sym, T_sym), lamdot_expr, 'numpy')

        # Compute functions of values t_list
        lam = lam_func(t_list, T)
        lamdot = lamdot_func(t_list, T)
        # Convert lamdot to a constant list to make it subscriptable by timestep
        if isinstance(lamdot, float): lamdot = [lamdot] * N_steps

        self.lam = lam
        self.lamdot = lamdot

        # Functions for t = g(lam) and derivative (later needed for opt pulses)
        # must only be calculated for COLD, not for LCD
        if method == 'COLD':
            lam_sym = sp.Symbol("lam")
            g_func = sp.lambdify((lam_sym, T_sym), self.sympy_g(), 'numpy')
            g_deriv_expr = sp.diff(self.sympy_g(), lam_sym)
            g_deriv_func = sp.lambdify((lam_sym, T_sym), g_deriv_expr, 'numpy')
            g_deriv = g_deriv_func(self.lam, T)

            self.g = g_func(self.lam, T)
            self.g_deriv = g_deriv


    def apply_lcd_hamiltonian(self, qarg, N_steps, T):
        """
        Simulate the local counterdiabatic driving (LCD) Hamiltonian on a 
        quantum argument via trotterization. The LCD Hamiltonian consists 
        of the system Hamiltonian and the adiabatic gauge potential (AGP).

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The quantum argument to which the quantum circuit is applied.
        N_steps : int
            Number of steps in which the timefunction ``lambda'' is split up.
        T : float
            Evolution time for the simulation.
        """

        self.qarg_prep(qarg)
        dt = T / N_steps

        # Compute time-function lamda(t, T) and the derivative lamdot(t, T)
        self.__precompute_timegrid(N_steps, T, 'LCD')
        
        # Apply hamiltonian to qarg for each timestep
        for s in range(N_steps):

            # Get alpha for the timestep
            alph = self.alpha(self.lam[s])

            # H_0 contribution scaled by dt
            H_step = (1-self.lam[s]) * self.H_init + self.lam[s] * self.H_prob

            # AGP contribution scaled by dt* lambda_dot(t)
            H_step += self.lamdot[s] * self.A_lam(alph)

            # Get unitary from trotterization and apply to qarg
            U = H_step.trotterization()
            U(qarg, t=dt)

    def apply_cold_hamiltonian(self, qarg, N_steps, T, opt_params, CRAB=False):
        """
        Simulate counterdiabatic optimized local driving (COLD) Hamiltonian 
        on a quantumvariable via trotterization. The COLD Hamiltonian consists 
        of the system Hamiltonian, the adiabatic gauge potential (AGP) and
        local pulses (given by ``H_control``) with optimized parameters.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The quantum argument to which the quantum circuit is applied.
        N_steps : int
            Number of steps in which the timefunction ``lambda'' is split up.
        T : float
            Evolution time for the simulation.
        opt_params : list
            Either the optimized parameters or the corresponding `sympy.Symbols <https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol>`_.
        CRAB : bool, optional
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        """
        
        def precompute_opt_pulses(t_list, N_opt):
            
            # Precompute f (sine) and f_deriv (cosine) for each timestep
            if CRAB:
                # Matrices and arrays must be sympy objects
                sin_matrix = sp.MutableDenseMatrix.zeros(N_steps, N_opt)
                cos_matrix = sp.MutableDenseMatrix.zeros(N_steps, N_opt)
                t_list = sp.Array(t_list)

                # Add symbols for random parameters to be called in optimizaton
                r_params = [sp.Symbol("r_"+str(i)) for i in range(N_opt)]
                for k in range(N_opt):
                    sin_matrix[:, k] = sp.Matrix(t_list.applyfunc(lambda t: sp.sin(sp.pi * (k+1+r_params[k]) * t / T)))
                    cos_matrix[:, k] = sp.Matrix(t_list.applyfunc(lambda t: (sp.pi/T * (k+1+r_params[k])) * sp.cos(sp.pi * (k+1+r_params[k]) * t / T)))

            # Use numpy if not random params are used
            else:
                sin_matrix = np.zeros((N_steps, N_opt))
                cos_matrix = np.zeros((N_steps, N_opt))

                for k in range(N_opt):
                    sin_matrix[:, k] = np.sin(np.pi * (k+1) * t_list/T)
                    cos_matrix[:, k] = (np.pi * (k+1)) * np.cos(np.pi * (k+1) * self.g) * self.g_deriv

            return sin_matrix, cos_matrix
        
        # Initialize qarg
        self.qarg_prep(qarg)

        # Compute time-function lamda(t, T) and the derivative lamdot(t, T)
        self.__precompute_timegrid(N_steps, T, 'COLD')

        # Precompute opt pulses
        dt = T / N_steps
        t_list = np.linspace(dt, T, int(N_steps))
        sin_matrix, cos_matrix = precompute_opt_pulses(t_list, N_opt=len(opt_params))

        # Transform opt_params to sympy to work with symbols for random values
        if CRAB:
            beta = sp.Matrix(opt_params)
        else:
            beta = opt_params

        # Apply hamiltonian to qarg for each timestep
        for s in range(N_steps):

            # Get alpha, f and f_deriv for the timestep
            f = sin_matrix[s, :] @ beta
            f_deriv = cos_matrix[s, :] @ beta
            alph = self.alpha(self.lam[s], f, f_deriv)

            # H_0 contribution scaled by dt
            H_step = (1-self.lam[s]) * self.H_init + self.lam[s] * self.H_prob

            # AGP contribution scaled by dt* lambda_dot(t)
            H_step += self.lamdot[s] * self.A_lam(alph)

            # Control pulse contribution 
            if CRAB:
                H_step += sum(f[i, 0] for i in range(f.rows)) * self.H_control
            else:
                H_step += f * self.H_control

            # Get unitary from trotterization and apply to qarg
            U = H_step.trotterization()
            U(qarg, t=dt)  

    
    def compile_U_cold(self, qarg, N_opt, N_steps, T, CRAB=False):
        """
        Compiles the circuit that is created by the :meth:`apply_cold_hamiltonian <qrisp.cold.DCQOProblem.apply_cold_hamiltonian>` method.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The argument to which the COLD circuit is applied.
        N_opt : int
            Number of optimization parameters in ``H_control``.
        N_steps : int
            Number of timesteps.
        T : float
            Evolution time for the simulation.
        CRAB : bool, optional
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        
        Returns
        -------
        compiled_qc : :ref:`QuantumCircuit`
            The compiled quantum circuit.

        """

        temp = list(qarg.qs.data)
        # Initzialize parameters as symbols
        params = [sp.Symbol("par_"+str(i)) for i in range(N_opt)]
        self.apply_cold_hamiltonian(qarg, N_steps, T, params, CRAB=CRAB)
        intended_measurements = list(qarg)
        compiled_qc = qarg.qs.compile(intended_measurements=intended_measurements)
        qarg.qs.data = temp

        return compiled_qc


    def optimization_routine(self, qarg, N_opt, qc, CRAB, optimizer, options
                             ): 
        """
        Subroutine for the optimization method used in COLD. 
        The initial values are set and the optimization via is conducted here.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The argument to which the H_prob circuit is applied.
        N_opt : int
            Number of optimization parameters in ``H_control``.
        qc : :ref:`QuantumCircuit`
            The COLD circuit that is applied before measuring the qarg.
        CRAB : bool, optional
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        optimizer : str
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
        options : dict
            A dictionary of solver options.

        Returns
        -------
        res.x: array
            The optimized parameters of the problem instance.
        """

        def objective(params):
            # Objective function to be minimized: 
            # Expectation value of the QUBO Hamiltonian

            # Dict to assign the optimization parameters
            subs_dic = {sp.Symbol("par_"+str(i)): params[i] for i in range(len(params))}

            cost = self.H_prob.expectation_value(qarg, 
                                                 compile=False, 
                                                 subs_dic=subs_dic, 
                                                 precompiled_qc=qc)()
            
            if self.callback:
                self.optimization_costs.append(cost)
            
            return cost
        
        init_point = np.random.rand(N_opt) * np.pi/2

        # Create CRAB objective to make sure randomization is applied at each optimization iteration
        if CRAB:
            objective = CRABObjective(self.H_prob, qarg, qc, N_opt)
        # Otherwise objective from above is used
        else:
            objective = objective

        res = minimize(objective,
                        init_point,
                        method=optimizer,
                        options=options
                        )
        
        return res.x

    def run(
        self, 
        qarg, 
        N_steps, 
        T, 
        N_opt=None, 
        CRAB=False, 
        optimizer="Powell",
        options={}
    ):
        """
        Run the specific DCQO problem instance with given quantum arguments, number of timesteps and
        evolution time.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The argument to which the DCQO circuit is applied.
        N_steps : int
            Number of time steps in which the function ``lambda'' is split up.
            Or: Number of time steps for the simulation.
        T : float
            Evolution time for the simulation.
        N_opt : int
            Number of optimization parameters in ``H_control``.
        CRAB : bool
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        optimizer : str, optional
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            We set the default to ``Powell``.
        options : dict
            A dictionary of solver options.

        Returns
        -------
        res_dict : dict
            The optimal result after running DCQO problem for a specific problem instance. It contains the measurement results after applying the optimal DCQO circuit to the quantum argument.

        """

        # If no prep for qarg is specified, use uniform superposition state
        if self.qarg_prep is None:
            def qarg_prep(q):
                return h(q)
            self.qarg_prep = qarg_prep

        # Run COLD routine if H_control is given
        if self.H_control is not None:
            qarg1, qarg2 = qarg.duplicate(), qarg.duplicate()

            # Compile COLD routine into a circuit
            U_circuit = self.compile_U_cold(qarg1, N_opt, N_steps, T, CRAB)

            # Find optimal params for control pulse
            opt_params = self.optimization_routine(qarg2, N_opt, U_circuit, CRAB, optimizer, options)

            # Apply hamiltonian with optimal parameters
            # Here we do not want the randomized parameters to be included -> CRAB=False
            self.apply_cold_hamiltonian(qarg, N_steps, T, opt_params, CRAB=False)

        # Otherwise run LCD routine
        else:
            self.apply_lcd_hamiltonian(qarg, N_steps, T)

        # Measure qarg
        res_dict = qarg.get_measurement()

        return res_dict
