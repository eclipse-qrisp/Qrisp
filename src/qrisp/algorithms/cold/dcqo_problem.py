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
from scipy.optimize import minimize, Bounds
import sympy as sp
from qrisp.algorithms.cold.AGP_params import solve_alpha_gamma_chi
from qrisp import h, z
from qrisp.operators import QubitOperator


class DCQOProblem:
    """
    General structure to formulate Digitized Counterdiabatic Quantum Optimization problems.
    This class is used to solve |dcqo_link|
    problems with the algorithms `COLD <https://doi.org/10.1103/PRXQuantum.4.010312>`_ 
    (counterdiabatic optimized local driving) or LCD (local counterdiabatic driving).
    To run the COLD algorithm on the problem, you need to specify the control Hamiltonian
    ``H_control`` and the inverse scheduling function ``g_func``. These are not needed for
    the LCD algorithm. To learn more about counterdiabatic driving, make sure to check out the `tutorial <https://www.qrisp.eu/general/tutorial/CD.html>`_.

    Parameters
    ----------
    Q : np.array
        The QUBO matrix.
    H_init : :ref:`QubitOperator`
        Hamiltonian, the system is at the time t=0.
    H_prob : :ref:`QubitOperator`
        Hamiltonian, the system evolves to for t=T.
    A_lam : :ref:`QubitOperator`
        Operator holding an appoximation for the adiabatic gauge potential (AGP).
    agp_coeffs : callable
        The parameters for the adiabatic gauge potential (AGP). If the COLD method is being used,
        they must depend on the optimization pulses in ``H_control``.
    lam_func : callable
        A function $\lambda(t, T)$ mapping $t \in [0, T]$ to $\lambda \in [0, 1]$. This function needs to return
        a `sympy <https://docs.sympy.org/>`_ expression with $t$ and $T$ as `sympy.Symbols <https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol>`_.
    g_func : callable, optional
        The inverse function of $\lambda(t, T)$. This function needs to return a `sympy <https://docs.sympy.org/>`_ expression 
        with $\lambda$ and $T$ as `sympy.Symbols <https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol>`_.
        Only needed for the COLD algorithm.
    H_control : :ref:`QubitOperator`, optional
        Hamiltonian specifying the control pulses for the COLD method. If not given, the LCD method is used automatically.
    qarg_prep : callable, optional
        A function receiving a :ref:`QuantumVariable` for preparing the inital state.
        By default, the groundstate of the x-operator $\ket{-}^n$ is prepared.


    Examples
    --------
    For a quick demonstration we build a DCQO problem instance for a 4x4 QUBO. We choose a first order AGP ansatz with uniform coefficients and solve it with LCD.

    ::

        import numpy as np
        import sympy as sp
        from qrisp.operators.qubit import X, Y, Z
        from qrisp.algorithms.cold import DCQOProblem
        from qrisp import QuantumVariable

        Q = np.array([
            [-1.2,  0.40, 0.0,  0.0],
            [ 0.40,  0.30, 0.20, 0.0],
            [ 0.0,   0.20,-1.1,  0.30],
            [ 0.0,   0.0,  0.30,-0.80]
        ])
        N = Q.shape[0]

        # Define QUBO problem hamiltonian
        h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
        J = 0.5 * Q

        H_init = 1 * sum([X(i) for i in range(N)])

        H_prob = (sum([sum([J[i][j]*Z(i)*Z(j) for j in range(i)]) for i in range(N)]) 
                + sum([h[i]*Z(i) for i in range(N)]))

        # Create AGP
        A_lam = sum([Y(i) for i in range(N)]) # uniform

        # Function for uniform AGP coefficients
        def alpha(lam):
            A = lam * h 
            B = 1 - lam
            nom = np.sum(A + 4*B*h)
            denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
            alph = nom/denom
            alph = [alph]*N
            return alph

        # Simple scheduling function 0 -> 1
        def lam():
            t, T = sp.symbols("t T", real=True)
            lam_expr = t/T
            return lam_expr

        # Create problem instance
        lcd_problem = DCQOProblem(Q, H_init, H_prob, A_lam, alpha, lam)

        # Run problem with LCD algorithm
        qarg = QuantumVariable(N)
        res = lcd_problem.run(qarg, N_steps=4, T=12, method="LCD")
        print(res)

    ::

        {'1011': [0.40630593694063055, np.float64(-2.5)], '1111': [0.16247837521624783, np.float64(-0.9999999999999999)], '0111': [0.13156868431315685, np.float64(-0.6000000000000001)], '1000': [0.06881931180688193, np.float64(-1.2)], '0011': [0.05949940500594993, np.float64(-1.3)], '1010': [0.04499955000449995, np.float64(-2.3)], '1101': [0.04084959150408495, np.float64(-0.9)], '0110': [0.019769802301976978, np.float64(-0.40000000000000013)], '1100': [0.01815981840181598, np.float64(-0.09999999999999998)], '0100': [0.013679863201367985, np.float64(0.3)], '0001': [0.010399896001039988, np.float64(-0.8)], '0000': [0.007659923400765992, np.float64(0.0)], '1110': [0.006329936700632993, np.float64(-0.7999999999999999)], '0101': [0.0052899471005289946, np.float64(-0.5)], '1001': [0.0024299757002429973, np.float64(-2.0)], '0010': [0.0017599824001759982, np.float64(-1.1)]}

    We get a dictionary where the key is the quantum state and the values are lists of [probability, cost]. 
    So our most likely result is '1011' with probabilty 0.4 and the QUBO cost $x^T Q x = -2.5$.

    .. |dcqo_link| raw:: html
    
       <a href="https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L042030" target="_blank">DCQO</a>
    """

    def __init__(
        self, 
        Q,
        H_init,
        H_prob,
        A_lam,
        agp_coeffs,
        lam_func, 
        g_func = None,
        H_control=None,
        qarg_prep=None, 

    ):
        
        # Scheduling function
        self.lam_func = lam_func
        self.g_func = g_func

        # Operators
        self.agp_coeffs = agp_coeffs
        self.H_init = H_init
        self.H_prob = H_prob
        self.A_lam = A_lam
        self.H_control = H_control
        self.qarg_prep = qarg_prep

        # Qubo characteristics
        self.Q = Q
        self.J = 0.5 * Q
        self.h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)

        # Placeholder for the _precompute_timegrid function
        self.lam = None
        self.lamdot = None

    def _precompute_timegrid(self, N_steps, T, method):
        """
        Compute lambda(t, T) and the time-derivative lambdadot(t, T) 
        for each timestep.

        Parameters
        ----------
        N_steps : int
            Number of timesteps.
        T : float
            Evolution time for the simulation.
        method : str
            The method to solve the QUBO with (either ``LCD`` or ``COLD``).
        
        Returns
        -------
        lam : array
            The parametrized timefunction, specified by ``lam_func`` for t in [0, T].
        lamdot : array
            The time derivative of ``lam_func`` for t in [0, T].
        """

        # Sympy symbols for t and T
        t_sym, T_sym = sp.symbols('t T', real=True)
        # Array for t values
        dt = T/N_steps
        t_list = np.linspace(dt, T, N_steps)

        # Sympy functions for lam and lamdot
        lam_func = sp.lambdify((t_sym, T_sym), self.lam_func(), 'numpy')
        lamdot_expr = sp.diff(self.lam_func(), t_sym)
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
            g_func = sp.lambdify((lam_sym, T_sym), self.g_func(), 'numpy')
            g_deriv_expr = sp.diff(self.g_func(), lam_sym)
            g_deriv_func = sp.lambdify((lam_sym, T_sym), g_deriv_expr, 'numpy')
            g_deriv = g_deriv_func(self.lam, T)

            self.g = g_func(self.lam, T)
            self.g_deriv = g_deriv

    def _precompute_opt_pulses(self, N_steps, T, t_list, N_opt, CRAB=False):
        """
        Precompute optimization pulses for COLD routine that will be scaled by optimized paramters.
        
        Parameters
        ----------
        N_steps : int
            Number of timesteps.
        T : float
            Evolution time for the simulation.
        t_list : list
            Time values for each step.
        N_opt : int
            Number of optimization parameters in ``H_control``.
        CRAB : bool
            If ``True``, the CRAB optimization method is being used. The default is ``False``.

        Returns
        -------
        sin_matrix : 
            Numpy or sympy (if CRAB) array holding the opt pulse for each timestep.
        cos_matrix :
            Numpy or sympy (if CRAB) array holding the derivative of the opt pulse for each timestep.
        """
            
        # Precompute f (sine) and f_deriv (cosine) for each timestep as numpy arrays
        sin_matrix = np.zeros((N_steps, N_opt))
        cos_matrix = np.zeros((N_steps, N_opt))
        
        if CRAB:
            # Random CRAB parameters
            r_params = np.random.uniform(-0.5, 0.5, N_opt)
        else:
            # Otherwise add nothing
            r_params = np.zeros(N_opt)

        for k in range(N_opt):
            sin_matrix[:, k] = np.sin(np.pi * (k+1+r_params[k]) * t_list/T)
            cos_matrix[:, k] = (np.pi * (k+1+r_params[k])) * np.cos(np.pi * (k+1+r_params[k]) * self.g) * self.g_deriv

        return sin_matrix, cos_matrix

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
            Number of steps in which the timefunction ``lambda`` is split up.
        T : float
            Evolution time for the simulation.
        """

        self.qarg_prep(qarg)
        dt = T / N_steps

        # Compute time-function lamda(t, T) and the derivative lamdot(t, T)
        self._precompute_timegrid(N_steps, T, 'LCD')

        # Trotterize Hamiltonian in different parts with each one needing different coefficients
        U1 = self.H_init.trotterization()
        U2 = self.H_prob.trotterization()
        if isinstance(self.A_lam, QubitOperator):
            # Uniform AGP coefficients
            U3 = self.A_lam.trotterization()
        else:
            # Non-uniform AGP coefficients
            U3 = [A_lam.trotterization() for A_lam in self.A_lam]
        
        # Apply hamiltonian to qarg for each timestep
        for s in range(N_steps):

            # Get alpha for the timestep
            coeffs = self.agp_coeffs(self.lam[s])

            # H_0 contribution scaled by dt
            U1(qarg, t=dt*(1-self.lam[s]))
            U2(qarg, t=dt*self.lam[s])

            # AGP contribution scaled by dt* lambda_dot(t)
            if isinstance(U3, list):
                for idx, U in enumerate(U3):
                    U(qarg, t=dt*self.lamdot[s]*coeffs[idx])
            else:
                U3(qarg, t=dt*self.lamdot[s]*coeffs[0])


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
            Number of steps in which the timefunction ``lambda`` is split up.
        T : float
            Evolution time for the simulation.
        opt_params : list
            Either the optimized parameters or the corresponding `sympy.Symbols <https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol>`_.
        CRAB : bool, optional
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        """
        
        # Initialize qarg
        self.qarg_prep(qarg)

        # Compute time-function lamda(t, T) and the derivative lamdot(t, T)
        self._precompute_timegrid(N_steps, T, 'COLD')

        # Precompute opt pulses
        dt = T/N_steps
        t_list = np.linspace(dt, T, int(N_steps))
        sin_matrix, cos_matrix = self._precompute_opt_pulses(N_steps, T, t_list, N_opt=len(opt_params), CRAB=CRAB)
        beta = opt_params

        # Trotterize Hamiltonian in different parts with each one needing different coefficients
        U1 = self.H_init.trotterization()
        U2 = self.H_prob.trotterization()
        if isinstance(self.A_lam, QubitOperator):
            # Uniform AGP coefficients
            U3 = self.A_lam.trotterization()
        else:
            # Non-uniform AGP coefficients
            U3 = [A_lam.trotterization() for A_lam in self.A_lam]
        U4 = self.H_control.trotterization()

        # Apply hamiltonian to qarg for each timestep
        for s in range(N_steps):

            # Get alpha, f and f_deriv for the timestep
            f = sin_matrix[s, :] @ beta
            f_deriv = cos_matrix[s, :] @ beta
            alpha = self.agp_coeffs(self.lam[s], f, f_deriv)

            # H_init contribution scaled by dt*(1-lam)
            U1(qarg, t=dt*(1-self.lam[s]))
            # H_prob contribution scaled by dt*lam
            U2(qarg, t=dt*(self.lam[s]))

            # AGP contribution scaled by dt*lambda_dot(t)*alpha
            if isinstance(U3, list):
                # Non-uniform alpha
                for idx, U in enumerate(U3):
                    U(qarg, t=dt*(self.lamdot[s]*alpha[idx]))
            else:
                # Uniform alpha
                U3(qarg, t=dt*(self.lamdot[s]*alpha[0]))

            # Control pulse contribution scaled by opt parameters f
            U4(qarg, t=dt*f)

    
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


    def optimization_routine(self, qarg, N_opt, N_steps, T, qc, CRAB, optimizer, 
                             options, objective, bounds
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
        N_steps : int
            Number of time steps for the simulation.
        T : float
            Evolution time for the simulation.
        qc : :ref:`QuantumCircuit`
            The COLD circuit that is applied before measuring the qarg.
        CRAB : bool, optional
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        optimizer : str
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
        options : dict
            A dictionary of solver options.
        objective : str
            The objective function to be minimized (``exp_value``, ``agp_coeff_magnitude``, ``agp_coeff_amplitude``). Default is ``exp_value``.
        bounds : tuple
            The parameter bounds for the optimizer. Default is (-2, 2).

        Returns
        -------
        res.x: array
            The optimized parameters of the problem instance.
        """
        
        # Different objective functions: exp_value, agp coeffs magnitude, agp coeffs amplitude

        # Expectation value of the QUBO Hamiltonian
        def objective_exp(params, CRAB):
            # Dict to assign the optimization parameters
            subs_dic = {sp.Symbol("par_"+str(i)): params[i] for i in range(len(params))}

            cost = self.H_prob.expectation_value(qarg, 
                                                 compile=False, 
                                                 subs_dic=subs_dic, 
                                                 precompiled_qc=qc)()
            
            return cost
        
        # Magnitude of the AGP coefficients (coeffs are treated as uniform for simplification)
        # (sum of absolute values for each timestep)
        def objective_mag(params, CRAB):
            # Precompute opt pulses to be multiplied with opt params
            t_list = np.linspace(T/N_steps, T, int(N_steps))
            sin_matrix, cos_matrix = self._precompute_opt_pulses(N_steps, T, t_list, N_opt=len(params), CRAB=CRAB)
            magnitude = 0

            # Iterate through lambda(t)
            for s in range(N_steps):
                # Get alpha, f and f_deriv for the timestep
                f = sin_matrix[s, :] @ params
                f_deriv = cos_matrix[s, :] @ params
                alpha, gamma, chi = solve_alpha_gamma_chi(self.h, self.J, self.lam[s], f, f_deriv, uniform=True)
                magnitude += (np.abs(gamma[0]) + np.abs(chi[0]) + np.abs(alpha[0]))

            return magnitude
        
        init_point = np.random.rand(N_opt) * np.pi/2

        # Define objective function
        if objective == "exp_value":
            objective = objective_exp

        elif objective == "agp_coeff_magnitude":
            objective = objective_mag

        else:
            raise ValueError("{objective} is not a valid option as objective.")

        
        res = minimize(objective,
                        init_point,
                        method=optimizer,
                        options=options,
                        bounds=Bounds(*bounds),
                        args=(CRAB)
                        )
        
        return res.x, objective(res.x, CRAB)
    
    def QUBO_cost(self, res):
        """
        Returns the cost y = x^T Q x for a given binary array x.
        
        Parameters
        ----------
        res : np.array
            The array to calculate the cost for.
        
        Returns
        -------
        cost : float
            The QUBO cost.
        
        """
        cost = res @ self.Q @ res
        return cost


    def run(
        self, 
        qarg, 
        N_steps, 
        T,
        method,
        N_opt=None, 
        CRAB=False, 
        optimizer="COBYQA",
        objective="agp_coeff_magnitude",
        bounds=(),
        options={},
        mes_kwargs={}
    ):
        """
        Run the specific DCQO problem instance with given quantum arguments, number of timesteps and
        evolution time.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The argument to which the DCQO circuit is applied.
        N_steps : int
            Number of time steps for the simulation.
        T : float
            Evolution time for the simulation.
        method : str
            Method to solve the QUBO with. Either ``LCD`` or ``COLD``.
        N_opt : int
            Number of optimization parameters in ``H_control``.
        CRAB : bool
            If ``True``, the CRAB optimization method is being used. The default is ``False``.
        optimizer : str, optional
            Specifies the `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            We set the default to ``Powell``.
        options : dict
            A dictionary of solver options.
        objective : str
            The objective function to be minimized (``exp_value``, ``agp_coeff_magnitude``). Default is ``agp_coeff_magnitude``.
        bounds : tuple
            The parameter bounds for the optimizer. Default is (-2, 2).
        options : dict
            Additional options for the Scipy solver.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary.
        backend : :ref:`BackendClient`, optional
            The backend to be used for the quantum simulation.
            By default, the Qrisp simulator is used.
        shots: : int
            The number of shots. The default is 5000.

        Returns
        -------
        res_dict : dict
            The optimal result after running DCQO problem for a specific problem instance. It contains the measurement results after applying the optimal DCQO circuit to the quantum argument.

        """

        # If no prep for qarg is specified, use uniform superposition state
        if self.qarg_prep is None:
            def qarg_prep(q):
                h(q)
                z(q)
                return q
            self.qarg_prep = qarg_prep

        # Run COLD routine
        if method == 'COLD':
            qarg1, qarg2 = qarg.duplicate(), qarg.duplicate()

            # If we optimize the Hamiltonian expectation value,
            # compile COLD routine into a circuit for the optimization
            if objective == 'exp_value':
                U_circuit = self.compile_U_cold(qarg1, N_opt, N_steps, T, CRAB)
            else:
                self._precompute_timegrid(N_steps, T, 'COLD')
                U_circuit = None

            # Find optimal params for control pulse
            opt_params, cost = None, None
            # Do optimization 3 times and choose best result
            for i in range(3):
                opt_params_temp, cost_temp = self.optimization_routine(qarg2, N_opt, N_steps, T, U_circuit, CRAB, 
                                                    optimizer, options, objective=objective, bounds=bounds)
                if cost is None or cost_temp < cost:
                    opt_params = opt_params_temp
                    cost = cost_temp
    
            # Apply hamiltonian with optimal parameters
            # Here we do not want the randomized parameters to be included -> CRAB=False in any case
            self.apply_cold_hamiltonian(qarg, N_steps, T, opt_params, CRAB=False)

        # Run LCD routine
        elif method == 'LCD':
            self.apply_lcd_hamiltonian(qarg, N_steps, T)

        else:
            raise ValueError(f'"{method}" is not an option for method. Choose "LCD" or "COLD".')

        # Measure qarg
        if not "shots" in mes_kwargs:
            mes_kwargs["shots"] = 5000
        res_dict = qarg.get_measurement(**mes_kwargs)

        # Add qubo cost in result dict
        for res in res_dict.keys():
            res_array = np.fromiter(res, dtype=int)
            res_dict[res] = [res_dict[res], self.QUBO_cost(res_array)]

        return res_dict
