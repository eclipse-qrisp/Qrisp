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

import time

import numpy as np
from scipy.optimize import minimize
from sympy import Symbol

from qrisp import QuantumArray, h, x
from qrisp.algorithms.qaoa.qaoa_benchmark_data import QAOABenchmark

import jax
import jax.numpy as jnp
from qrisp.jasp import check_for_tracing_mode, sample, jrange
from qrisp.jasp.optimization_tools.optimize import minimize as jasp_minimize
import sympy as sp
from qrisp.algorithms.cold.crab import CRABObjective

class COLDproblem:


    def __init__(
        self, 
        qarg_prep, 
        H_i,
        H_p,
        A_lam,
        H_control,
        sympy_lambda, 
        alpha,
        
        callback=False
    ):
        
        self.qarg_prep = qarg_prep
        self.H_i = H_i
        self.H_p = H_p
        self.A_lam = A_lam
        self.H_control = H_control

        
        self.sympy_lambda = sympy_lambda
        self.alpha = alpha
        #t, T = sp.symbols('t T', real=True)
        #self.lam_deriv = sp.diff(lamb, t)
        

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
        Set the initial state preparation function for the QAOA problem.

        Parameters
        ----------
        init_function : function
            The initial state preparation function for the specific QAOA problem instance.
        """
        self.init_function = init_function

    # Precompute f (sine) and f_deriv (cosine) for each timestep
    


    def apply_cold_hamiltonian(self, 
                               qarg, N_steps, T, beta, CRAB=False):
        
        

        def precompute_opt_pulses(t_list, N_opt):

            if CRAB:
                # Matrices and arrays must be sympy objects
                sin_matrix = sp.MutableDenseMatrix.zeros(N_steps, N_opt)
                cos_matrix = sp.MutableDenseMatrix.zeros(N_steps, N_opt)
                t_list = sp.Array(t_list)

                # Add symbols for random parameters to be called in optimizaton
                r_params = [Symbol("r_"+str(i)) for i in range(N_opt)]
                for k in range(N_opt):
                    sin_matrix[:, k] = sp.Matrix(t_list.applyfunc(lambda t: sp.sin(sp.pi * (k+1+r_params[k]) * t / T)))
                    cos_matrix[:, k] = sp.Matrix(t_list.applyfunc(lambda t: (sp.pi/T * (k+1+r_params[k])) * sp.cos(sp.pi * (k+1+r_params[k]) * t / T)))

            else:
                sin_matrix = np.zeros((N_steps, N_opt))
                cos_matrix = np.zeros((N_steps, N_opt))

                for k in range(N_opt):
                    sin_matrix[:, k] = np.sin(np.pi * (k+1) * t_list/T)
                    cos_matrix[:, k] = (np.pi/T * (k+1)) * np.cos(np.pi * (k+1) * t_list/T)

            return sin_matrix, cos_matrix
        
        # Initialize qarg
        #self.qarg_prep(qarg)
        from qrisp import h as hadamard
        """ def qarg_prep(qarg):
            hadamard(qarg) """

        self.qarg_prep(qarg)

        # Precompute timegrid
        N_steps = int(N_steps)
        N_opt = len(beta)
        dt = T / N_steps
        time = np.linspace(dt, T, int(N_steps))

        tl, Tl = sp.symbols('t T', real=True)
        
        lam_deriv_expr = sp.diff(self.sympy_lambda(), tl)
        lam_t_func = sp.lambdify((tl, Tl), self.sympy_lambda(), 'numpy')
        lam_deriv_func = sp.lambdify((tl, Tl), lam_deriv_expr, 'numpy')
        lam = lam_t_func(time, T)
        lamdot = lam_deriv_func(time, T)
        sin_matrix, cos_matrix = precompute_opt_pulses(time,  N_opt)

        if CRAB:
            # Transform beta to sympy to work with symbols for random values
            beta = sp.Matrix(beta)

        # Trotterize Hamiltonians with different time-dependent prefactors
        """ U1 = self.H_i.trotterization()
        U2 = self.H_p.trotterization()
        U3 = self.A_lam.trotterization()
        U4 = self.H_control.trotterization() """

        # Apply hamiltonian to qarg for each timestep
        for s in range(N_steps):



            # Get alpha, f and f_deriv for the timestep
            f = sin_matrix[s-1, :] @ beta
            f_deriv = cos_matrix[s-1, :] @ beta
            alph = self.alpha(lam[s], f, f_deriv)
            
            if hasattr(alph, "__len__"):
                #print("has_len")
                alph = alph[0]
                f = f[0] 

            # H_0 contribution scaled by dt
            H_step = dt *(1-lam[s-1])* self.H_i + dt * lam[s-1]*self.H_p

            # AGP contribution scaled by dt* lambda_dot(t)
            H_step = dt * lamdot[s-1] * alph* self.A_lam

            # Control pulse contribution 
            H_step = H_step + dt * f*  self.H_control
            # Get unitary from trotterization and apply to qarg
            U = H_step.trotterization()
            U(qarg)  

            """ # Prefactor for U1
            a = dt * (1 - lam[s-1])
            U1(qarg, a)
            # Prefactor for U2
            b = dt * lam[s-1]
            U2(qarg, b)
            # Prefactor for U3
            c = dt * lamdot[s-1] * alph
            U3(qarg, c)
            # Prefactor for U4
            #if CRAB:
            #    d = dt * f #+sum(f[i, 0] for i in range(f.rows)) # richtig?
            #else:
            d = dt * f
            U4(qarg, d) """
        

    
    def compile_U(self,qarg, N_opt, N_steps, T, CRAB=False):

        temp = list(qarg.qs.data)
        # Initzialize parameters as symbols
        params = [Symbol("par_"+str(i)) for i in range(N_opt)]
        self.apply_cold_hamiltonian(qarg, N_steps, T, params, CRAB=CRAB)
        intended_measurements = list(qarg)
        qc = qarg.qs.compile(intended_measurements=intended_measurements)
        qarg.qs.data = temp

        return qc


    def optimization_routine(self, qarg, N_opt, qc, CRAB
                             ): 
        """
        Wrapper subroutine for the optimization method used in QAOA. The initial values are set and the optimization via ``COBYLA`` is conducted here.

        Parameters
        ----------
        qarg_prep : callable
            A function returning a :ref:`QuantumVariable` or :ref:`QuantumArray` to which the QAOA circuit is applied.
        depth : int
            The amont of QAOA layers.
        symbols : list
            The list of symbols used in the quantum circuit.
        mes_kwargs : dict, optional
            The keyword arguments for the measurement function. Default is an empty dictionary, as defined in previous functions.
        init_type : string, optional
            Specifies the way the initial optimization parameters are chosen. Available are ``random`` and ``tqa``. The default is ``random``.
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
        ndarray, shape (n,)
            The optimized parameters of the problem instance.
        float or jax.Array
            The expectation value of the classical cost function for the optimized parameters.
        """

        def objective(params):
            # Objective function to be minimized: 
            # Expectation value of the QUBO Hamiltonian

            # Dict to assign the optimization parameters
            subs_dic = {Symbol("par_"+str(i)): params[i] for i in range(len(params))}

            cost = self.H_p.expectation_value(qarg, compile=False, subs_dic=subs_dic, precompiled_qc=qc)()
            
            return cost
        
        init_point = np.random.rand(N_opt) * np.pi/2

        # Create CRAB objective to make sure randomization is applied at each optimization iteration
        if CRAB:
            objective = CRABObjective(self.H_p, qarg, qc, N_opt)
        # Otherwise objective from above is used
        else:
            objective = objective

        res = minimize(objective,
                        init_point,
                        method='Powell'
                        )
        
        return res.x
    

    def run(self, qarg, N_steps, T, N_opt, CRAB=False):

        # create the H_p and A_lam via the usage of Q only here? 

        qarg1, qarg2 = qarg.duplicate(), qarg.duplicate()

        # Compile COLD routine into a circuit
        U_circuit = self.compile_U(qarg1, N_opt, N_steps, T, CRAB)

        # Find optimal params for control pulse
        opt_params = self.optimization_routine(qarg2, N_opt, U_circuit, CRAB)

        # Apply hamiltonian with optimal parameters
        # Here we do not want the randomized parameters to be included -> CRAB=False
        self.apply_cold_hamiltonian(qarg, N_steps, T, opt_params, CRAB=False)

        # Measure qarg
        res_dict = qarg.get_measurement()

        return res_dict
