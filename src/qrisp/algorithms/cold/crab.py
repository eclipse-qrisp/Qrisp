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
from sympy import Symbol


class CRABObjective:
    """
    Class to incorporate CRAB (chopped random basis) optimization for the COLD method. 
    TODO: Mehr Erklärung

    Parameters
    ----------
    H_prob : :ref:`QubitOperator`
        Problem hamiltonian, used to compute the optimization objective via the ``expectation_value``.
    qarg : :ref:`QuantumVariable`
        The quantum argument to which the optimization circuit is applied.
    qc : :ref:`QuantumCircuit`
        The COLD circuit that is applied before measuring the qarg.
    N_opt : int
        Number of optimization parameters.
    last_x : None, optional
        Last result of the optimizer, to keep track of change of optimization loops.
    random_pulses : array, optional
        Random distribution of size ``N_opt`` to choose first random deviation. Default is a uniform distribution between [-0.5, 0.5].
    iteration : int, optional
        Optimization loop counter.
    rtol : float, optional
        tbd

    """

    def __init__(self, H_prob, qarg, qc, N_opt):
        self.H_prob = H_prob
        self.qarg = qarg
        self.qc = qc
        self.N_opt = N_opt
        self.last_x = None  # Keep track of last result
        self.random_pulses = np.random.uniform(-0.5, 0.5, N_opt)
        self.iteration = 0
        self.atol = 1e-6

    def __call__(self, params):
        
        # Detect new iteration by comparing parameter vectors
        if self.last_x is None or np.allclose(params, self.last_x, atol=self.atol) == False:
            # When optimizer moves away from last point, refresh random distribution
            if not self.last_x is None:
                # Change random pulse where parameters have changed
                new_pulse_index = np.where(self.last_x != params)[0]
                self.random_pulses[new_pulse_index] = np.random.uniform(-0.5, 0.5)
            self.iteration += 1
        self.last_x = params.copy()

        # Parameters to give to the quantum circuit for compilation
        # Optimization params
        subs_dic = {Symbol(f"par_{i}"): params[i] for i in range(len(params))}
        # Random pulse params
        subs_dic.update({Symbol(f"r_{k}"): self.random_pulses[k] for k in range(self.N_opt)})

        # Evluate cost
        cost = self.H_prob.expectation_value(self.qarg, compile=False,
                                          subs_dic=subs_dic, precompiled_qc=self.qc)()
        return cost