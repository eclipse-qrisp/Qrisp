"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp import h, app_sb_phase_polynomial
import sympy as sp
import math


def create_maxsat_cost_polynomials(clauses):
    """
    Generates a list of polynomials representing the cost function for each clause, and a list of symbols.
    
    Parameters
    ----------
    clauses : list[list[int]]
        The clauses of the maximum satisfiability problem.

    Returns
    -------
    cost_polynomials : list[sympy.Expr]
        A list of the cost functions for each clause as SymPy polynomials.
    symbols : list[sympy.Symbol]
        A list of SymPy symbols.

    """

    all_indices = list(set(abs(index) for clause in clauses for index in clause))
    symbols = [sp.Symbol(f"x{i}") for i in range(1,len(all_indices)+1)]
    cost_polynomials = []
    for clause in clauses:
        C = 1 - sp.prod((1-symbols[index-1]) if index>0 else symbols[-index-1] for index in clause)
        cost_polynomials.append(C)

    return cost_polynomials, symbols


def create_maxsat_cl_cost_function(clauses):
    """
    Generates the classical cost function for an instance of the maximum satisfiability problem.

    Parameters
    ----------
    clauses : list[list[int]]
        The clauses of the maximum satisfiability problem.

    Returns
    -------
    cl_cost_function : function
        The classical function for the problem instance, which takes a dictionary of measurement results as input.

    """

    def cl_cost_function(res_dic):
        energy = 0
        for state, prob in res_dic.items():
            for clause in clauses:
                energy += -(1-math.prod((1-int(state[index-1])) if index>0 else int(state[-index-1]) for index in clause))*prob

        return energy

    return cl_cost_function 


def create_maxsat_cost_operator(clauses):
    r"""
    Generates the cost operator for an instance of the maximum satisfiability problem.
    For a given cost function 

    .. math::

        C(x) = \sum_{\alpha=1}^m C_{\alpha}(x)

    the cost operator is given by $e^{-i\gamma C}$ where $C=\sum_x C(x)\ket{x}\bra{x}$.

    Parameters
    ----------
    clauses : list[list[int]]
        The clauses of the maximum satisfiability problem.

    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\beta$.
        This function performs the application of the cost operator.

    """
   
    cost_polynomials, symbols = create_maxsat_cost_polynomials(clauses)

    def cost_operator(qv, beta):
        for P in cost_polynomials:
            app_sb_phase_polynomial([qv], -P, symbols, t=beta)

    return cost_operator


def maxsat_init_function(qv):
    r"""
    Prepares the initial state $\ket{+}^{\otimes n}$.
    
    Parameters
    ----------
    qv : :ref:`QuantumVariable`
        The quantum argument.
    
    """
    h(qv)
