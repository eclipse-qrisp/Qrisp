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

from qrisp import app_sb_phase_polynomial
import sympy as sp


def create_maxsat_cost_polynomials(clauses, num_vars):
    """
    Generates a list of polynomials representing the cost function for each clause, and a list of symbols.
    
    Parameters
    ----------
    clauses : list[list[int]]
        The clauses of the maximum satisfiability problem.
    num_vars : int
        The number of variables.

    Returns
    -------
    cost_polynomials : list[sympy.Expr]
        A list of the cost functions for each clause as SymPy polynomials.
    symbols : list[sympy.Symbol]
        A list of SymPy symbols.

    """

    symbols = [sp.Symbol(f"x{i}") for i in range(1,num_vars+1)]
    cost_polynomials = []
    for clause in clauses:
        C = 1 - sp.prod((1-symbols[index-1]) if index>0 else symbols[-index-1] for index in clause)
        cost_polynomials.append(C)

    return cost_polynomials, symbols


def create_maxsat_cl_cost_function(cost_polynomials, symbols):
    """
    Generates the classical cost function for an instance of the maximum satisfiability problem.

    Parameters
    ----------
    cost_polynomials : list[sympy.Expr]
        A list of the cost functions for each clause as SymPy polynomials.
    symbols : list[sympy.Symbol]
        A list of SymPy symbols.

    Returns
    -------
    cl_cost_function : function
        The classical function for the problem instance, which takes a dictionary of measurement results as input.

    """

    def cl_cost_function(res_dic):
        energy = 0
        for state, prob in res_dic.items():
            for P in cost_polynomials:
                energy += -P.subs({symbols[k]:int(state[k]) for k in range(len(symbols))})*prob

        return energy

    return cl_cost_function 


def create_maxsat_cost_operator(cost_polynomials, symbols):
    r"""
    Generates the cost operator for an instance of the maximum satisfiability problem.
    For a given cost function 

    .. math::

        C(x) = \sum_{\alpha=1}^m C_{\alpha}(x)

    the cost operator is given by $e^{-i\beta C}$ where $C=\sum_x C(x)\ket{x}\bra{x}$.

    Parameters
    ----------
    cost_polynomials : list[sympy.Expr]
        A list of the cost functions for each clause as SymPy polynomials.
    symbols : list[sympy.Symbol]
        A list of SymPy symbols.

    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\beta$.
        This function performs the application of the cost operator.

    """

    def cost_operator(qv, beta):
        for P in cost_polynomials:
            app_sb_phase_polynomial([qv], -P, symbols, t=beta)

    return cost_operator


