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

from qrisp import h, cx, rz, conjugate
import itertools


def parity(qarg, indices):
    n = len(indices)
    for i in range(n-1):
        cx(qarg[indices[i]],qarg[indices[i+1]])


def create_e3lin2_cost_operator(clauses):
    r"""
    Creates the cost operator for an instance of the E3Lin2 problem following `Hadfield et al. <https://arxiv.org/abs/1709.03489>`_
    The cost operator is given by $e^{-i\gamma H}$ where

    .. math::

        H=\sum_{j=1}^m H_j,\qquad H_j=(-1)^{b_j}Z_{a_{1,j}}Z_{a_{2,j}}Z_{a_{3,j}}

    Parameters
    ----------
    clasues : list[list[int]]
        The clasues defining the problem.

    Returns
    -------
    function
        A Python function receiving a ``QuantumVariable`` and real parameter $\beta$. 
        This function performs the application of the mixer associated to the graph $G$.

    """

    def cost_operator(qv, gamma):
        for clause in clauses:
            with conjugate(parity)(qv, clause[:3]):
                rz((-1)**clause[3]*gamma,qv[clause[2]])

    return cost_operator


def create_e3lin2_cl_cost_function(clauses):
    """
    Creates the cost operator for an instance of the E3Lin2 problem.

    Parameters
    ----------
    clasues : list[list[int]]
        The clasues defining the problem.

    Returns
    -------
    cl_cost_function : function
        The classical cost function for the problem instance, which takes a dictionary of measurement results as input.

    """

    def cl_cost_function(res_dic):
        cost = 0
        for state, prob in res_dic.items():
               for clause in clauses:
                   if sum(int(state[clause[k]]) for k in range(3)) % 2 == clause[3]:
                       cost -= prob
        
        return cost

    return cl_cost_function 


def e3lin2_init_function(qv):
    r"""
    Prepares the initial state $\ket{+}^{\otimes n}$.
    
    Parameters
    ----------
    qv : :ref:`QuantumVariable`
        The quantum argument.
    
    """
    h(qv)


def e3lin2_problem(clauses):
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    clauses : list[list[int]]
        The clauses of the E3Lin2 problem instance.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for E3Lin2 for given ``clauses``.

    """        
    from qrisp.qaoa import QAOAProblem, RX_mixer

    return QAOAProblem(cost_operator=create_e3lin2_cost_operator(clauses),
                        mixer=RX_mixer,
                        cl_cost_function=create_e3lin2_cl_cost_function(clauses),
                        init_function=e3lin2_init_function)

