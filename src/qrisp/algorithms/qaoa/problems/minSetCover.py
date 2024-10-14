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

from qrisp import QuantumVariable, QuantumBool, x, mcx
from qrisp.algorithms.qaoa.mixers import controlled_RX_mixer_gen


def create_min_set_cover_mixer(sets, universe):
    r"""
    Creates the ``controlled_RX_mixer`` for an instance of the minimum set cover problem following `Hadfield et al. <https://arxiv.org/abs/1709.03489>`_

    The belonging ``predicate`` function indicates if a set can be swapped out of the solution.

    Parameters
    ----------
    sets : list[set]
        A list of sets.
    universe : set
        The union of all sets.

    Returns
    -------
    function
        A Python function receiving a ``QuantumVariable`` and real parameter $\beta$. 
        This function performs the application of the mixer associated to the problem instance.

    """

    membership_dict = {element: [i for i, subset in enumerate(sets) if element in subset] for element in universe}

    def predicate(qv,i):
        anc = QuantumVariable(len(sets[i]))
        x(anc)
        for anc_index, element in enumerate(sets[i]):    
            other_sets = [item for item in membership_dict[element] if item != i]
            mcx([qv[set_index] for set_index in other_sets],anc[anc_index],ctrl_state="0"*len(other_sets))
        qbl = QuantumBool()
        mcx(anc,qbl)
        return qbl
    
    controlled_RX_mixer=controlled_RX_mixer_gen(predicate)

    return controlled_RX_mixer


def create_min_set_cover_cl_cost_function(sets, universe):
    """
    Creates the classical cost function for an instance of the minimum set cover problem.

    Parameters
    ----------
    sets : list[set]
        A list of sets.
    universe : set
        The union of all sets.

    Returns
    -------
    cl_cost_function : function
        The classical cost function for the problem instance, which takes a dictionary of measurement results as input.

    """

    def cl_cost_function(res_dic):
        cost = 0
        for state, prob in res_dic.items():
            indices = [index for index, value in enumerate(state) if value == '1']
            solution_sets = [sets[index] for index in indices]
            if len(solution_sets)>0 and set.union(*solution_sets)==universe:
                cost += len(indices)*prob
            else:
                cost += len(sets)

        return cost

    return cl_cost_function 


def min_set_cover_init_function(qv):
    r"""
    Prepares the initial state $\ket{1}^{\otimes n}$.
    
    Parameters
    ----------
    qv : :ref:`QuantumVariable`
        The quantum argument.
    
    """
    x(qv)


def min_set_cover_problem(sets, universe):
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    sets : list[set]
        A list of sets.
    universe : set
        The union of all sets.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for MinSetCover for given ``sets`` and ``universe``.

    """        
    from qrisp.qaoa import QAOAProblem, RZ_mixer
    
    return QAOAProblem(cost_operator=RZ_mixer,
                        mixer= create_min_set_cover_mixer(sets, universe),
                        cl_cost_function=create_min_set_cover_cl_cost_function(sets, universe),
                        init_function=min_set_cover_init_function)

