"""
********************************************************************************
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
********************************************************************************
"""

import itertools

import networkx as nx

from qrisp.algorithms.qaoa.problems.maxIndepSet import (
    create_max_indep_set_cl_cost_function,
    create_max_indep_set_mixer,
    max_indep_set_init_function,
)


def create_max_clique_cl_cost_function(G):
    """
    Creates the classical cost function for an instance of the maximum clique problem for a given graph ``G``.

    Parameters
    ----------
    G : nx.Graph
        The Graph for the problem instance.

    Returns
    -------
    cl_cost_function : function
        The classical cost function for the problem instance, which takes a dictionary of measurement results as input.

    """

    def cl_cost_function(res_dic):
        cost = 0
        for state, prob in res_dic.items():
            temp = True
            indices = [index for index, value in enumerate(state) if value == "1"]
            combinations = list(itertools.combinations(indices, 2))
            for combination in combinations:
                if combination not in G.edges():
                    temp = False
                    break
            if temp:
                cost += -len(indices) * prob

        return cost

    return cl_cost_function


def max_clique_problem(G):
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for MaxClique for a given graph ``G``.

    """
    from qrisp.qaoa import QAOAProblem, RZ_mixer

    G_complement = nx.complement(G)

    return QAOAProblem(
        cost_operator=RZ_mixer,
        mixer=create_max_indep_set_mixer(G_complement),
        cl_cost_function=create_max_indep_set_cl_cost_function(G_complement),
        init_function=max_indep_set_init_function,
    )
