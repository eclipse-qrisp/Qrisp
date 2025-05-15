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

from qrisp.algorithms.qaoa.problems.maxIndepSet import (
    create_max_indep_set_cl_cost_function,
    create_max_indep_set_mixer,
    max_indep_set_init_function,
)
import networkx as nx
from itertools import combinations


def max_set_packing_problem(sets):
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    sets : list[set]
        The sets for the problem instance.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for MaxSetPacking for given ``sets``.

    """
    from qrisp.qaoa import QAOAProblem, RZ_mixer

    def non_empty_intersection(sets):
        return [
            (i, j)
            for (i, s1), (j, s2) in combinations(enumerate(sets), 2)
            if s1.intersection(s2)
        ]

    # create constraint graph
    G = nx.Graph()
    G.add_nodes_from(range(len(sets)))
    G.add_edges_from(non_empty_intersection(sets))

    return QAOAProblem(
        cost_operator=RZ_mixer,
        mixer=create_max_indep_set_mixer(G),
        cl_cost_function=create_max_indep_set_cl_cost_function(G),
        init_function=max_indep_set_init_function,
    )
