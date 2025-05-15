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

from itertools import combinations
import networkx as nx


def transform_max_set_pack_to_mis(problem):
    """
    Transforms a Maximum Set Packing problem instance into a Maximum Independent Set (MIS) problem instance.

     Parameters
     ----------
     problem : list[set]
         A list of sets specifying the problem.

     Returns
     -------
     G : nx.Graph
         The corresponding graph to be solved by an MIS implementation.


    """

    def non_empty_intersection(problem):
        return [
            (i, j)
            for (i, s1), (j, s2) in combinations(enumerate(problem), 2)
            if s1.intersection(s2)
        ]

    # create constraint graph
    G = nx.Graph()
    G.add_nodes_from(range(len(problem)))
    G.add_edges_from(non_empty_intersection(problem))

    return G
