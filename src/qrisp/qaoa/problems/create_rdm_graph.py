"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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


import networkx as nx

def create_rdm_graph(n,p, seed = 123):
    """
    Create a random networkx graph to be used for QAOAProblem implementations

    Parameters
    ----------

    n : int 
        number of nodes in the graph

    p : float 
        likelyhood of edge between any two nodes, chose between 0 and 1

    """
    G = nx.erdos_renyi_graph(n, p,seed = seed,  directed=False)
    return G