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

from qrisp import h, rz, rx ,rzz
import itertools


def maxIndepSetCostOp(G):
    """

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    function
        A Python function receiving a ``QuantumVariable`` and real parameter $\gamma$. 
        This function performs the application of the cost operator associated to the graph $G$.

    """

    def partialcostMixer(qv, gamma):
        for pair in list(G.edges()):
            #cx(qv[pair[0]], qv[pair[1]])
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        rz(gamma, qv)

    return partialcostMixer


def maxIndepSetCostfct(G):
    """

    Parameters
    ----------
    G : nx.Graph
        The Graph for the problem instance.

    Returns
    -------
    cl_cost_function : function
        The classical function for the problem instance, which takes a dictionary of measurement results as input.

    """

    def cl_cost_function(res_dic):
        tot_energy = 0.001
        for state, prob in res_dic.items():
            temp = True
            energy = 0 
            indices = [index for index, value in enumerate(state) if value == '1']
            # get all combinations of vertices in graph that are marked as |1> by the solution 
            combinations = list(itertools.combinations(indices, 2))
            # if any combination is found in the list of G.edges(), the solution is invalid, and energy == 0
            for combination in combinations:
                if combination in G.edges():
                    temp = False
                    break
            # else we subtract the number of vertices marked as |1>
            if temp: 
                energy = -len(indices)
                tot_energy += energy*prob

        return tot_energy

    return cl_cost_function 

