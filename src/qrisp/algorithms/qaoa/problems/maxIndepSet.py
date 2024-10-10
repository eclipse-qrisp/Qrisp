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

from qrisp import QuantumBool, x, mcx
from qrisp.qaoa import controlled_RX_mixer_gen
import itertools


def create_max_indep_set_mixer(G):
    r"""
    Generates the mixer for an instance of the maximal independet set problem for a given graph $G$.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    function
        A Python function receiving a ``QuantumVariable`` and real parameter $\beta$. 
        This function performs the application of the mixer associated to the graph $G$.

    """

    neighbors_dict = {node: list(G.adj[node]) for node in G.nodes()}

    def predicate(qv,i):
        qbl = QuantumBool()
        if len(neighbors_dict[i])==0:
            x(qbl)
        else:
            mcx([qv[j] for j in neighbors_dict[i]],qbl,ctrl_state='0'*len(neighbors_dict[i]))
        return qbl

    controlled_RX_mixer=controlled_RX_mixer_gen(predicate)

    return controlled_RX_mixer


def create_max_indep_set_cl_cost_function(G):
    """
    Generates the classical cost function for an instance of the maximal independet set problem for a given graph $G$.

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
        energy = 0
        for state, prob in res_dic.items():
            temp = True
            indices = [index for index, value in enumerate(state) if value == '1']
            combinations = list(itertools.combinations(indices, 2))
            for combination in combinations:
                if combination in G.edges():
                    temp = False
                    break
            if temp: 
                energy += -len(indices)*prob

        return energy

    return cl_cost_function 


def max_indep_set_init_function(qv):
    """
    
    Parameters
    ----------
    qv : :ref:`QuantumVariable`
        The quantum argument.
    
    """
    pass