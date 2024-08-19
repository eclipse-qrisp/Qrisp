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
    Based on PennyLane unconstrained mixer implementation, initial state in :math:`(|0>+|1>)^(\otimes n)`. For explanation see the Pennylane implementation

    
    Parameters
    ----------
    G : nx.Graph
        Graph of the problem instance

    Returns
    -------
    QuantumCircuit: qrisp.QuantumCircuit
        the Operator applied to the circuit-QuantumVariable

    """

    def partialcostMixer(qv, gamma):
        for pair in list(G.edges()):
            #cx(qv[pair[0]], qv[pair[1]])
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        rz(gamma, qv)
        #return qv

    return partialcostMixer



def maxIndepSetclCostfct(G):
    """

    Parameters
    ----------
    G : nx.Graph
        Graph of the problem instance

    Returns
    -------
    Costfunction : function
        the classical function for the problem instance, which takes a dictionary of measurement results as input

    """

    def aClcostFct(res_dic):
        tot_energy = 0.001
        tot_counts = 0
        for state in res_dic.keys():
            # we assume solution is right
            temp = True
            energy = 0 
            #intlist = [int(s) for s in list(state)]
            intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
            # get all combinations of vertices in graph that are marked as |1> by the solution 
            #combinations = list(itertools.combinations(list(np.nonzero(intlist)[0]), 2))
            combinations = list(itertools.combinations(intlist, 2))
            # if any combination is found in the list of G.edges(), the solution is wrong, and energy == 0
            for combination in combinations:
                if combination in G.edges():
                    temp = False
                    break
            # else we just add the number of marked as |1> nodes
            if temp: 
                energy = -len(intlist)
                #energy = -sum(intlist)
            
            tot_energy += energy * res_dic[state]
            tot_counts += res_dic[state]

        #print(tot_energy/tot_counts)

        return tot_energy/tot_counts

    return aClcostFct



def init_state(qv):
    h(qv)
    return qv