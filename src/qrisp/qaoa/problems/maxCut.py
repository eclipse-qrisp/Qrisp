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

from qrisp import QuantumVariable, h, barrier, rz, rx , cx, QuantumArray,x 
import numpy as np
from scipy.optimize import minimize
from sympy import Symbol
import itertools

def maxcut_obj(x,G):
        cut = 0
        for i, j in G.edges():
            if x[i] != x[j]:                        
                cut -= 1    
        return cut

def create_maxcut_cl_cost_function(G):
    """
    Creates the classical cost function for maxcut for the specific graph G that we are attempting to cut.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    cl_cost_function : function
        Classical cost function, which in the end returns the ratio between 
        the energy calculated using the maxcut_obj objective funcion and the 
        amount of counts used in the experiment.

    """    
    def cl_cost_function(counts):
        
        def maxcut_obj(x, G):
            cut = 0
            for i, j in G.edges():
                # the edge is cut
                if x[i] != x[j]:                          
                    cut -= 1
            return cut
        
        energy = 0
        for meas, meas_count in counts.items():
            obj_for_meas = maxcut_obj(meas, G)
            energy += obj_for_meas * meas_count
        return energy
    
    return cl_cost_function


def create_maxcut_cost_operator(G):
    """
    Creates the maxcut operator as a sequence of unitary gates. In the QAOA overview section this is also called the phase separator U_P.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    cost_operator function.

    """
    def maxcut_cost_operator(qv, gamma):
        
        if len(G) != len(qv):
            raise Exception(f"Tried to call MaxCut cost Operator for graph of size {len(G)} on argument of invalid size {len(qv)}")
        
        for pair in list(G.edges()):
            cx(qv[pair[0]], qv[pair[1]])
            rz(2 * gamma, qv[pair[1]])
            cx(qv[pair[0]], qv[pair[1]])
            barrier(qv)
        
    return maxcut_cost_operator


def maxcut_problem(G):
    """
    Creates a QAOA problem instance taking the phase separator, appropriate mixer, and
    appropriate classical cost function into account.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    QAOAProblem : function
        QAOA problem instance for maxcut with which the QAOA algorithm is ran for.

    """        
    from qrisp.qaoa import QAOAProblem, RX_mixer
    
    return QAOAProblem(create_maxcut_cost_operator(G), RX_mixer, create_maxcut_cl_cost_function(G))