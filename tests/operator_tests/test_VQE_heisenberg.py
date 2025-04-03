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

from qrisp import QuantumVariable, QuantumFloat
from qrisp.vqe.problems.heisenberg import *
from qrisp.jasp import jaspify
import numpy as np
import jax.numpy as jnp
import networkx as nx

def test_vqe_heisenberg():

    # Create a graph
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(0,3)])
    
    vqe = heisenberg_problem(G,1,1)

    results = []
    for i in range(5):
        res = vqe.run(lambda : QuantumVariable(G.number_of_nodes()),
                depth=2,
                max_iter=50)
        results.append(res)
    
    assert np.abs(min(results)-(-8.0)) < 2e-2


def test_jasp_vqe_heisenberg():

    @jaspify(terminal_sampling=True)
    def main():
        # Create a graph
        G = nx.Graph()
        G.add_edges_from([(0,1),(1,2),(2,3),(0,3)])
    
        vqe = heisenberg_problem(G,1,1)

        results = jnp.array([0.0]*5)
        for i in range(5):
            res = vqe.run(lambda : QuantumFloat(G.number_of_nodes()),
                    depth=1,
                    max_iter=50)
            results = results.at[i].set(res)

        return results

    results = main()
    
    assert np.abs(min(results)-(-8.0)) < 1