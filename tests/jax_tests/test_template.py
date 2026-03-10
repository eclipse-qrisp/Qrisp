"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from qrisp import *
from qrisp.vqe.problems.heisenberg import *

def test_template():
    
    @jaspify
    def main():
        
        qv = QuantumFloat(4, 1)
        template = qv.template()
        
        @quantum_kernel
        def inner_function(template):
            qv = template.construct()
            x(qv[0])
            return measure(qv)
        
        return inner_function(template)

    assert main() == 2

    @jaspify
    def main():
        
        qv = QuantumFloat(4, 1)
        template = qv.template()
        
        @quantum_kernel
        def inner_function():
            qv = template.construct()
            x(qv[0])
            return measure(qv)
        
        return inner_function()

    assert main() == 2

    qv = QuantumFloat(4, 1)
    template = qv.template()

    @jaspify
    def main():
        
        @quantum_kernel
        def inner_function():
            qv = template.construct()
            x(qv[0])
            return measure(qv)
        
        return inner_function()

    assert main() == 2

    ###### Test https://github.com/eclipse-qrisp/Qrisp/issues/172    
    import networkx as nx
    @jaspify(terminal_sampling=True) # WORKS without Jasp
    def main():
         # Create a graph
         G = nx.Graph()
         G.add_edges_from([(0,1),(1,2),(2,3),(0,3)])
         M = nx.maximal_matching(G)
         sing = create_heisenberg_init_function(M)

         qv = QuantumFloat(G.number_of_nodes())
         temp = qv.template()
         def qarg_prep(temp):
              return temp.construct()

         def prep(temp):
              # qf = QuantumFloat(G.number_of_nodes()) # WORKS without template
              qf = qarg_prep(temp)
              sing(qf)
              return qf
         
         H = create_heisenberg_hamiltonian(G, 1, 1)

         E = H.expectation_value(prep,diagonalisation_method='commuting_qw')(temp)

         return E

    assert abs(main()+6) < 0.5
    
    
    # Test static ehavior
    qv = QuantumVariable(4)
    temp = qv.template()
    qv2 = temp.construct()
    x(qv2)
    assert qv2.get_measurement() == {"1111": 1}
