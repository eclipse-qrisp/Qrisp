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


from qrisp import h, QuantumFloat, multi_measurement, auto_uncompute, QuantumBool, mcx
from qrisp.quantum_backtracking import QuantumBacktrackingTree
import numpy as np
from sympy import nsimplify, Float

def test_backtracking():
    
    @auto_uncompute    
    def P(tree):
        temp_0 = (tree.h == 0)
        predicate = QuantumBool()
        mcx(list(tree.branch_qa) + list(temp_0), predicate)
        
        return predicate

    @auto_uncompute    
    def Q(tree):
        return (tree.h != tree.max_depth) & (tree.h != (tree.max_depth-1)) & (tree.branch_qa[0] == tree.branch_qa[1])

    tree = QuantumBacktrackingTree(3, QuantumFloat(1, name = "branch_qf*"), P, Q)
    tree.init_node([])
    res = tree.estimate_phase(4)
    
    mes_res = res.get_measurement()
    
    assert mes_res[0] < 0.25
    
    
    @auto_uncompute    
    def P(tree):
        predicate = QuantumBool()
        mcx(list(tree.branch_qa), predicate)
        return predicate

    @auto_uncompute    
    def Q(tree):
        return QuantumBool()
    
    tree = QuantumBacktrackingTree(3, QuantumFloat(1, name = "branch_qf*"), P, Q)
    tree.init_node([])
    res = tree.estimate_phase(4)
    
    mes_res = res.get_measurement()
    
    assert mes_res[0] > 0.375
    
    for i in range(1,5):
    
        tree = QuantumBacktrackingTree(i, QuantumFloat(1, name = "branch_qf*"), P, Q)
        
        tree.init_phi(i*[1])
        
        temp_sv_0 = tree.qs.statevector().expand().evalf()
        tree.quantum_step()
        temp_sv_1 = tree.qs.statevector().expand().evalf()
        
        diff = (temp_sv_0 - temp_sv_1).expand().evalf()
        diff_atoms = list(diff.atoms(Float))
        s = sum(diff_atoms)
        assert abs(s) < 1E-4
        
        tree = QuantumBacktrackingTree(i, QuantumFloat(1, name = "branch_qf*"), P, Q)
        tree.init_node(i*[1])
        
        temp_sv_0 = tree.qs.statevector().expand().evalf()
        tree.qstep_diffuser(even = True)
        temp_sv_1 = tree.qs.statevector().expand().evalf()
        
        diff = (temp_sv_0 - temp_sv_1).expand().evalf()
        diff_atoms = list(diff.atoms(Float))
        s = sum(diff_atoms)

        assert abs(s) < 1E-4
    
        
    tree = QuantumBacktrackingTree(3, QuantumFloat(1, name = "branch_qf*"), P, Q)
    solution = tree.find_solution(precision = 4)
    assert solution == [1,1,1]
    
    
    #Test non-binary tree
    
    @auto_uncompute    
    def accept(tree):
        height_condition = tree.h == 0
        path_condition = QuantumBool()
        qubit_list = sum([list(qv) for qv in tree.branch_qa.flatten()], [])
        mcx(qubit_list, path_condition, method = "balauca")
        return height_condition & path_condition

    @auto_uncompute    
    def reject(tree):
        return QuantumBool()

    max_depth = 2
    
    tree = QuantumBacktrackingTree(max_depth, QuantumFloat(2), accept, reject, subspace_optimization = True)
    tree.init_node([])
    qpe_res = tree.estimate_phase(3)
    
    mes_res = qpe_res.get_measurement()
    
    assert mes_res[0] > 0.375
    
    @auto_uncompute    
    def reject(tree):
        return tree.h == 1

    max_depth = 2
    
    tree = QuantumBacktrackingTree(max_depth, QuantumFloat(2), accept, reject, subspace_optimization = True)
    tree.init_node([])
    qpe_res = tree.estimate_phase(3)
    
    mes_res = qpe_res.get_measurement()
    
    assert mes_res[0] < 0.25
