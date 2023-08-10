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


from qrisp import *

@auto_uncompute    
def reject(tree):
    exclude_init = (tree.h < tree.max_depth -1 )
    alternation_condition = (tree.branch_qa[0] == tree.branch_qa[1])

    return exclude_init & alternation_condition

@auto_uncompute    
def accept(tree):
    height_condition = (tree.h == tree.max_depth - 3)
    path_condition = (tree.branch_qa[0] == 0)
    path_condition = path_condition & (tree.branch_qa[1] == 0)
    path_condition = path_condition & (tree.branch_qa[2] == 1)

    return height_condition & path_condition


from qrisp.quantum_backtracking import QuantumBacktrackingTree

tree = QuantumBacktrackingTree(max_depth = 3,
                               branch_qv = QuantumFloat(1),
                               accept = accept,
                               reject = reject)

tree.init_node([])


print(tree.statevector())
import matplotlib.pyplot as plt
tree.visualize_statevector()
plt.show()


qpe_res = tree.estimate_phase(precision = 4)


mes_res = qpe_res.get_measurement()
print(mes_res[0])

def reject(tree):
    return QuantumBool()

tree = QuantumBacktrackingTree(max_depth = 3,
                               branch_qv = QuantumFloat(1),
                               accept = accept,
                               reject = reject)

tree.init_node([])

qpe_res = tree.estimate_phase(precision = 4)


#%%


@auto_uncompute    
def accept(tree):
    height_condition = (tree.h == tree.max_depth - 3)
    path_condition = QuantumBool()
    mcx(tree.branch_qa[:3], path_condition)

    return height_condition & path_condition

def reject(tree):
    return QuantumBool()

max_depth = 4
tree = QuantumBacktrackingTree(max_depth,
                                   branch_qv = QuantumFloat(1),
                                   accept = accept,
                                   reject = reject)

tree.find_solution(precision = 5)
