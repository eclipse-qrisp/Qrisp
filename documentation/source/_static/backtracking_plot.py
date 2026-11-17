# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Created on Wed Sep 27 16:13:02 2023

@author: sea
"""

from qrisp.quantum_backtracking import *

from qrisp import *


def accept(tree):
    return QuantumBool()


@auto_uncompute
def reject(tree):
    return (tree.h == 1) & (tree.branch_qa[1] == 1)


max_depth = 3
tree = QuantumBacktrackingTree(max_depth, QuantumFloat(1), accept, reject)

tree.init_node([])

tree.quantum_step()
tree.quantum_step()
tree.quantum_step()
tree.quantum_step()

tree.visualize_statevector()
# plt.savefig("test.svg", format = "svg")
