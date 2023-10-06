# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:13:02 2023

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
import matplotlib.pyplot as plt
# plt.savefig("test.svg", format = "svg")
