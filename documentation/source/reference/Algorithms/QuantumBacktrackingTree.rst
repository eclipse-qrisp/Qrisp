.. _QuantumBacktrackingTree:

Quantum Backtracking
====================

.. currentmodule:: qrisp.quantum_backtracking
.. autoclass:: QuantumBacktrackingTree

Example
-------

Define accept and reject functions that evaluate conditions on the tree's current node, then use :meth:`estimate_phase <qrisp.quantum_backtracking.QuantumBacktrackingTree.estimate_phase>` to find the solution:

>>> from qrisp import QuantumFloat, QuantumBool, auto_uncompute
>>> from qrisp.quantum_backtracking import QuantumBacktrackingTree
>>> @auto_uncompute
... def reject(tree):
...     exclude_init = tree.h < tree.max_depth - 1
...     alternation_condition = tree.branch_qa[0] == tree.branch_qa[1]
...     return exclude_init & alternation_condition
>>> @auto_uncompute
... def accept(tree):
...     height_condition = tree.h == tree.max_depth - 3
...     path_condition = tree.branch_qa[0] == 0
...     path_condition = path_condition & (tree.branch_qa[1] == 0)
...     path_condition = path_condition & (tree.branch_qa[2] == 1)
...     return height_condition & path_condition
>>> tree = QuantumBacktrackingTree(max_depth = 3, branch_qv = QuantumFloat(1), accept = accept, reject = reject)
>>> tree.init_node([])
>>> qpe_res = tree.estimate_phase(precision = 4)
>>> mes_res = qpe_res.get_measurement()
>>> print(mes_res[0]) # doctest: +SKIP
0.025330506610132204

Methods
=======

Tree quantum operations
-----------------------

.. autosummary::
   :toctree: generated/
   
   QuantumBacktrackingTree.qstep_diffuser
   QuantumBacktrackingTree.quantum_step
   QuantumBacktrackingTree.estimate_phase
   QuantumBacktrackingTree.find_solution
   QuantumBacktrackingTree.init_phi
   QuantumBacktrackingTree.init_node
   

Tree evaluation
---------------

.. autosummary::
   :toctree: generated/
   
   QuantumBacktrackingTree.statevector
   QuantumBacktrackingTree.visualize_statevector
   QuantumBacktrackingTree.statevector_graph
   

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   QuantumBacktrackingTree.subtree
   QuantumBacktrackingTree.copy
   QuantumBacktrackingTree.path_decoder
