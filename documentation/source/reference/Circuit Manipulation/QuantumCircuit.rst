.. _QuantumCircuit:

QuantumCircuit
==============

.. currentmodule:: qrisp
.. autoclass:: QuantumCircuit


Methods
=======

Constructing QuantumCircuits
----------------------------
.. autosummary::
   :toctree: generated/
   
   QuantumCircuit.append
   QuantumCircuit.to_op
   QuantumCircuit.to_gate
   QuantumCircuit.add_qubit
   QuantumCircuit.add_clbit
   QuantumCircuit.copy
   QuantumCircuit.clearcopy
   QuantumCircuit.inverse
   QuantumCircuit.extend
   QuantumCircuit.bind_parameters
   QuantumCircuit.transpile

Evaluating QuantumCircuits
--------------------------

.. autosummary::
   :toctree: generated/

   QuantumCircuit.run
   QuantumCircuit.statevector_array
   QuantumCircuit.depth
   QuantumCircuit.t_depth
   QuantumCircuit.cnot_depth
   QuantumCircuit.num_qubits
   QuantumCircuit.count_ops
   QuantumCircuit.get_unitary
   QuantumCircuit.compare_unitary
   QuantumCircuit.to_latex
   QuantumCircuit.qasm
   
 
Interfacing
-----------

.. autosummary::
   :toctree: generated/

   QuantumCircuit.from_qiskit
   QuantumCircuit.to_qiskit
   QuantumCircuit.to_pennylane
   QuantumCircuit.to_pytket
   QuantumCircuit.to_stim
   QuantumCircuit.from_qasm_str
   QuantumCircuit.from_qasm_file
   QuantumCircuit.to_cirq
   


Operation application methods
-----------------------------

.. note::

   Each Qubit and Clbit parameter in these methods can be replaced by an integer, a list of integers or a list of Qubit/Clbit objects.

   >>> from qrisp import QuantumCircuit
   >>> qc = QuantumCircuit(5)
   >>> qc.cx(qc.qubits[1:], qc.qubits[0])
   >>> qc.x([0,1,2,3])
   >>> print(qc)
   
   .. code-block:: none
   
             ┌───┐┌───┐     ┌───┐┌───┐┌───┐
       qb_0: ┤ X ├┤ X ├─────┤ X ├┤ X ├┤ X ├
             └─┬─┘└─┬─┘┌───┐└─┬─┘└─┬─┘└───┘
       qb_1: ──■────┼──┤ X ├──┼────┼───────
                    │  ├───┤  │    │       
       qb_2: ───────■──┤ X ├──┼────┼───────
                       └───┘  │    │  ┌───┐
       qb_3: ─────────────────■────┼──┤ X ├
                                   │  └───┘
       qb_4: ──────────────────────■───────



.. autosummary::
   :toctree: generated/
   
   QuantumCircuit.measure
   QuantumCircuit.cx
   QuantumCircuit.cy
   QuantumCircuit.cz
   QuantumCircuit.h
   QuantumCircuit.x
   QuantumCircuit.y
   QuantumCircuit.z
   QuantumCircuit.mcx
   QuantumCircuit.ccx
   QuantumCircuit.rx
   QuantumCircuit.ry
   QuantumCircuit.rz
   QuantumCircuit.p
   QuantumCircuit.cp
   QuantumCircuit.u3
   QuantumCircuit.swap
   QuantumCircuit.t
   QuantumCircuit.t_dg
   QuantumCircuit.s
   QuantumCircuit.s_dg  
   QuantumCircuit.sx
   QuantumCircuit.sx_dg
   QuantumCircuit.rxx
   QuantumCircuit.rzz
   QuantumCircuit.xxyy
   QuantumCircuit.reset
   QuantumCircuit.unitary
   QuantumCircuit.gphase
   QuantumCircuit.id

  