.. _combine_single_qubit_gates:

combine_single_qubit_gates
==========================

.. currentmodule:: qrisp
.. autofunction:: combine_single_qubit_gates

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import combine_single_qubit_gates

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.z(0)
    qc.y(0)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
          ┌───┐┌───┐┌───┐
    qb_0: ┤ X ├┤ Z ├┤ Y ├
          └───┘└───┘└───┘

::

    pm = PassManager()
    pm += combine_single_qubit_gates
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
          ┌───────────┐
    qb_0: ┤ U3(0,0,0) ├
          └───────────┘

::

    # X, Z, Y combined into a single U3 gate
