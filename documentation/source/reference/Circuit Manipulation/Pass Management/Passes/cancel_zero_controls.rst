.. _cancel_zero_controls:

cancel_zero_controls
====================

.. currentmodule:: qrisp
.. autofunction:: cancel_zero_controls

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import cancel_zero_controls

    qc = QuantumCircuit(2)
    qc.cx(0, 1)  # Qubit 0 starts in |0⟩ — the CX is a no-op
    qc.h(1)       # Now qubit 1 is marked as used

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                     
    qb_0: ──■───────
          ┌─┴─┐┌───┐
    qb_1: ┤ X ├┤ H ├
          └───┘└───┘

::

    pm = PassManager()
    pm += cancel_zero_controls
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
                
    qb_0: ─────
          ┌───┐
    qb_1: ┤ H ├
          └───┘

::

    # Only H(1) remains — CX cancelled because control was |0⟩
