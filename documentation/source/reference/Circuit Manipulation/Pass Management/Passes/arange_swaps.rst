.. _arange_swaps:

arange_swaps
============

.. currentmodule:: qrisp
.. autofunction:: arange_swaps

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import arange_swaps

    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.swap(1, 2)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                   
    qb_0: ──■─────
          ┌─┴─┐   
    qb_1: ┤ X ├─X─
          └───┘ │ 
    qb_2: ──────X─

::

    pm = PassManager()
    pm.add_pass(arange_swaps)
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")
    # SWAP internally reordered for better CX decomposition later
