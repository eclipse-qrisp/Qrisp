.. _resolve_swaps:

resolve_swaps
=============

.. currentmodule:: qrisp
.. autofunction:: resolve_swaps

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import resolve_swaps

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.swap(0, 1)
    qc.cx(0, 1)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
          ┌───┐        
    qb_0: ┤ H ├─X───■──
          └───┘ │ ┌─┴─┐
    qb_1: ──────X─┤ X ├
                  └───┘

::

    pm = PassManager()
    pm.add_pass(resolve_swaps)
    routable_qc = pm.run(qc)

    print("After:", routable_qc, sep="\n")

.. code-block:: none

   After:
          ┌───┐┌───┐
    qb_0: ┤ H ├┤ X ├
          └───┘└─┬─┘
    qb_1: ───────■──

::

    # SWAP removed — operands remapped via permutation table
