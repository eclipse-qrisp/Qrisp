.. _commute_swaps:

commute_swaps
=============

.. currentmodule:: qrisp
.. autofunction:: commute_swaps

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import commute_swaps

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.swap(0, 1)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
          ┌───┐   
    qb_0: ┤ H ├─X─
          └───┘ │ 
    qb_1: ──────X─

::

    pm = PassManager()
    pm.add_pass(commute_swaps)
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
                   
    qb_0: ─X──────
           │ ┌───┐
    qb_1: ─X─┤ H ├
             └───┘

::

    # H commuted past SWAP — now appears after SWAP on qubit 1
