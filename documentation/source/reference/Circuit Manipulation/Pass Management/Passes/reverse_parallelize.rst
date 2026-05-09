.. _reverse_parallelize:

reverse_parallelize
===================

.. currentmodule:: qrisp
.. autofunction:: reverse_parallelize

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import reverse_parallelize

    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.cx(0, 1)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                   
    qb_0: ─X───■──
           │ ┌─┴─┐
    qb_1: ─X─┤ X ├
             └───┘

::

    pm = PassManager()
    pm += reverse_parallelize
    optimized_qc = pm.run(qc)
    # Reverse-parallelization exposes SWAP commutation opportunities
