.. _decompose:

decompose
=========

.. currentmodule:: qrisp
.. autofunction:: decompose

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import decompose

    qc = QuantumCircuit(3)
    qc.mcx([0, 1], 2)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                
    qb_0: ──■──
            │  
    qb_1: ──■──
          ┌─┴─┐
    qb_2: ┤ X ├
          └───┘

::

    pm = PassManager()
    pm += decompose()
    decomposed_qc = pm.run(qc)

    # MCX is fully decomposed into elementary gates

::

    # Decompose only specific gates with a predicate
    pm2 = PassManager()
    pm2 += decompose(decompose_predicate=lambda op: "cx" in op.name)

    # Decompose only one layer
    pm3 = PassManager()
    pm3 += decompose(level=1)
