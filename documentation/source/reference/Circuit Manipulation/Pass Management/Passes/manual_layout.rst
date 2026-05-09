.. _manual_layout:

manual_layout
=============

.. currentmodule:: qrisp
.. autofunction:: manual_layout

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import manual_layout

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(1, 2)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
          ┌───┐
    qb_0: ┤ H ├
          └───┘
    qb_1: ──■──
          ┌─┴─┐
    qb_2: ┤ X ├
          └───┘

::

    pm = PassManager()
    pm += manual_layout([2, 0, 1])  # Logical 0→2, 1→0, 2→1
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
                
    qb_1: ──■──
          ┌─┴─┐
    qb_2: ┤ X ├
          ├───┤
    qb_0: ┤ H ├
          └───┘

::

    # Qubit layout permuted: H(0) moved to bottom, CX now on qb_1,qb_2
