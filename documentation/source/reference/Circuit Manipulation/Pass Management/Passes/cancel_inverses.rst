.. _cancel_inverses:

cancel_inverses
===============

.. currentmodule:: qrisp
.. autofunction:: cancel_inverses

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import cancel_inverses

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                     
    qb_0: ──■────■──
          ┌─┴─┐┌─┴─┐
    qb_1: ┤ X ├┤ X ├
          └───┘└───┘

::

    pm = PassManager()
    pm.add_pass(cancel_inverses)
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
           
    qb_0: 
           
    qb_1: 

::

    # Empty circuit — both CX gates cancelled
