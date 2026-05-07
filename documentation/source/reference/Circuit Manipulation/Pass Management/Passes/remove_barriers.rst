.. _remove_barriers:

remove_barriers
===============

.. currentmodule:: qrisp
.. autofunction:: remove_barriers

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import remove_barriers

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
          ┌───┐ ░      
    qb_0: ┤ H ├─░───■──
          └───┘ ░ ┌─┴─┐
    qb_1: ──────░─┤ X ├
                ░ └───┘

::

    pm = PassManager()
    pm.add_pass(remove_barriers)
    clean_qc = pm.run(qc)

    print("After:", clean_qc, sep="\n")

.. code-block:: none

   After:
          ┌───┐     
    qb_0: ┤ H ├──■──
          └───┘┌─┴─┐
    qb_1: ─────┤ X ├
               └───┘

::

    # Barrier (░) removed
