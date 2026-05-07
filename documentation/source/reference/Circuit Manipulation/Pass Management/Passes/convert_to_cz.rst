.. _convert_to_cz:

convert_to_cz
=============

.. currentmodule:: qrisp
.. autofunction:: convert_to_cz

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import convert_to_cz

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                
    qb_0: ──■──
          ┌─┴─┐
    qb_1: ┤ X ├
          └───┘

::

    pm = PassManager()
    pm.add_pass(convert_to_cz())           # lenient: unknown gates pass through
    # pm.add_pass(convert_to_cz(strict=True))  # strict: raises on unknown gates
    optimized_qc = pm.run(qc)

    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
                        
    qb_0: ──────■──────
          ┌───┐ │ ┌───┐
    qb_1: ┤ H ├─■─┤ H ├
          └───┘   └───┘

::

    # CX decomposed into H—CZ—H
