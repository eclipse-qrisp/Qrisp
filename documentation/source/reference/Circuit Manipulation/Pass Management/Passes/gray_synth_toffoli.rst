.. _gray_synth_toffoli:

gray_synth_toffoli
==================

.. currentmodule:: qrisp
.. autofunction:: gray_synth_toffoli

Example
-------

::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import gray_synth_toffoli

    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

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
    pm.add_pass(gray_synth_toffoli)
    optimized_qc = pm.run(qc)

    # Toffoli replaced by Gray-synthesis decomposition (6 CNOT)
    # The circuit diagram uses the same Toffoli symbol but the
    # underlying instructions are decomposed
