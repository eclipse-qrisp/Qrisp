.. _is_toffoli:

is_toffoli
==========

.. currentmodule:: qrisp
.. autofunction:: is_toffoli

Example
-------

::

    from qrisp import QuantumCircuit
    from qrisp import is_toffoli

    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    print(qc)

.. code-block:: none

                    
    qb_0: ──■──
            │  
    qb_1: ──■──
          ┌─┴─┐
    qb_2: ┤ X ├
          └───┘

::

    result = is_toffoli(qc.data[0].op)
    print(result)  # True
