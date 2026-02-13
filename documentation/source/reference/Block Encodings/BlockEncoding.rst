.. _BlockEncoding:

BlockEncoding
=============

.. currentmodule:: qrisp.block_encodings
.. autoclass:: BlockEncoding

Constructors
------------

.. list-table::
    :header-rows: 0
    :widths: 30 70

    * - :func:`~qrisp.block_encodings.BlockEncoding.from_array`
      - Constructs a BlockEncoding from a 2-D array.
    * - :func:`~qrisp.block_encodings.BlockEncoding.from_lcu`
      - Constructs a BlockEncoding using the Linear Combination of Unitaries (LCU) protocol.
    * - :func:`~qrisp.block_encodings.BlockEncoding.from_operator`
      - Constructs a BlockEncoding from an operator.

.. toctree:: 

   from_array
   from_lcu
   from_operator

Utilities
---------

.. list-table::
    :header-rows: 0
    :widths: 30 70

    * - :func:`~qrisp.block_encodings.BlockEncoding.apply`
       - Applies the BlockEncoding unitary to the given operands.
    * - :func:`~qrisp.block_encodings.BlockEncoding.apply_rus`
       - Applies the BlockEncoding using :ref:`RUS`.
    * - :func:`~qrisp.block_encodings.BlockEncoding.chebyshev`
       - Returns a BlockEncoding representing $k$-th Chebyshev polynomial of the first kind applied to the operator.
    * - :func:`~qrisp.block_encodings.BlockEncoding.create_ancillas`
       - Returns a list of ancilla QuantumVariables for the BlockEncoding.
    * - :func:`~qrisp.block_encodings.BlockEncoding.expectation_value`
       - Measures the expectation value of the operator using the Hadamard test protocol.
    * - :func:`~qrisp.block_encodings.BlockEncoding.qubitization`
       - Returns a BlockEncoding representing the qubitization walk operator.
    * - :func:`~qrisp.block_encodings.BlockEncoding.resources`
       - Estimate the quantum resources required for the BlockEncoding.

.. toctree::
    :hidden:

    apply
    apply_rus
    chebyshev
    create_ancillas
    expectation_value
    qubitization
    resources

.. autosummary::

    BlockEncoding.apply
    BlockEncoding.apply_rus
    BlockEncoding.chebyshev
    BlockEncoding.create_ancillas
    BlockEncoding.expectation_value
    BlockEncoding.qubitization
    BlockEncoding.resources

Arithmetic
----------

.. list-table::
    :header-rows: 0
    :widths: 30 70

    * - :func:`~qrisp.block_encodings.BlockEncoding.__add__`
       - Returns a BlockEncoding of the sum of two operators.
    * - :func:`~qrisp.block_encodings.BlockEncoding.__matmul__`
       - Returns a BlockEncoding of the product of two operators.
    * - :func:`~qrisp.block_encodings.BlockEncoding.__mul__`
       - Returns a BlockEncoding of the scaled operator.
    * - :func:`~qrisp.block_encodings.BlockEncoding.__neg__`
       - Returns a BlockEncoding of the negated operator.
    * - :func:`~qrisp.block_encodings.BlockEncoding.__sub__`
       - Returns a BlockEncoding of the difference between two operators.
    * - :func:`~qrisp.block_encodings.BlockEncoding.kron`
       - Returns a BlockEncoding of the Kronecker product (tensor product) of two operators.

.. toctree::
    :hidden:

    add
    matmul
    mul
    neg
    sub
    kron

.. autosummary::

    BlockEncoding.__add__
    BlockEncoding.__matmul__
    BlockEncoding.__mul__
    BlockEncoding.__neg__
    BlockEncoding.__sub__
    BlockEncoding.kron

Algorithms & Applications
-------------------------

.. list-table::
    :header-rows: 0
    :widths: 30 70

    * - :func:`~qrisp.block_encodings.BlockEncoding.inv`
       - Returns a BlockEncoding approximating the matrix inversion of the operator.
    * - :func:`~qrisp.block_encodings.BlockEncoding.poly`
       - Returns a BlockEncoding representing a polynomial transformation of the operator.
    * - :func:`~qrisp.block_encodings.BlockEncoding.sim`
       - Returns a BlockEncoding approximating Hamiltonian simulation of the operator.

.. toctree::
    :hidden:

    inv
    poly
    sim

.. autosummary::

    BlockEncoding.inv
    BlockEncoding.poly
    BlockEncoding.sim
