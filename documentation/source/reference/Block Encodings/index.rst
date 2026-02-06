.. _block_encodings:

Block Encodings
===============

.. currentmodule:: qrisp.block_encodings

.. toctree::
   :hidden:
   
   BlockEncoding
   block_encodings_applications
   block_encodings_numpy
   block_encodings_nisq


**A High-Level Abstraction for Quantum Linear Algebra**

The :ref:`BlockEncoding` provides a powerful programming abstraction for Quantum Linear Algebra. 
While traditional quantum programming often focuses on low-level gate operations (in qrisp: higher-level operations using functions and variables), 
this interface allows developers to work directly with non-unitary matrices by embedding them into larger unitary operators.

**Key Concepts**

- **Matrix-Centric Design:** Instead of manually constructing circuits, users define a target matrix :math:`A`. The module automatically handles the construction of a unitary :math:`U` such that :math:`A` is represented in the top-left block (scaled by a factor :math:`\alpha`):

.. math::

    U = \begin{pmatrix} A/\alpha & \cdot\\ \cdot & \cdot \end{pmatrix}

- **Algebraic Composability:** Perform complex operations—such as the addition or multiplication of linear operators — through a modular interface that manages underlying ancilla qubits and normalization factors.