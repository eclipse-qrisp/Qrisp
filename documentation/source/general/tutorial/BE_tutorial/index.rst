.. _BE101:

Quantum Linear Algebra with Qrisp
=================================

If I may dare to reference Futurama... "Welcome to the world of Tomorrow!" Or rather, to the future of programming Quantum Linear Algebra. 

Before anything, let's just show what the future looks like:

::

  import numpy as np
  from qrisp import *
  from qrisp.operators import X, Y, Z
  from qrisp.block_encodings import BlockEncoding

  A = np.array([[ 0.66,  0.02, -0.11, -0.16],
    [ 0.02,  0.82,  0.01, -0.12],
    [-0.11,  0.01,  0.93, -0.07],
    [-0.16, -0.12, -0.07,  0.69]])

  B = X(0)*X(1) + Y(0)*Y(1) + Z(0)*Z(1)

  epsilon = 0.01
  kappa = np.linalg.cond(A)

  def operand_prep():
    return QuantumFloat(2)

  BE_A = BlockEncoding.from_array(A)
  BE_B = B.pauli_block_encoding()

  B_C = BE_A.poly([1, 1, -2]) + BE_B.inv(epsilon, kappa)

  # Use a 2-qubit variable for a 2^2 by 2^2 matrix.
  res_dict = B_C.resources(QuantumFloat(2))
  print(res_dict)
  # {'gate counts': {'s': 60, 'ry': 2, 'cx': 1312, 'gphase': 62, 
  # 'h': 414, 'cz': 70, 's_dg': 60, 't_dg': 576, 't': 447, 'x': 238, 
  # 'p': 84, 'cy': 62, 'u3': 234}, 'depth': 1904, 'qubits': 16}
   
It's meaningful to provide insight on ``epsilon`` and ``kappa`` - the former corresponds to the precision of your approximation of the inverse matrix, 
while the latter is the condition number of the matrix, which is a measure of how "well-behaved" the matrix is for inversion. 
The higher the condition number, the more sensitive the solution is to changes in the input, and thus the more resources are required to achieve a certain precision. 
You will get to learn more about it in the tutorials.

If the concept of block encodings is not all that familiar to you, don't worry. 
You will learn everything you need to know about it in the tutorials that are listed below through their respective content contained in their respective tables. 
The idea here was to let the simplicity speak for itself, and to show you that you can do all of this with a few lines of code, 
without having to worry about the underlying circuit construction, ancilla management, 
or any of the other complexities that come with programming quantum linear algebra at the gate level. All of this (and more) is explained in the related tutorials.

This tutorial is written to cater to these target audiences:

- software engineers wary of venturing into quantum because of scary, spooky circuits and gate-level spaghetti code,

- researchers wondering whether this is the shiny new tool that will allow you to finally implement your clever new block-encoding approach (the answer to this is yes, by the way).

If you are a part of any or both of these groups (or other, for that matter) you've arrived at the right place. 

Furthermore, if you felt directly addressed as a part of the latter target audience, 
this tutorial will also teach you to add that one missing quantum resource analysis figure that is missing from your upcoming paper 
(whose conference submission deadline is, as always, breathing down your neck).

Quantum Linear Algebra has been a rapidly explored field in the recent years, with Quantum Signal Processing being at the center of it all as an unifying model to quantum algorithms. 
While the theory is extremely elegant and surprisingly simple (depending of the amount of exposure to it, I guess), 
moving from mathematical expressions and circuit schematics on paper (or your favorite e-reader) to a functioning quantum circuit has historically been a manual, 
rather complex labor. Qrisp's :ref:`BlockEncoding` class changes that, allowing to program quantum linear algebra as easily as you handle arrays in Numpy, 
with the compiler handling all the heavy lifting of the ancilla management of the underlying circuit (aah, scary word) construction behind the taken abstractions.

This is the central reference hub of a two part **Quantum Linear Algebra with Qrisp** schnellkurs (that is German for "quick course", in case you were wondering):

- BlockEncoding class 101: Block encoding, LCU, resource estimation

- BlockEncoding class 201: Qubitization, Chebyshev polynomials, QSP

BlockEncoding class 101: Block encoding, LCU, resource estimation
-----------------------------------------------------------------
In classical computer science, we take for granted the ability to invert or exponentiate matrices. 
In quantum computing, nature demands unitarity: all operations must be reversible and preserve the norm (represented by matrices $U$ such that $U^\dagger U = \mathbb{I}$).

But the matrices that matter, like Hamiltonians describing the energy of a molecule—are rarely unitary. 
To bridge this gap, we use Block Encodings to "embed" these non-unitary matrices into the top-left corner of a larger unitary system.

In this tutorial, you will learn to build these embeddings from scratch and manipulate them with high-level syntax.

.. list-table::
   :widths: 35 30 35
   :header-rows: 1

   * - Method
     - Purpose
     - Typical Use Case
   * - :meth:`.from_array() <qrisp.block_encodings.BlockEncoding.from_array>`
     - Encodes a NumPy matrix
     - Quick prototyping of small, specific matrices
   * - :meth:`.from_operator() <qrisp.block_encodings.BlockEncoding.from_operator>`
     - Encodes a Hamiltonian
     - Physics and Chemistry simulations
   * - :meth:`.from_lcu() <qrisp.block_encodings.BlockEncoding.from_lcu>`
     - Custom weighted sum of unitaries
     - High-efficiency implementation
   * - :meth:`.resources() <qrisp.block_encodings.BlockEncoding.resources>`
     - Estimates gate counts, depth, qubits
     - Resource analysis and benchmarking
   * - :meth:`.apply() <qrisp.block_encodings.BlockEncoding.apply>`
     - Applies a matrix to a quantum state
     - NISQ hardware with manual post-selection
   * - :meth:`.apply_rus() <qrisp.block_encodings.BlockEncoding.apply_rus>`
     - Applies a matrix to a quantum state
     - Fault-tolerant / Repeat-Until-Success logic
   * - Operators: ``+``, ``-``, ``*``, ``@``, ``.kron()``
     - Algebraic arithmetic on encodings
     - Constructing complex composite systems

BlockEncoding class 201: Qubitization, Chebyshev polynomials, QSP
-----------------------------------------------------------------
Once you have your matrix $A$ block-encoded, the real fun begins. Simply having $A$ isn't enough; you usually want to do something to it. 
You might want to compute $e^{-iAt}$ to simulate a physical system, or find $A^{-1}$ to solve a massive system of linear equations.

Classically, we use Taylor series or polynomial approximations. Quantumly, we use Quantum Signal Processing (QSP). 
This tutorial takes you from basic embeddings to "Quantum Walks" via a technique called Qubitization.

We will explore how to block-encode Chebyshev polynomials to perform near-optimal transformations. 
You’ll learn the Quantum Lanczos method for ground-state energy estimation and the Childs-Kothari-Somma (CKS) algorithm for linear systems.

Most importantly, we will show how the BlockEncoding class leverages QSP to make high-level operations as simple as a single method call. 
In Qrisp, performing arbitrary polynomial transformations is done via :meth:`.poly() <qrisp.block_encodings.BlockEncoding.poly>`, solving linear systems becomes :meth:`.inv() <qrisp.block_encodings.BlockEncoding.inv>`, 
and Hamiltonian simulation is achieved with :meth:`.sim() <qrisp.block_encodings.BlockEncoding.sim>`. No gate-level manual labor required; just clean, functional, and qrispy quantum linear algebra.

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Method
     - Purpose
     - Mathematical Basis
   * - :meth:`.from_array() <qrisp.block_encodings.BlockEncoding.qubitization>`
     - Encodes $A$ into a "walk operator" $W$
     - :ref:`reflection`
   * - :meth:`.chebyshev() <qrisp.block_encodings.BlockEncoding.chebyshev>`
     - Computes the $k$-th Chebyshev polynomial $T_k(A)$
     - Iterative application of $W^k$
   * - :meth:`.poly() <qrisp.block_encodings.BlockEncoding.poly>`
     - Applies an arbitrary polynomial transformation $P(A)$
     - `GQSP <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_, `GQET <https://arxiv.org/pdf/2312.00723>`_
   * - :meth:`.inv() <qrisp.block_encodings.BlockEncoding.inv>`
     - Solves the linear system $Ax = b$
     - `GQSP <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_, $1/x$ Chebyshev approximation
   * - :meth:`.sim() <qrisp.block_encodings.BlockEncoding.sim>`
     - Simulates Hamiltonian evolution $e^{-iHt}$
     - `GQSP <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_, Jacobi-Anger expansion (Bessel functions)

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   BE_vol1.ipynb
   BE_vol2.ipynb