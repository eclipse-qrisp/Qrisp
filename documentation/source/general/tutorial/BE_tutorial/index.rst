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

   B = sum(X(i)*X(i+1) + Y(i)*Y(i+1) + Z(i)*Z(i+1) for i in range(3))

   epsilon = 0.01
   kappa = np.linalg.cond(A)

   def operand_prep():
      return QuantumFloat(2)

   BE_A = BlockEncoding.from_array(A)
   BE_B = B.pauli_block_encoding()

   B_C = BE_A.poly([1, 1, -2]) + BE_B.inv(epsilon, kappa)

   gate_count = B_C.resources(operand_prep)()
   print(gate_count)

   # {'t': 1578, 'h': 1168, 's': 60, 'cx': 4125, 'x': 528, 't_dg': 2084, 's_dg': 60, 'cz': 186, 'ry': 2, 'u3': 466, 'p': 84, 'gphase': 62, 'cy': 178}
   

Whether you’re a software engineer wary of venturing into quantum because of scary, spooky circuits and gate-level spaghetti code, or a researcher wondering whether this is the shiny new tool that will allow you to finally implement your clever new block-encoding approach to block encoding (the answer to this is yes, by the way), you've arrived at the right place. If you felt directly addressed as a part of the latter target audience, this tutorial will also teach you to add that one missing quantum resource analysis figure that is missing from your upcoming paper (whose conference submission deadline is, as always, breathing down your neck).

Quantum Linear Algebra has been a rapidly explored field in the recent years, with Quantum Signal Processing being at the center of it all as an unifying model to quantum algorithms. While the theory is extremely elegant and surprisingly simple (depending of the amount of exposure to it, I guess), moving from mathematical expressions and circuit schematics on paper (or your favorite e-reader) to a functioning quantum circuit has historically been a manual, rather complex labor. Qrisp's BlockEncoding class changes that, allowing to program quantum linear algebra as easily as you handle arrays in Numpy, with the compiler handling all the heavy lifting of the ancilla management of the underlying circuit (aah, scary word) construction behind the taken abstractions.

This is the central reference hub of a two part **Quantum Linear Algebra with Qrisp** schnellkurs (that is German for "quick course", in case you were wondering):

- BlockEncoding class 101: Block encoding, LCU, resource estimation

- BlockEncoding class 201: Qubitization, Chebyshev polynomials, QSP

And yes, since this question has the probability of 0.82 to appear as an intrusive thought, let us already answer that with Qrisp you can do all of the below:

- compile circuits;

- perform resource estimation;

- simulate (with or without real-time measurements utilizing the repeat until success feature); and

- apply post-processing after having run these compiled circuits on existing hardware.

With this answered, let's get into the outline of this three part series!

BlockEncoding class 101: Block encoding, LCU, resource estimation
-----------------------------------------------------------------
In classical computer science, we take for granted the ability to invert or exponentiate matrices. In quantum computing, nature demands unitarity: all operations must be reversible and preserve the norm (represented by matrices $U$ such that $U^\dagger U = \mathbb{1}$).

But the matrices that matter, like Hamiltonians describing the energy of a molecule—are rarely unitary. To bridge this gap, we use Block Encodings to "embed" these non-unitary matrices into the top-left corner of a larger unitary system.

In this tutorial, you will learn to build these embeddings from scratch and manipulate them with high-level syntax.

| Method | Purpose | Typical Use Case |
| :--- | :--- | :--- |
| ``.from_array(arr)`` | Encodes a NumPy matrix | Quick prototyping of small, specific matrices. |
| ``.from_operator(op)`` | Encodes a Hamiltonian | Physics and Chemistry simulations. |
| ``.from_lcu(coeffs, unitaries)`` | Custom weighted sum of unitaries | High-efficiency hardware implementation. |
| ``.resources(qv)`` | Estimates gate counts and depth | Resource analysis and benchmarking. |
| ``.apply(qv)`` | Adds gates to the circuit | NISQ hardware with manual post-selection. |
| ``.apply_rus(prep_func)`` | Deterministic matrix application | Fault-tolerant / Repeat-Until-Success logic. |
| ``+``, ``-``, ``*``, ``@``, ``.kron()`` | Algebraic arithmetic on encodings | Constructing complex composite systems. |

BlockEncoding class 201: Qubitization, Chebyshev polynomials, QSP
-----------------------------------------------------------------
Once you have your matrix $A$ block-encoded, the real fun begins. Simply having $A$ isn't enough; you usually want to do something to it. You might want to compute $e^{-iAt}$ to simulate a physical system, or find $A^{-1}$ to solve a massive system of linear equations.

Classically, we use Taylor series or polynomial approximations. Quantumly, we use Quantum Signal Processing (QSP). This tutorial takes you from basic embeddings to "Quantum Walks" via a technique called Qubitization.

We will explore how to block-encode Chebyshev polynomials to perform near-optimal transformations. You’ll learn the Quantum Lanczos method for ground-state estimation and the Childs-Kothari-Somma (CKS) algorithm for linear systems.

Most importantly, we will show how the BlockEncoding class leverages QSP to make high-level operations as simple as a single method call. In Qrisp, performing arbitrary polynomial transformations is done via .poly, solving linear systems becomes .inv, and Hamiltonian simulation is achieved with .sim. No gate-level manual labor required; just clean, functional, and qrispy quantum linear algebra.

| Method | Purpose | Mathematical Basis |
| :--- | :--- | :--- |
| ``.qubitization()`` | Transforms $A$ into a "walk operator" $W$ | Interleaved ``reflection`` + ``qswitch`` |
| ``.chebyshev(k)`` | Computes the $k$-th Chebyshev polynomial $T_k$ | Iterative application of $W^k$ (``.qubitization``) |
| ``.poly(coeffs)`` | Applies an arbitrary polynomial transformation $P(A)$ | GQSP |
| ``.inv(eps, kappa)`` | Solves the linear system $Ax = b$ | $1/x$ polynomial approximation |
| ``.sim(t, N)`` | Simulates Hamiltonian evolution $e^{-iHt}$ | Jacobi-Anger expansion (Bessel functions) |