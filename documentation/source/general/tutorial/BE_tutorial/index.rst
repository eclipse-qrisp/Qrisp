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

This is the central reference hub of a three part **Quantum Linear Algebra with Qrisp** series.

And yes, since this question has the probability of 0.82 to appear as an intrusive thought, let us already answer that with Qrisp you can do all of the below:

- compile circuits;

- perform resource estimation;

- simulate (with or without real-time measurements utilizing the repeat until success feature);

- apply post-processing after having run these compiled circuits on hardware of your choice.

With this answered, let's get into the outline of this three part series!

Roadmap: From theory through construction to application
--------------------------------------------------------

This series is divided into three parts designed to take you from "What is an LCU?" to "I just solved a 1D-Laplacian linear system in three lines of code."

Part 0: Theoretical deep dive: From Block Encodings to Quantum Signal Processing (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before exploring the BlockEncoding class, we'll catch up on (just enough) theory to satisfy even the most rigorous academic. We'll dive into the math and connect the dots between a recent stream of publications (we're talking about papers who are all still less than a decade old!). 

This tutorial will cover:

- Block Encodings: The "Top-Left" trick. How to embed a non-unitary matrix $A$ into a unitary $U$ so it can actually live on a quantum circuit.

- Qubitization: The "Walk." By pairing $U$ with a reflection, we create a walk operator that encodes the eigenvalues of $A$ as controllable rotations. This is where we meet our best friends: Chebyshev polynomials. It's exactly these Chebyshevs that provide an optimal basis for bounded polynomial approximations. Take an inverse, for a popular example, to also entertain the QML fans among the readers and immediately plant the idea of solving linear systems and applying them as the basis for Quantum Support Vector Machines. I guess this kind of non-subliminal messaging is just messaging, huh?

- Quantum Signal Processing (QSP): The "Transformation." How a sequence of single-qubit phases allows us to apply a polynomial $P(A)$—enabling everything from matrix inversion to Hamiltonian simulation.

- Generalized QSP (GQSP): The "Universal Solver." The final evolution that allows for non-Hermitian or asymmetric polynomial transformations. This is the base of a lot of the methods of the BlockEncoding class, and can also be used for spectral filtering, essentially making the quantum world your oyster.

With the results, lemmas, theorems, and corollaries from the past two years now converging, the theoretical pieces fell into place. Stepping a step away and observing the mosaic with the final pieces added allowed for seeing the bigger picture and provide the common denominator of all these papers, the block encoding, as a standalone programming abstraction in Qrisp.

Part I: From Theory to Code: Constructing a BlockEncoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the theoretical tutorial was too much for you, that's OK! We've designed the BlockEncoding class in a way that you don't even need to (although, it would definitely help) understand the underlying concepts, and just use the classical numpy-like syntax you're already for quantum (definitely not classical) linear algebra.

In this tutorial we focus on the "How." You will learn to generate block encodings from various sources:

- Constructors: Use ``.from_operator``, ``.from_array``, or the powerful ``.from_LCU`` (Linear Combination of Unitaries).

- Arithmetic with BlockEncodings: Perform "Quantum Algebra" using standard Pythonic syntax. Need to add two matrices? Just use ``A + B``. Want to scale an operator? Boom. ``0.5 * A``. Multiply two block encodings? ``A @ B``... You get the idea.

- Execution and Quantum Resource Analysis: Move from abstraction to reality. ``.apply()`` automatically synthesizes the gates (no hand weaving circuits, I promise!), handling all ancilla management. For simulation, ``.apply_rus`` invokes a Repeat-Until-Success procedure returning the result only if the all ancillas are measured in $\ket{0}$. To peek under the hood and see exactly how many gates of all kind, you can use the ``.resources`` method.

Part II: Advanced Applications and executing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With getting familiar with the basic functionalities of the BlockEncoding class, we now shift up a gear and tackle some of the recent, relevant open questions. We explore and showcase the methods that allow you to use the BlockEncodings for:

- Qubitization: Leverage .qubitization and .chebyshev to walk through Krylov subspaces with optimal query complexity.

- Solving Linear Systems: Perform matrix inversion on the block encoding of your construction with ``.inv``.

- Polynomial Transformations: Apply an arbitrary polynomial transformation of a matrix in either the polynomial or Chebyshev basis using ``.poly``. This is also useful to perform GQSP filtering, which is a topic of a separate tutorial.

- Hamiltonian simulation: Want to perform a Hamiltonian simulation with your block encoding? Sure, by all means... ``.sim`` can help with that.

Until now implementing a non-unitary matrix and/or a custom block encoding tied to one specific case on a quantum computer meant manually calculating normalization factors, managing ancilla registers, and (in most cases) hand-weaving "Prepare" and "Select" oracles. Let's just not do that and instead start coding quantum linear algebra using the BlockEncoding class.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   BE_vol1.ipynb
   BE_vol2.ipynb

