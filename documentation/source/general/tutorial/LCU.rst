.. _LCU_tutorial:

Linear Combination of Unitaries (LCU) primitive and its applications
====================================================================

If you’ve ever wondered how to bend the rules of quantum mechanics (without actually breaking them), you’re in the right place. This tutorial is your roadmap to the Linear Combination of Unitaries (LCU) protocol—a powerful tool that lets you simulate non-unitary operations by cleverly mixing unitaries, opening doors to advanced quantum algorithms and simulations.

Here’s what you’ll discover as you journey through this tutorial:

- We’ll start by demystifying the LCU protocol: why it’s needed and what problems it solves. You’ll learn the core ideas behind representing complex quantum operations as sums of simpler, unitary building blocks aka block encodings.

- Next, we’ll roll up our sleeves and see how LCU comes alive in Qrisp (Frankenstein intensifies). This section is hands-on: you’ll explore annotated code examples, understand the structure of Qrisp’s LCU implementation, and see how to prepare ancilla registers, orchestrate controlled unitaries, and interpret the results.

- We won't stop at just understanding LCU, but instead also use it as an algorithmic building block (primitive) to perform another algorithm. In our final section, we combine the strengths of Trotterization and LCU to unlock the Linear Combination of Hamiltonian Simulation (LCHS) protocol. Here, you’ll learn how to simulate functions of Hamiltonians—like $\cos(H)$.

If all goes well (if not, let us know about which parts should be elaborated upon further), you'll be motivated to apply LCU and extend it to tackle `Quantum Signal
Processing (QSP) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368>`_, and/or `Quantum Singular Value Transformation (QSVT) <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`_. Until the implementation in Qrisp, the proof to this is left for the reader (which is not a broad time-window).

Nothing more to say other than let's go!


LCU in theory
-------------

So, you want to perform operations that aren't strictly allowed by the quantum rulebook?
Enter the Linear Combination of Unitaries (LCU) protocol—a foundational quantum algorithmic primitive that lets you implement a non-unitary operator $A$ by cleverly expressing it as a weighted sum of unitary operators: 

.. math::
    A=\sum_i\alpha_iU_i

This is the quantum equivalent of ordering a custom pizza: you pick your favorite toppings (unitaries), assign them weights (coefficients), and hope the outcome is deliciously non-classical.

Core components
^^^^^^^^^^^^^^^

The LCU protocol works by embedding your non-unitary operator into a larger, unitary quantum circuit. The magic happens in three acts, known as block encoding:

- **PREPARE**: Prepares an ancilla quantum variable in a superposition encoding the normalized coefficients $\alpha_i\geq0$ of the target operator

.. math ::

        \mathrm{PREPARE}|0\rangle=\sum_i\sqrt{\frac{\alpha_i}{\lambda}}|i\rangle

- **SELECT**: Applies the unitary $U_i$ to the input state $\ket{\psi}$, controlled on the ancilla variable being in state $|i\rangle$.

.. math ::

    \mathrm{SELECT}|i\rangle|\psi\rangle=|i\rangle U_i|\psi\rangle

- **PREPARE**$^\dagger$: Applies the inverse prepartion to the ancilla.

Success condition
^^^^^^^^^^^^^^^^^

The LCU protocol is deemed successful only if the ancilla variable is measured in the $\ket{0}$ state, which occurs with a probability proportional to :math:`\frac{\langle\psi|A^{\dagger}A|\psi\rangle}{\lambda^2}` where $\lambda=\sum_i\alpha_i$.
This function does not perform the measurement; it returns the ancilla variable and the transformed target variable.

The approach you’ve just studied was pioneered by Nathan Wiebe, whose contributions have fundamentally shaped the field of quantum algorithm design, particularly in Hamiltonian simulation and quantum linear systems.

If you’re eager for more than equations and want intuition delivered with clarity, Nathan Wiebe’s YouTube seminar series is a goldmine. His channel is packed with lucid explanations of quantum primitives and routines, making even the most complex ideas accessible. The video below, authored by Wiebe himself, distills the essence of the LCU protocol and its applications—so whether you’re seeking a first encounter or a deeper understanding, this is a resource not to miss.

.. youtube:: irMKrOIrHP4
LCU in Qrisp
------------

Trotterization + LCU = LCHS
---------------------------