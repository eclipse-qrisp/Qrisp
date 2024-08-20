.. _phase_polynomials:

Phase Polynomial Tools
======================

These methods facilitate the efficient application of diagonal Hamiltonians defined by polynomials.
That is, they implement transformations of the form

$$\\ket{x}\\rightarrow e^{itH}\\ket{x}$$

where the Hamiltonian $H$ is of the form $H=\sum_xP(x)\ket{x}\bra{x}$ for a polynomial $P(x)$.

These methods can be utilized, e.g., as part of the :ref:`Quantum Approximate Optimization Algorithm <QAOA101>`:
In many cases the objective function of an optimization problem is given by a polynomial. The phase separation operator within QAOA then requires 
the application of phases corresponding to the objective function values.

The application of general Hamiltonian functions can be achieved in the following ways:

- The :ref:`first method <DiagonalHamiltonianApplication>` consists of computing the Hamiltonian function on all outcome labels of the input QuantumVariables, and subsequently applying the classically computed phases using logic synthesis. While this enables a convenient application of arbitray Hamiltonian functions, it does not scale to large systems as all phases have to be computed classically.

- The second method is based on evaluating the Hamiltonian function on its input QuantumVariables in superposition into an auxiliary variable of type QuantumFloat, subsequently applying the phases on the auxiliary variable, and finally uncomputing the auxiliary variable. This approch requires additional qubits and performing arithmetic on the quantum computer. Therefore, it introduces an additional burden on simulators, and moreover it is not suitable for NISQ devices.

In the special case where the Hamiltonian function is given by a polynomial, its application can be achieved by employing (multi-) controlled phase gates accordingly. 
The following functions facilitate the efficient application of such phase polynomials.

.. currentmodule:: qrisp

.. autosummary::
   :toctree: generated/
   
   app_sb_phase_polynomial
   app_phase_polynomial