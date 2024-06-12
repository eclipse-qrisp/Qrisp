Phase polynomial tools
======================

These functions facilitate convenient and efficient application of diagonal Hamiltonians that are given by (multivariate) polynomials. 
The application of such diagonal Hamiltonians can be utilized, e.g., as part of the :ref:`Quantum Approximate Optimization Algorithm <QAOA101>`:
In many cases the objective function of optimization problems is given by a polynomial. The phase separation operator within QAOA then requires 
the application of phases corresponding to the objective function values.

The application of general diagonal Hamiltonian functions can be achieved in the following ways:

- The first method is computing the Hamiltonian on all outcome labels of the input QuantumVariables, and subsequently applying the classically computed phases using logic synthesis. 
While this facilitates a convenient application of arbitray diagonal Hamiltonian functions, it does not scale to large systems as all phases have to be computed classically.

- The second method is evaluating the Hamiltonian on its input QuantumVariables into an auxiliary variable of type QuantumFloat, subsequently applying the phases on the auxiliary variable, 
and finally uncomputing the auxiliary variable. This approch requires additional qubits and performing arithmetic on the quantum computer. 
Therefore, on the one hand it poses an additional burden for simulators, and on the other hand is not applicable on NISQ devices.

In the special case where the Hamiltonian function is given by a polynomial, its application can be achieved by employing (multi-) controlled phase gates accordingly. 
The following functions facilitate convenient and efficient application of such diagonal Hamiltonians that are given by (multivariate) polynomials.

.. currentmodule:: qrisp

.. autosummary::
   :toctree: generated/
   
   app_sb_phase_polynomial
   app_phase_polynomial