.. _GroundStateEnergyQPE:

Ground State Energy with QPE
============================

.. currentmodule:: qrisp

Example Hydrogen
================

We caluclate the ground state energy of the Hydrogen molecule with :ref:`Quantum Phase Estimation <QPE>`. 

Utilizing symmetries, one can find a reduced two qubit Hamiltonian for the Hydrogen molecule. Frist, we define the Hamiltonian, and compute the 
ground state energy classically.

::

    from qrisp import QuantumVariable, x, QPE
    from qrisp.operators.pauli.pauli import X,Y,Z
    import numpy as np

    # Hydrogen (reduced 2 qubit Hamiltonian)
    H = -1.05237325 + 0.39793742*Z(0) -0.39793742*Z(1) -0.0112801*Z(0)*Z(1) + 0.1809312*X(0)*X(1)
    E0 = H.ground_state_energy()
    # Yields: -1.85727502928823

In the following, we utilize the ``trotterization`` method of the :ref:`QubitOperator` to obtain a function **U** that applies **Hamiltonian Simulation**
via Trotterization. If we start in a state that is close to the ground state and apply :ref:`QPE`, we get an estimate of the ground state energy.

::

    # ansatz state
    qv = QuantumVariable(2)
    x(qv[0])
    E1 = H.get_measurement(qv)
    E1
    # Yields: -1.83858104676077


We calculate the ground state energy with quantum phase estimation. As the results of the phase estimation are modulo :math:`2\pi`, we subtract :math:`2\pi`.

::

    U = H.trotterization()

    qpe_res = QPE(qv,U,precision=10,kwargs={"steps":3},iter_spec=True)

    results = qpe_res.get_measurement()    
    sorted_results= dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    phi = list(sorted_results.items())[0][0]
    E_qpe = 2*np.pi*(phi-1) # Results are modulo 2*pi, therefore subtract 2*pi 
    E_qpe
    # Yields: -1.8591847149174