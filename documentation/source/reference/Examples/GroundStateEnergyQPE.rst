.. _GroundStateEnergyQPE:

Ground State Energy with QPE
============================

.. currentmodule:: qrisp


For a given Hamiltonian $H$ and an eigenstate $\ket{\lambda_j}$ (e.g., the ground state $\ket{\lambda_0}$), the goal is to find the eigenvalue $E_j$ such that

$$e^{iH}\\ket{\\lambda_j} = e^{iE_j}\\ket{\\lambda_j}$$

This is achieved by applying quantum phase estimation to the unitary $U=e^{iH}$ with the initial state $\ket{\lambda_j}$. 
With the equation $e^{iE_j}=e^{2\pi i\theta_j}$, the energy $E_j$ can be calculated from the resulting phase $\theta_j\in [0,1)$.
In practice, the exact eigenstate $\lambda_j$ is unknown, and one uses an approximation $\ket{\psi_0}$ which can be expressed in the eigenbasis of the Hamiltonian as

$$\\ket{\\psi_0}=\\sum_j a_j\\ket{\\lambda_j}$$

If the approximation $\ket{\psi_0}$ is sufficently close to the eigenstate $\ket{\lambda_j}$, the correct phase, and hence the eigenvalue $E_j$, is obtained with high probability.


Example Hydrogen
================

We caluclate the ground state energy of the electronic Hamiltonian for the Hydrogen molecule with :ref:`Quantum Phase Estimation <QPE>`. 

Utilizing symmetries, one can find a reduced two qubit Hamiltonian for the Hydrogen molecule. Frist, we define the Hamiltonian, and compute the 
ground state energy classically.

::

    from qrisp import QuantumVariable, x, QPE
    from qrisp.operators import X,Y,Z
    import numpy as np

    # Hydrogen (reduced 2 qubit Hamiltonian)
    H = -1.05237325 + 0.39793742*Z(0) -0.39793742*Z(1) -0.0112801*Z(0)*Z(1) + 0.1809312*X(0)*X(1)
    E0 = H.ground_state_energy()
    print(E0)
    # Yields: -1.85727502928823

The $\ket{10}$ state provides a good approximation of the ground state:

::

    # ansatz state
    qv = QuantumVariable(2)
    x(qv[0])
    E1 = H.get_measurement(qv)
    print(E1)
    # Yields: -1.83858104676077

In the following, we utilize the :meth:`trotterization <qrisp.operators.qubit.QubitOperator.trotterization>` method of the :ref:`QubitOperator` to obtain a function **U** that applies **Hamiltonian Simulation**
via Trotterization. If we start in a state that is close to the ground state and apply :ref:`QPE`, we get an estimate of the ground state energy.
As the results of the phase estimation are modulo :math:`2\pi` and the searched for eigenvalue is between $-2\pi$ and 0, we subtract :math:`2\pi`.

::

    U = H.trotterization(forward_evolution=False)

    qpe_res = QPE(qv,U,precision=10,kwargs={"steps":3},iter_spec=True)

    results = qpe_res.get_measurement()    
    sorted_results= dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    phi = list(sorted_results.items())[0][0]
    E_qpe = 2*np.pi*(phi-1) # Results are modulo 2*pi, therefore subtract 2*pi 
    print(E_qpe)
    # Yields: -1.8591847149174