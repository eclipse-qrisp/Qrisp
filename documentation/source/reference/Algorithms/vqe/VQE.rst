.. _VQE:

Variational Quantum Eigensolver
===============================

.. currentmodule:: qrisp.vqe

.. toctree::
   :hidden:
   
   VQEProblem
   VQEBenchmark
   VQEImplementations
   JaspVQE

This modules implements the `Variational Quantum Eigensolver (VQE) <https://arxiv.org/pdf/2111.05176>`_. 

The VQE is a hybrid quantum-classical algorithm for finding eigenvalues of a quantum Hamiltonian. 
It is an alternative to pure quantum algorithms such as quantum phase estimation that have higher requirements on quantum hardware.
The method has been be applied to various problems in quantum chemistry and quantum physics.
A VQE problem is given by:

- A quantum :ref:`QubitOperator` $$H=\\sum\\limits_{j}\\alpha_jP_j$$ where each $P_j=\prod_i\sigma_i^j$ is a Pauli product, and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.
- A parameter dependend quantum state (ansatz) $$\\ket{\\psi(\\theta)}=U(\\theta)\\ket{\\psi_0}$$ for an initial state $\ket{\psi_0}$.
  The unitary $U(\theta)$ consists of $p$ layers
  $$U(\\theta)=\\prod\\limits_{l=1}^{p}\\tilde{U}(\\theta_l)$$
  each depending on $m$ real parameters, i.e., $\theta_l=(\theta_{l,1},\dotsc,\theta_{l,m})$.

Then the minimum eigenvalue $E_0$ of the Hamiltonian $H$ satisfies
$$E_0\\leq E(\\theta)=\\langle\\psi(\\theta)|H|\\psi(\\theta)\\rangle=\\sum\\limits_j\\alpha_j\\langle\\psi(\\theta)|P_j|\\psi(\\theta)\\rangle$$

The ansatz parameters are variationally optimized in order to minimize the expected value $E(\theta)$. This serves as approximation for the ground state energy $E_0$.

The central data structure of the VQE module is the :ref:`VQEProblem` class.

:ref:`VQEProblem`
------------------

The :ref:`VQEProblem` class is like a blueprint for a implementing VQE for a specific problem instance we're trying to solve. When we create an instance of this class, we need to provide three things:

- Hamiltonian : A :ref:`QubitOperator` for which the ground state energy is to be determined.
- Ansatz function: A function that implements the unitary $\tilde U(\theta)$ corresponding to one layer of the ansatz.
- Number of parameters $m$ per layer.

Apart from the basic three ingredients mentioned above, some problems require the specification of the initial state. This can be achieved using the :meth:`.set_init_function <qrisp.vqe.VQEProblem.set_init_function>` method.

The :meth:`.run <qrisp.vqe.VQEProblem.run>` method prepares the initial state, applies $p$ layers of ansatz, and compiles a quantum circuit. Subsequently, the optimization algorithm is executed and the expected value of the Hamiltonian with respect to the optimized circuit is returned.

For benchmarking, we provide the :meth:`.benchmark <qrisp.vqe.VQEProblem.benchmark>` method, which allows you to collect performance data about your implementation.

Additionally, a circuit can be pretrained with the method :meth:`.train_function <qrisp.vqe.VQEProblem.train_function>` . This allows preparing a new :ref:`QuantumVariable` with already optimized parameters, such that no new optimization is conducted.
   
:ref:`VQEBenchmark`
--------------------

As an approximation algorithm, benchmarking various aspects such as solution quality and execution cost is a central question for VQE. The results of the benchmarks of a :ref:`VQEProblem` are represented in a class called :ref:`VQEBenchmark`. This enables convenient evaluation of several important metrics as well as visualization of the results.


Collection of implemented problem instances
-------------------------------------------
   
The following problem instances have already been successfully implemented using the Qrisp framework:

.. list-table::
   :widths: 45 45 10
   :header-rows: 1

   * - PROBLEM INSTANCE
     - ANSATZ TYPE
     - IMPLEMENTED IN QRISP
   * - :ref:`Heisenberg model <VQEHeisenberg>`
     - Hamiltonian Variational Ansatz
     -    ✅
   * - :ref:`Electronic structure <VQEElectronicStructure>`
     - QCCSD
     -    ✅ 